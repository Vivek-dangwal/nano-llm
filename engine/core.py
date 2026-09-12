"""
Mini LLM Inference Engine -- core.py

NEW in this version: officeholder detection now handles more phrasings.
Previously only "president of India" style (office THEN entity) matched.
Now also handles:
  - possessive form: "India's current president" (entity THEN office)
  - multiple offices in one question: "India's current PM and president"
    resolves BOTH facts via Wikidata, not just the first one matched.

Full changelog (earlier fixes, still included):
1. Repetition penalty only tracks tokens generated in THIS response.
2. System prompt added to anchor behavior and reduce topic drift.
3. Per-channel INT8 quantization instead of per-tensor.
4. Fixed a dead stop-token string check.
5. Swapped SmolLM2-135M -> Qwen2.5-0.5B-Instruct.
6. Lightweight Wikipedia fact-lookup (free, no key), skipped for greetings.
7. Forces PyTorch to use all available CPU threads.
8. torch.inference_mode(), trimmed default max_new_tokens.
9. Debug timing prints for retrieval / prefill / decode speed.
10. Wikidata officeholder resolution for "current president/PM/CEO" style
    questions, now handling both phrasing orders and multi-office queries.
"""

import os
import re
import time
import torch
import torch.nn as nn
import requests
from typing import Generator, List, Dict, Optional, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful, direct assistant. Answer only the question the user just asked, "
    "in as few words as needed. Do not bring up unrelated topics from earlier in the "
    "conversation. If you are not confident about a fact (like a price, a date, or a "
    "specific number), say you're not sure instead of guessing."
)

WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
WIKIDATA_API = "https://www.wikidata.org/w/api.php"

_SKIP_RETRIEVAL_PHRASES = {
    "hi", "hello", "hey", "thanks", "thank you", "ok", "okay", "bye",
    "goodbye", "yes", "no", "cool", "nice", "great",
}

_OFFICE_PROPERTY_MAP = {
    "president": ("P35", "head of state"),
    "king": ("P35", "head of state"),
    "queen": ("P35", "head of state"),
    "monarch": ("P35", "head of state"),
    "prime minister": ("P6", "head of government"),
    "pm": ("P6", "head of government"),
    "chancellor": ("P6", "head of government"),
    "ceo": ("P169", "chief executive officer"),
    "chief executive": ("P169", "chief executive officer"),
}

_OFFICE_KEYWORDS = sorted(_OFFICE_PROPERTY_MAP.keys(), key=len, reverse=True)
_OFFICE_ALTERNATION = "|".join(re.escape(k) for k in _OFFICE_KEYWORDS)
_OFFICE_KEYWORD_PATTERN = re.compile(rf"\b({_OFFICE_ALTERNATION})\b", re.IGNORECASE)

# "president of India" / "prime minister of Japan" -- office THEN entity
_ENTITY_PATTERN_OF = re.compile(
    rf"(?:{_OFFICE_ALTERNATION})\s+of\s+([A-Za-z][A-Za-z\s]*)", re.IGNORECASE
)
# "India's current president" / "India's PM and president" -- entity THEN office
_ENTITY_PATTERN_POSSESSIVE = re.compile(
    rf"([A-Za-z][A-Za-z\s]*?)'s\s+(?:current\s+)?(?:{_OFFICE_ALTERNATION})", re.IGNORECASE
)


def _should_attempt_retrieval(query: str) -> bool:
    q = query.strip().lower().rstrip("!.?")
    if not q or q in _SKIP_RETRIEVAL_PHRASES:
        return False
    if len(q.split()) <= 1:
        return False
    return True


def _detect_officeholder_queries(query: str) -> List[Tuple[str, str, str]]:
    """
    Returns a list of (property_id, office_label, entity_name) for every
    recognized office mentioned in the query -- so a question like
    "India's current PM and president" resolves BOTH facts, not just one.
    Returns an empty list if no entity name could be confidently found.
    """
    entity_name = None
    m = _ENTITY_PATTERN_OF.search(query)
    if m:
        entity_name = m.group(1)
    if not entity_name:
        m = _ENTITY_PATTERN_POSSESSIVE.search(query)
        if m:
            entity_name = m.group(1)
    if not entity_name:
        return []

    entity_name = entity_name.strip().rstrip("?.! ")
    if not entity_name or len(entity_name) > 40:
        return []

    results = []
    seen_properties = set()
    for match in _OFFICE_KEYWORD_PATTERN.finditer(query):
        office = match.group(1).lower()
        mapped = _OFFICE_PROPERTY_MAP.get(office)
        if mapped and mapped[0] not in seen_properties:
            seen_properties.add(mapped[0])
            results.append((mapped[0], mapped[1], entity_name))

    return results


def fetch_wikidata_officeholder(entity_name: str, property_id: str) -> Optional[str]:
    """Resolves one 'who currently holds position X for Y' fact via Wikidata."""
    try:
        search_resp = requests.get(
            WIKIDATA_API,
            params={
                "action": "wbsearchentities",
                "search": entity_name,
                "language": "en",
                "format": "json",
                "limit": 1,
            },
            timeout=3,
        )
        search_resp.raise_for_status()
        results = search_resp.json().get("search", [])
        if not results:
            return None
        entity_qid = results[0]["id"]

        claims_resp = requests.get(
            WIKIDATA_API,
            params={
                "action": "wbgetclaims",
                "entity": entity_qid,
                "property": property_id,
                "format": "json",
            },
            timeout=3,
        )
        claims_resp.raise_for_status()
        claims = claims_resp.json().get("claims", {}).get(property_id, [])
        if not claims:
            return None

        holder_qid = None
        for claim in claims:
            has_end_time = "P582" in claim.get("qualifiers", {})
            value_id = claim.get("mainsnak", {}).get("datavalue", {}).get("value", {}).get("id")
            if value_id and not has_end_time:
                holder_qid = value_id
                break
        if not holder_qid and claims:
            holder_qid = claims[0].get("mainsnak", {}).get("datavalue", {}).get("value", {}).get("id")
        if not holder_qid:
            return None

        label_resp = requests.get(
            WIKIDATA_API,
            params={
                "action": "wbgetentities",
                "ids": holder_qid,
                "props": "labels",
                "languages": "en",
                "format": "json",
            },
            timeout=3,
        )
        label_resp.raise_for_status()
        entities = label_resp.json().get("entities", {})
        return entities.get(holder_qid, {}).get("labels", {}).get("en", {}).get("value")
    except Exception:
        return None


def fetch_wikipedia_context(query: str, max_chars: int = 500) -> Optional[str]:
    """Free, no-key Wikipedia lookup -- combined search+summary in one request."""
    try:
        resp = requests.get(
            WIKIPEDIA_API,
            params={
                "action": "query",
                "generator": "search",
                "gsrsearch": query,
                "gsrlimit": 1,
                "prop": "extracts",
                "exintro": True,
                "explaintext": True,
                "format": "json",
            },
            timeout=3,
        )
        resp.raise_for_status()
        pages = resp.json().get("query", {}).get("pages", {})
        for page in pages.values():
            extract = page.get("extract", "").strip()
            if extract:
                return extract[:max_chars]
        return None
    except Exception:
        return None


class QuantizedLinear(nn.Module):
    """Per-CHANNEL symmetric INT8 linear layer (own scale per output neuron)."""
    def __init__(self, original_linear: nn.Linear):
        super().__init__()
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features

        w = original_linear.weight.data
        max_per_channel = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
        scale = max_per_channel / 127.0
        self.register_buffer("scale", scale)

        q_weight = torch.clamp(torch.round(w / scale), -128, 127).to(torch.int8)
        self.register_buffer("q_weight", q_weight)

        if original_linear.bias is not None:
            self.bias = nn.Parameter(original_linear.bias.data.clone())
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dequant_weight = self.q_weight.to(x.dtype) * self.scale.to(x.dtype)
        return nn.functional.linear(x, dequant_weight, self.bias)


def quantize_model_layers(model: nn.Module):
    for name, child in model.named_children():
        if isinstance(child, nn.Linear) and name != "lm_head":
            setattr(model, name, QuantizedLinear(child))
        else:
            quantize_model_layers(child)


class MiniLLMEngine:
    def __init__(
        self,
        model_id: str = MODEL_ID,
        quantize: bool = True,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        history_turns: int = 4,
        debug: bool = True,
    ):
        self.debug = debug

        num_threads = os.cpu_count() or 4
        torch.set_num_threads(num_threads)
        self._log(f"Using {num_threads} CPU threads (torch reports: {torch.get_num_threads()})")
        self._log(f"quantize={quantize}")

        self._log(f"Initializing engine with model: {model_id}")
        self.system_prompt = system_prompt
        self.history_turns = history_turns

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.stop_token_ids = {self.tokenizer.eos_token_id}
        for extra in ("<|im_end|>", "<|endoftext|>"):
            tid = self.tokenizer.convert_tokens_to_ids(extra)
            if tid is not None and tid != self.tokenizer.unk_token_id:
                self.stop_token_ids.add(tid)

        self._log("Loading base weights into RAM...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.float32,
            low_cpu_mem_usage=True,
        )

        if quantize:
            self._log("Applying custom per-channel INT8 quantization...")
            quantize_model_layers(self.model.model)

        self.model.eval()
        self._log("Engine ready for inference.")

    def _log(self, msg: str):
        if self.debug:
            print(f"[MiniLLMEngine] {msg}")

    def _build_prompt(
        self,
        messages: List[Dict[str, str]],
        retrieved_context: Optional[str] = None,
    ) -> str:
        clean_messages = []
        if self.system_prompt:
            clean_messages.append({"role": "system", "content": self.system_prompt})

        if retrieved_context:
            clean_messages.append({
                "role": "system",
                "content": (
                    "Reference information that may help answer the user's next "
                    "question (may or may not be relevant):\n"
                    f"{retrieved_context}\n"
                    "If it isn't relevant, ignore it and answer normally. Never "
                    "mention that you were given reference information."
                ),
            })

        turns = messages[-self.history_turns:] if self.history_turns > 0 else messages
        for msg in turns:
            role = msg.get("role", "user")
            content = msg.get("content", "").strip()
            if content and role in ("user", "assistant"):
                clean_messages.append({"role": role, "content": content})

        return self.tokenizer.apply_chat_template(
            clean_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def _apply_repetition_penalty(
        self,
        logits: torch.Tensor,
        response_token_ids: List[int],
        penalty: float,
    ) -> torch.Tensor:
        for token_id in set(response_token_ids):
            if logits[0, token_id] > 0:
                logits[0, token_id] /= penalty
            else:
                logits[0, token_id] *= penalty
        return logits

    def _pick_next_token(
        self,
        logits: torch.Tensor,
        temperature: float,
        top_p: float,
    ) -> torch.Tensor:
        if temperature <= 0.0:
            return torch.argmax(logits, dim=-1, keepdim=True)

        probs = torch.softmax(logits / temperature, dim=-1)
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)

        cutoff = cumulative > top_p
        cutoff[..., 1:] = cutoff[..., :-1].clone()
        cutoff[..., 0] = False
        sorted_probs[cutoff] = 0.0
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

        choice = torch.multinomial(sorted_probs, num_samples=1)
        return sorted_idx.gather(-1, choice)

    def _resolve_context(self, last_user_msg: str) -> Optional[str]:
        officeholder_queries = _detect_officeholder_queries(last_user_msg)
        if officeholder_queries:
            facts = []
            t0 = time.time()
            for property_id, office_label, entity_name in officeholder_queries:
                holder_name = fetch_wikidata_officeholder(entity_name, property_id)
                if holder_name:
                    facts.append(
                        f"{holder_name} currently holds the position of {office_label} "
                        f"for {entity_name.strip().title()}."
                    )
            self._log(f"[timing] Wikidata officeholder lookup(s): {time.time() - t0:.2f}s "
                      f"({len(facts)}/{len(officeholder_queries)} resolved)")
            if facts:
                return " ".join(facts) + " (Source: Wikidata)"
            # none resolved -- fall through to Wikipedia text lookup below

        if _should_attempt_retrieval(last_user_msg):
            t0 = time.time()
            context = fetch_wikipedia_context(last_user_msg)
            self._log(f"[timing] Wikipedia retrieval: {time.time() - t0:.2f}s "
                      f"({'found' if context else 'nothing found'})")
            return context

        return None

    def generate_chat_stream(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 150,
        repetition_penalty: float = 1.15,
        temperature: float = 0.0,
        top_p: float = 0.9,
        use_retrieval: bool = True,
    ) -> Generator[str, None, None]:
        t_start = time.time()

        retrieved_context = None
        if use_retrieval and messages:
            last_user_msg = messages[-1].get("content", "").strip()
            if last_user_msg:
                retrieved_context = self._resolve_context(last_user_msg)

        formatted_prompt = self._build_prompt(messages, retrieved_context)
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt")
        input_ids = inputs["input_ids"]
        self._log(f"[timing] Prompt length: {input_ids.shape[1]} tokens")
        response_token_ids: List[int] = []

        with torch.inference_mode():
            t_prefill = time.time()
            outputs = self.model(input_ids, use_cache=True)
            self._log(f"[timing] Prefill: {time.time() - t_prefill:.2f}s")
            past_key_values = outputs.past_key_values

            logits = outputs.logits[:, -1, :].clone()
            logits = self._apply_repetition_penalty(logits, response_token_ids, repetition_penalty)
            next_token_id = self._pick_next_token(logits, temperature, top_p)

            if next_token_id.item() in self.stop_token_ids:
                return

            response_token_ids.append(next_token_id.item())
            token_str = self.tokenizer.decode(next_token_id[0], skip_special_tokens=True)
            if token_str:
                yield token_str

            t_decode = time.time()
            for _ in range(max_new_tokens - 1):
                outputs = self.model(
                    input_ids=next_token_id,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                past_key_values = outputs.past_key_values

                logits = outputs.logits[:, -1, :].clone()
                logits = self._apply_repetition_penalty(logits, response_token_ids, repetition_penalty)
                next_token_id = self._pick_next_token(logits, temperature, top_p)

                if next_token_id.item() in self.stop_token_ids:
                    break

                response_token_ids.append(next_token_id.item())
                token_str = self.tokenizer.decode(next_token_id[0], skip_special_tokens=True)
                if token_str:
                    yield token_str

            decode_time = time.time() - t_decode
            n = len(response_token_ids)
            if n > 0 and decode_time > 0:
                self._log(f"[timing] Decode speed: {n / decode_time:.2f} tok/s "
                          f"({n} tokens, {decode_time:.2f}s)")
            self._log(f"[timing] TOTAL turn time: {time.time() - t_start:.2f}s\n")


if __name__ == "__main__":
    print("=== TEST 1: quantize=True ===")
    engine = MiniLLMEngine(quantize=True)
    for tok in engine.generate_chat_stream([{"role": "user", "content": "hello"}], max_new_tokens=20):
        print(tok, end="", flush=True)
    print("\n")

    print("=== TEST 2: quantize=False ===")
    engine2 = MiniLLMEngine(quantize=False)
    for tok in engine2.generate_chat_stream([{"role": "user", "content": "hello"}], max_new_tokens=20):
        print(tok, end="", flush=True)
    print("\n")