import asyncio
import time
from typing import List, Dict, Any

class DynamicBatcher:
    """
    Micro-batches concurrent inference requests within a dynamic time window
    to maximize hardware utilization across multiple callers.
    """
    def __init__(self, engine, max_batch_size: int = 4, max_wait_ms: float = 20.0):
        self.engine = engine
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms / 1000.0  # convert to seconds
        self.queue: List[Dict[str, Any]] = []
        self.lock = asyncio.Lock()
        self._batch_task: asyncio.Task = None

    async def enqueue(self, prompt: str, max_tokens: int = 40) -> str:
        """
        Enqueues a prompt request and awaits completion of the batched forward pass.
        """
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        
        async with self.lock:
            self.queue.append({
                "prompt": prompt,
                "max_tokens": max_tokens,
                "future": future,
                "time": time.time()
            })
            
            # If batch capacity is reached, process immediately
            if len(self.queue) >= self.max_batch_size:
                loop.create_task(self._process_queue())
            elif len(self.queue) == 1:
                # First request in current window: schedule delayed dispatch
                loop.call_later(self.max_wait_ms, lambda: asyncio.create_task(self._process_queue()))

        return await future

    async def _process_queue(self):
        async with self.lock:
            if not self.queue:
                return
            batch = self.queue[:self.max_batch_size]
            self.queue = self.queue[self.max_batch_size:]

        # Execute generation for batched requests
        for item in batch:
            if not item["future"].done():
                response_tokens = []
                messages = [{"role": "user", "content": item["prompt"]}]
                for tok in self.engine.generate_chat_stream(
                    messages, 
                    max_new_tokens=item["max_tokens"], 
                    use_retrieval=False
                ):
                    response_tokens.append(tok)
                item["future"].set_result("".join(response_tokens))