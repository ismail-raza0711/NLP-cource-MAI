# src/llm_client_llamacpp.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from llama_cpp import Llama


@dataclass
class LlamaCppConfig:
    model_path: str
    n_ctx: int = 4096
    n_threads: int = 8
    n_gpu_layers: int = 0  # 0 = CPU; set >0 if you enable GPU
    temperature: float = 0.8
    top_p: float = 0.95
    repeat_penalty: float = 1.10
    max_tokens: int = 140

    # IMPORTANT:
    # Use a chat format suitable for instruct models.
    # For Phi-3 instruct GGUF, "chatml" is usually a good default.
    chat_format: str = "chatml"


class LlamaCppChatClient:
    def __init__(self, cfg: LlamaCppConfig):
        self.cfg = cfg
        self.llm = Llama(
            model_path=cfg.model_path,
            n_ctx=cfg.n_ctx,
            n_threads=cfg.n_threads,
            n_gpu_layers=cfg.n_gpu_layers,
            chat_format=cfg.chat_format,
            verbose=False,
        )

    def chat(self, messages: List[Dict[str, str]]) -> str:
        """
        messages: [{"role":"system"|"user"|"assistant", "content": "..."}]
        """
        out = self.llm.create_chat_completion(
            messages=messages,
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
            repeat_penalty=self.cfg.repeat_penalty,
            max_tokens=self.cfg.max_tokens,
        )
        return (out["choices"][0]["message"]["content"] or "").strip()
