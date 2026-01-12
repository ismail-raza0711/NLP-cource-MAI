# src/llm_client_transformers.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class HFClientConfig:
    model_name_or_path: str
    device: str = "auto"  # "auto" | "cpu" | "cuda"
    max_new_tokens: int = 140
    temperature: float = 0.9
    top_p: float = 0.95
    repetition_penalty: float = 1.05

    trust_remote_code: bool = False
    use_safetensors: bool = True
    local_files_only: bool = True
    revision: str = "main"


class HFChatClient:
    def __init__(self, cfg: HFClientConfig):
        self.cfg = cfg

        if cfg.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = cfg.device

        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_name_or_path,
            use_fast=True,
            local_files_only=cfg.local_files_only,
            revision=cfg.revision,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = torch.float16 if self.device == "cuda" else torch.float32

        # Key: force local resolution; do not trigger hub conversion checks
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name_or_path,
            dtype=dtype,
            trust_remote_code=cfg.trust_remote_code,
            use_safetensors=cfg.use_safetensors,
            local_files_only=cfg.local_files_only,
            revision=cfg.revision,
        )

        self.model.to(self.device)
        self.model.eval()

    def _build_prompt(self, messages: List[Dict[str, str]]) -> str:
        parts = []
        for m in messages:
            role = m.get("role", "user")
            content = (m.get("content") or "").strip()
            if not content:
                continue
            if role == "system":
                parts.append(f"system: {content}\n")
            elif role == "assistant":
                parts.append(f"assistant: {content}\n")
            else:
                parts.append(f"user: {content}\n")
        parts.append("assistant: ")
        return "".join(parts)

    @torch.inference_mode()
    def chat(self, messages: List[Dict[str, str]]) -> str:
        prompt = self._build_prompt(messages)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to(self.device)

        out = self.model.generate(
            **inputs,
            do_sample=True,
            max_new_tokens=self.cfg.max_new_tokens,
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
            repetition_penalty=self.cfg.repetition_penalty,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        decoded = self.tokenizer.decode(out[0], skip_special_tokens=True)

        if "assistant:" in decoded:
            reply = decoded.split("assistant:")[-1].strip()
        else:
            reply = decoded[len(prompt):].strip()

        for stop in ["\nuser:", "\nsystem:", "\nassistant:"]:
            if stop in reply:
                reply = reply.split(stop)[0].strip()

        return reply
