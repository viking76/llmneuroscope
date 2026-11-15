from __future__ import annotations

import json
import re

from deepeval.metrics import BaseMetric
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase

JUDGE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


class LocalInstructJudge(DeepEvalBaseLLM):
    def __init__(self, model_name: str = JUDGE_MODEL):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16)
        self.model.eval()
        self.model_name_ = model_name

    def load_model(self):
        return self.model

    def get_model_name(self) -> str:
        return self.model_name_

    def generate(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        encoded = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
        with self.torch.no_grad():
            output = self.model.generate(**encoded, max_new_tokens=100, do_sample=False)
        return self.tokenizer.decode(
            output[0][encoded["input_ids"].shape[1] :], skip_special_tokens=True
        )

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)


class LocalCoherenceMetric(BaseMetric):
    threshold = 0.5

    def __init__(self, judge: LocalInstructJudge, name: str = "cohérence locale"):
        self.judge = judge
        self.name = name

    def measure(self, test_case: LLMTestCase) -> float:
        prompt = (
            "You are a strict evaluator. Rate the coherence and fluency of this text "
            "on a scale from 1 (incoherent) to 5 (perfectly coherent).\n\n"
            f"Text: {test_case.actual_output}\n\n"
            'Respond ONLY with a JSON object: {"score": <1-5>, "reason": "<one sentence>"}'
        )
        raw = self.judge.generate(prompt)
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            raise ValueError(f"Réponse du juge sans JSON: {raw[:120]}")
        parsed = json.loads(match.group(0))
        self.score = float(parsed["score"]) / 5.0
        self.reason = parsed.get("reason", "")
        self.success = self.score >= self.threshold
        return self.score
