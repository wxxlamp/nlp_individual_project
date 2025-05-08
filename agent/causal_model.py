import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from agent.base_model import BaseModel, LABELS


class CausalModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained("gpt2")
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")

    def build_prompt(self, premise, hypothesis):
        return f"""Premise: {premise}
        Hypothesis: {hypothesis}
        Is this entailment, neutral, or contradiction? Answer:"""

    def lm_prob_distribution(self, premise, hypothesis):
        prompt = self.build_prompt(premise, hypothesis)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits[0, -1, :]  # 最后一个位置的logits
        probs = torch.softmax(logits, dim=-1)

        # 获取标签词的概率
        label_scores = {}
        for label in LABELS:
            token_id = self.tokenizer.encode(label, add_special_tokens=False)[0]
            label_scores[label] = probs[token_id].item()
        return max(label_scores, key=lambda k: label_scores[k])

    def lm_likelihood(self, premise, hypothesis):
        label_scores = {}
        for label in LABELS:
            # 构造完整Prompt（包含标签）
            full_prompt = self.build_prompt(premise, hypothesis) + " " + label
            inputs = self.tokenizer(full_prompt, return_tensors="pt")
            with torch.no_grad():
                outputs = self.model(**inputs, labels=inputs["input_ids"])
            log_likelihood = -outputs.loss * inputs["input_ids"].shape[1]
            label_scores[label] = log_likelihood.item()
        return max(label_scores, key=lambda k: label_scores[k])
