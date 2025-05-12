import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from agent import verbalizer
from agent.base_model import BaseModel, LABELS


class CausalModel(BaseModel):
    def __init__(self, model_name="openai-community/gpt2"):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def build_prompt(self, premise, hypothesis):
        return f"""Given a premise and a hypothesis, determine if their relationship is entailment, neutral, or contradiction. Only give one word of the answer.
            Premise: {premise} 
            Hypothesis: {hypothesis}
            Answer: """

    def lm_sampled_completion(self, premise, hypothesis):
        inputs = self.prepare_input(premise, hypothesis)

        outputs = self.model.generate(
            **inputs,
            max_length=1024,
            temperature=0.7,
            top_k=50,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return verbalizer.mapping(generated.split("Answer:")[-1])


    def lm_prob_distribution(self, premise, hypothesis):
        inputs = self.prepare_input(premise, hypothesis)
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits[0, -1, :]  # 最后一个位置的logits
        probs = torch.softmax(logits, dim=-1)

        # 获取标签词的概率
        label_scores = {}
        for label in LABELS:
            token_ids = self.tokenizer.encode(label, add_special_tokens=False)
            token_probs = probs[token_ids]  # 获取所有 token 的概率
            label_scores[label] = token_probs.mean().item()  # 取平均
        return max(label_scores, key=lambda k: label_scores[k])

    def lm_likelihood(self, premise, hypothesis):
        label_scores = {}
        for label in LABELS:
            # 构造完整Prompt（包含标签）
            inputs = self.prepare_input(premise, hypothesis, label=label)
            with torch.no_grad():
                outputs = self.model(**inputs, labels=inputs["input_ids"])
            log_likelihood = -outputs.loss * inputs["input_ids"].shape[1]
            label_scores[label] = log_likelihood.item()
        return max(label_scores, key=lambda k: label_scores[k])

    def prepare_input(self, premise, hypothesis, max_premise_length=512, max_length=1024, label=None):

        premise = self.tokenizer.decode(
            self.tokenizer.encode(premise, truncation=True, max_length=max_premise_length),
            skip_special_tokens=True
        )
        prompt = self.build_prompt(premise, hypothesis)
        if label:
            prompt += " " + label
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=max_length, add_special_tokens=True
        ).to(self.device)
        return inputs
