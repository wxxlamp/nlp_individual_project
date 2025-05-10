import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from agent import verbalizer
from agent.base_model import BaseModel, LABELS


class SeqModel(BaseModel):
    def __init__(self, model_name="google-t5/t5-base"):
        super().__init__()
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def build_prompt(self, premise, hypothesis):
        return f"""nli task: premise: {premise} hypothesis: {hypothesis}"""

    def lm_sampled_completion(self, premise, hypothesis):
        prompt = self.build_prompt(premise, hypothesis)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(
            **inputs,
            max_length=50,
            temperature=0.7,
            top_k=50,
            do_sample=True
        )
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return verbalizer.mapping(generated)

    def lm_prob_distribution(self, premise, hypothesis):
        prompt = self.build_prompt(premise, hypothesis)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                output_scores=True,
                return_dict_in_generate=True,
                max_length=5  # 限制生成长度以提高效率
            )
        # 获取第一个生成词的概率分布
        probs = torch.softmax(outputs.scores[0][0], dim=-1)
        label_scores = {}
        for label in LABELS:
            token_id = self.tokenizer.encode(label, add_special_tokens=False)[0]
            label_scores[label] = probs[token_id].item()
        return max(label_scores, key=lambda k: label_scores[k])

    def lm_likelihood(self, premise, hypothesis):
        return ''
