import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from agent import verbalizer
from agent.base_model import BaseModel, LABELS


class SeqModel(BaseModel):
    def __init__(self, model_name="google-t5/t5-base"):
        super().__init__()
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def build_prompt(self, premise, hypothesis):
        return f"""Given a premise and a hypothesis, determine if their relationship is entailment, neutral, or contradiction. Replace [LABEL] with your answer.
            Premise: {premise} 
            Hypothesis: {hypothesis}
            Answer: [LABEL]"""

    def lm_sampled_completion(self, premise, hypothesis):
        inputs = self.prepare_input(premise, hypothesis)

        outputs = self.model.generate(
            **inputs,
            max_length=512,
            temperature=0.7,
            top_k=50,
            do_sample=True
        )
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return verbalizer.mapping(generated)

    def lm_prob_distribution(self, premise, hypothesis):
        # 获取编码后的输入（仅 Encoder 部分）
        encoder_inputs = self.prepare_input(premise, hypothesis)
        label_scores = {}

        for label in LABELS:
            clean_label = label.lower().strip()
            # 编码标签并检查有效性
            encoded_label = self.tokenizer(
                clean_label, add_special_tokens=False, return_tensors="pt"
            ).to(self.device)
            label_ids = encoded_label.input_ids[0]

            # 检查无效 token 或空标签
            if len(label_ids) == 0 or any(tid >= self.tokenizer.vocab_size for tid in label_ids):
                label_scores[label] = -float("inf")
                continue

            # 构造 Decoder 输入：decoder_start_token + label[:-1]
            decoder_start_token_id = self.model.config.decoder_start_token_id
            decoder_input_ids = torch.cat([
                torch.tensor([[decoder_start_token_id]], device=self.device),
                label_ids[:-1].unsqueeze(0)
            ], dim=1)

            # 模型前向计算
            with torch.no_grad():
                outputs = self.model(
                    input_ids=encoder_inputs.input_ids,
                    attention_mask=encoder_inputs.attention_mask,
                    decoder_input_ids=decoder_input_ids
                )
            logits = outputs.logits  # (1, seq_len, vocab_size)

            # 计算对数概率并求和（数值稳定）
            log_probs = torch.log_softmax(logits, dim=-1)
            try:
                selected_log_probs = log_probs[0, torch.arange(len(label_ids)), label_ids]
            except IndexError:
                # 处理长度不匹配（如标签被截断）
                label_scores[label] = -float("inf")
                continue

            total_log_prob = selected_log_probs.sum().item()
            label_scores[label] = total_log_prob

        # 返回概率最高的标签
        return max(label_scores, key=lambda k: label_scores[k])

    def lm_likelihood(self, premise, hypothesis):
        return ''

    def prepare_input(self, premise, hypothesis, max_premise_length=256, max_length=512):
        # 清理 Premise 并构建任务提示
        premise = self.tokenizer.decode(
            self.tokenizer.encode(premise, truncation=True, max_length=max_premise_length),
            skip_special_tokens=True
        )
        prompt = self.build_prompt(premise, hypothesis)  # 确保包含任务前缀（如 "mnli:"）
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=max_length, add_special_tokens=True
        ).to(self.device)
        return inputs
