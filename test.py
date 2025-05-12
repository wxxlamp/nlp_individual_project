from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")

# 1. 获取概率分布
inputs = tokenizer("Hello, how are you?", return_tensors="pt")
outputs = model(**inputs, output_hidden_states=True)
logits = outputs.logits  # 形状：(batch_size, seq_len, vocab_size)
prob_dist = torch.softmax(logits[:, -1, :], dim=-1)  # 最后一个位置的词汇概率分布

# 2. 计算似然值
input_ids = tokenizer.encode("Hello, how are you?", return_tensors="pt")
with torch.no_grad():
    outputs = model(input_ids, labels=input_ids)
log_likelihood = -outputs.loss * input_ids.size(1)  # 对数似然近似值

# 3. 生成采样文本
generated = model.generate(
    input_ids,
    do_sample=True,
    temperature=0.7,
    top_k=50,
    max_length=50
)
sampled_text = tokenizer.decode(generated[0], skip_special_tokens=True)
