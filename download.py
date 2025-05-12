import os

from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score


# // 设置环境变量
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# // 下载模型
# os.system('huggingface-cli download --resume-download openai-community/gpt2')
# os.system('huggingface-cli download --token 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 --resume-download FacebookAI/roberta-base')


# dataset = load_dataset("potsawee/wiki_bio_gpt3_hallucination")

import json

dataset = load_dataset("potsawee/wiki_bio_gpt3_hallucination", split="test")
for example in dataset:
    test = {
        "wiki_bio_text": example['wiki_bio_text'],
        "gpt3_text": example['gpt3_text'],
        "gpt3_sentences": example['gpt3_sentences'],
        "annotation": example['annotation']
    }
    first_row_json = json.dumps(test, ensure_ascii=False)
    print(first_row_json)
    break

