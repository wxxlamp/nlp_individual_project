import json
import os
from datasets import Dataset, load_dataset


def load_jsonl(file_path, test=False):
    data = []
    count = 0

    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line))
            count += 1

            if test and count >= 10:  # 如果是测试模式且已读取了10条记录，则退出循环
                break

    return Dataset.from_list(data)

def load_data_1(test=False):
    target_file_path = "./data/target_data.jsonl"

    if os.path.exists(target_file_path):
        return load_jsonl(target_file_path, test)
    else:
        mismatched_data = load_jsonl("data/dev_mismatched_sampled-1.jsonl")
        matched_data = load_jsonl("data/dev_matched_sampled-1.jsonl")

        filtered_mismatched_data = [
            {"premise": item["sentence1"], "hypothesis": item["sentence2"], "gold_label": item["gold_label"],
             'genre': item["genre"], "source": "mismatched"}
            for item in mismatched_data
        ]

        filtered_matched_data = [
            {"premise": item["sentence1"], "hypothesis": item["sentence2"], "gold_label": item["gold_label"],
             'genre': item["genre"], "source": "matched"}
            for item in matched_data
        ]

        combined_filtered_data = filtered_mismatched_data + filtered_matched_data

        with open(target_file_path, "w") as f:
            for item in combined_filtered_data:
                f.write(json.dumps(item) + "\n")

        return Dataset.from_list(combined_filtered_data)

def load_data_2(test=False):
    target_path = "./data/wikibio_hallucination_nli_format.jsonl"
    if os.path.exists(target_path):
        return load_jsonl(target_path, test)

    # 加载原始数据集
    dataset = load_dataset("potsawee/wiki_bio_gpt3_hallucination", split="evaluation")
    # 展开嵌套结构
    new_data = []
    for example in dataset:
        transformed_examples = []
        for hypo, label in zip(example["gpt3_sentences"], example["annotation"]):
            transformed = {
                "premise": example["wiki_bio_text"],
                "hypothesis": hypo,
                "gold_label": 0 if label == "accurate" else 1,
                "genre": "hallucination",
                "source": None
            }
            transformed_examples.append(transformed)
        new_data.extend(transformed_examples)

    # 存储
    revert_data = Dataset.from_list(new_data)
    revert_data.to_json(target_path, orient="records", lines=True)
    return revert_data if not test else revert_data[0]
