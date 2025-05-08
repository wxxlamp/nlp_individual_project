import json
import os
from datasets import Dataset

def load_jsonl(file_path):
    data = []
    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return Dataset.from_list(data)


def load_data():
    target_file_path = "./data/target_data.jsonl"

    if os.path.exists(target_file_path):
        return load_jsonl(target_file_path)
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
