import os
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from agent.causal_model import CausalModel
from agent.seq2seq_model import SeqModel
from data_loader import load_data_2
from model_pool import ModelPool

def chunk_dataset(dataset, chunk_size=50):
    """将数据集拆分为固定大小的块"""
    return [
        [{k: dataset[k][j] for k in dataset.column_names}
         for j in range(i, min(i+chunk_size, len(dataset)))]
        for i in range(0, len(dataset), chunk_size)
    ]

def evaluate_chunk(chunk, model_func, model_name, method_name, chunk_idx):
    """评估单个数据块"""
    middle_dir = "./result/nli_halluc_middel"
    os.makedirs(middle_dir, exist_ok=True)

    file_name = f"{model_name}_{method_name}_chunk{chunk_idx}.jsonl"
    file_path = os.path.join(middle_dir, file_name)

    print(f"[{threading.current_thread().name}] {file_name} begin to be evaluated, loading...")

    if os.path.exists(file_path):
        print(f"[{threading.current_thread().name}] {file_name} has been evaluated, loaded success")
        with open(file_path, "r") as f:
            return [json.loads(line) for line in f]

    results = []
    for example in chunk:
        try:
            pred = model_func(example["premise"], example["hypothesis"])
        except RuntimeError as e:
            print(f"[{threading.current_thread().name}] {str(e)} error at sample premise is {example['premise']}, hypothesis is {example['hypothesis']}")
            raise
        # 将预测结果转换为0/1标签
        pred_label = 0 if pred == "entailment" else 1
        results.append({
            "premise": example["premise"],
            "hypothesis": example["hypothesis"],
            "gold_label": example["gold_label"],
            "predicted_label": pred_label,
            "model": model_name,
            "method": method_name
        })

    with open(file_path, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

    return results

def merge_results(final_results):
    """合并所有中间结果"""
    middle_dir = "./result/nli_halluc_middel"

    processed = set([(r["model"], r["method"]) for r in final_results])
    for filename in os.listdir(middle_dir):
        if filename.endswith(".jsonl"):
            parts = filename.split("_")
            model_name = parts[0]
            method_name = parts[1]
            if (model_name, method_name) in processed:
                continue

            with open(os.path.join(middle_dir, filename), "r") as f:
                final_results.extend([json.loads(line) for line in f])

    return final_results

def save_results(results, path):
    """保存结果到文件"""
    with open(path, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

def evaluate_concurrently():
    # 初始化模型（保持与eval一致）
    model_pools = {
        "CausalModel": ModelPool(CausalModel, 'openai-community/gpt2', pool_size=2),
        "SeqModel": ModelPool(SeqModel, 'google/flan-t5-small', pool_size=2)
    }

    methods = {
        "CausalModel": {
            "Likelihood": lambda p, h: model_pools["CausalModel"].run(p, h, 'lm_likelihood'),
            "Sampled": lambda p, h: model_pools["CausalModel"].run(p, h, 'lm_sampled_completion'),
            "ProbDist": lambda p, h: model_pools["CausalModel"].run(p, h, 'lm_prob_distribution')
        },
        "SeqModel": {
            "Sampled": lambda p, h: model_pools["SeqModel"].run(p, h, 'lm_sampled_completion'),
            "ProbDist": lambda p, h: model_pools["SeqModel"].run(p, h, 'lm_prob_distribution')
        }
    }

    # 加载数据并分块
    dataset = load_data_2()
    chunks = chunk_dataset(dataset, chunk_size=50)

    final_dir = "./result/nli_halluc_final"
    os.makedirs(final_dir, exist_ok=True)

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for model_name, model_methods in methods.items():
            for method_name, method_func in model_methods.items():
                for chunk_idx, chunk in enumerate(chunks):
                    futures.append(
                        executor.submit(
                            evaluate_chunk,
                            chunk,
                            method_func,
                            model_name,
                            method_name,
                            chunk_idx
                        )
                    )

        results = []
        for future in tqdm(as_completed(futures), total=len(futures)):
            try:
                chunk_results = future.result()
                results.extend(chunk_results)
            except Exception as e:
                print(f"Error in task: {str(e)}")

    # 合并最终结果
    final_results = merge_results(results)

    # 计算评估指标
    metric_stats = {}
    for result in final_results:
        key = (result["model"], result["method"])
        if key not in metric_stats:
            metric_stats[key] = {
                "true": [],
                "pred": [],
                "correct": []
            }
        metric_stats[key]["true"].append(result["gold_label"])
        metric_stats[key]["pred"].append(result["predicted_label"])
        metric_stats[key]["correct"].append(result["predicted_label"] == result["gold_label"])

    # 生成报告
    print("\nModel-Method Accuracy Breakdown:")
    final_report_path = os.path.join(final_dir, "metric_report.txt")
    with open(final_report_path, "w") as f:
        f.write("Model-Method | Accuracy | Precision | Recall | F1 Score | True Count\n")
        f.write("-------------------------------------------------------\n")

        for (model, method), stats in metric_stats.items():
            true = stats["true"]
            pred = stats["pred"]

            accuracy = accuracy_score(true, pred)
            precision = precision_score(true, pred)
            recall = recall_score(true, pred)
            f1 = f1_score(true, pred)

            count_true = stats["correct"].count(True)

            line = (f"{model} ({method}): "
                    f"Acc: {accuracy:.4f} | "
                    f"Precision: {precision:.4f} | "
                    f"Recall: {recall:.4f} | "
                    f"F1: {f1:.4f} | "
                    f"True Count: {count_true}")

            print(line)
            f.write(f"{model} | {method} | {accuracy:.4f} | {precision:.4f} | {recall:.4f} | {f1:.4f} | {count_true}\n")

    print(f"\nThe Evaluation result save to: {final_report_path}")

if __name__ == "__main__":
    evaluate_concurrently()
