import os
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from agent.causal_model import CausalModel
from agent.seq2seq_model import SeqModel
from data_loader import load_data_1
from model_pool import ModelPool


def chunk_dataset(dataset, chunk_size=50):
    """将数据集拆分为固定大小的块"""
    return [
        [{k: dataset[k][j] for k in dataset.column_names}  # 将每个样本转换为字典格式
         for j in range(i, min(i+chunk_size, len(dataset)))]
        for i in range(0, len(dataset), chunk_size)
    ]

def evaluate_chunk(chunk, model_func, model_name, method_name, chunk_idx):
    """评估单个数据块"""
    middle_dir = "./result/nli_base_middle"
    os.makedirs(middle_dir, exist_ok=True)

    # 生成唯一文件名
    file_name = f"{model_name}_{method_name}_chunk{chunk_idx}.jsonl"
    file_path = os.path.join(middle_dir, file_name)

    print(f"[{threading.current_thread().name}] {file_name} begin to be evaluated, loading...")

    # 如果中间文件已存在，直接加载
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
        results.append({
            "premise": example["premise"],
            "hypothesis": example["hypothesis"],
            "gold_label": example["gold_label"],
            "source": example["source"],
            "predicted_label": pred,
            "model": model_name,
            "method": method_name
        })

    # 保存中间结果
    with open(file_path, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

    return results

def merge_results(final_results):
    """合并所有中间结果到最终目录"""
    middle_dir = "./result/nli_base_middle"

    # 合并新结果
    processed = set([(r["model"], r["method"], r["source"]) for r in final_results])
    for filename in os.listdir(middle_dir):
        if filename.endswith(".jsonl"):
            model_name, method_name = filename.split("_")[:2]
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

    # 初始化模型
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
    dataset = load_data_1()
    chunks = chunk_dataset(dataset, chunk_size=50)

    final_dir = "./result/nli_base_final"
    os.makedirs(final_dir, exist_ok=True)

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []

        # 为每个模型/方法/数据块创建任务
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

        # 使用tqdm显示进度
        results = []
        for future in tqdm(as_completed(futures), total=len(futures)):
            try:
                chunk_results = future.result()
                results.extend(chunk_results)
            except Exception as e:
                print(f"Error in task: {str(e)}")

    # 合并最终结果
    final_results = merge_results(results)
    print(f"Total evaluated samples: {len(final_results)}")

    # 计算准确率
    accuracy_stats = {}
    for result in final_results:
        # 统计每个source的准确率
        source_key = (result["model"], result["method"], result["source"])
        total_key = (result["model"], result["method"], "total")
        correct = result["predicted_label"] == result["gold_label"]

        # 更新source统计
        if source_key not in accuracy_stats:
            accuracy_stats[source_key] = {"correct": 0, "total": 0}
        accuracy_stats[source_key]["correct"] += int(correct)
        accuracy_stats[source_key]["total"] += 1

        # 更新total统计
        if total_key not in accuracy_stats:
            accuracy_stats[total_key] = {"correct": 0, "total": 0}
        accuracy_stats[total_key]["correct"] += int(correct)
        accuracy_stats[total_key]["total"] += 1


    print("\nModel-Method Accuracy Breakdown:")
    final_report_path = os.path.join(final_dir, "accuracy_report.txt")
    with open(final_report_path, "w") as f:
        f.write("Model-Method Accuracy Breakdown:\n")

        # 获取所有唯一的model-method组合
        model_methods = {(model, method) for (model, method, _) in accuracy_stats.keys()}

        for model, method in sorted(model_methods):
            # 获取三种统计维度
            total_stats = accuracy_stats.get((model, method, "total"), {"correct":0, "total":0})
            mismatched_stats = accuracy_stats.get((model, method, "mismatched"), {"correct":0, "total":0})
            matched_stats = accuracy_stats.get((model, method, "matched"), {"correct":0, "total":0})

            # 生成报告行
            lines = [
                f"{model} ({method}) - Total: {total_stats['correct']/total_stats['total']:.4f} ({total_stats['correct']}/{total_stats['total']})",
                f"{model} ({method}) - Mismatched: {mismatched_stats['correct']/mismatched_stats['total']:.4f} ({mismatched_stats['correct']}/{mismatched_stats['total']})",
                f"{model} ({method}) - Matched: {matched_stats['correct']/matched_stats['total']:.4f} ({matched_stats['correct']}/{matched_stats['total']})"
            ]

            # 打印和写入文件
            for line in lines:
                print(line)
                f.write(line + "\n")
            print()
            f.write("\n")

    print(f"\nThe Evaluation result save to: {final_report_path}")

if __name__ == "__main__":
    evaluate_concurrently()
