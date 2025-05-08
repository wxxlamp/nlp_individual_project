from agent.causal_model import CausalModel
from agent.seq2seq_model import SeqModel
from data_loader import load_data

def evaluate_model(dataset, model_func, model_name, method_name):
    correct = 0
    total = len(dataset)
    for example in dataset:
        pred = model_func(example["premise"], example["hypothesis"])
        if pred == example["gold_label"]:
            correct += 1
    accuracy = correct / total
    print(f"{model_name} ({method_name}) Accuracy: {accuracy:.4f}")
    return accuracy

causal_model = CausalModel()
seq2seq_model = SeqModel()
# 评估所有方法
methods = {
    "Causal LM (Likelihood)": causal_model.lm_likelihood,
    "Causal LM (Sampled)": causal_model.lm_sampled_completion,
    "Causal LM (Prob Dist)": causal_model.lm_prob_distribution,
    "Seq2Seq LM (Sampled)": seq2seq_model.lm_sampled_completion,
    "Seq2Seq LM (Prob Dist)": seq2seq_model.lm_prob_distribution
}

for name, (func) in methods.items():
    wrapped_func = lambda p, h: func(p, h)
    data = load_data()
    evaluate_model(data, wrapped_func, name.split()[0], name.split()[-1])
