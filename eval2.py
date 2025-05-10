from agent.causal_model import CausalModel
from agent.seq2seq_model import SeqModel
from data_loader import load_data_2
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_model(dataset, model_func, model_name, method_name):
    all_preds = []
    all_true = []
    for example in dataset:
        pred = model_func(example["premise"], example["hypothesis"])
        pred = 0 if pred == "entailment" else 1
        all_preds.append(pred)
        all_true.append(example["gold_label"])
    accuracy = accuracy_score(all_true, all_preds)
    precision = precision_score(all_true, all_preds)
    recall = recall_score(all_true, all_preds)
    f1 = f1_score(all_true, all_preds)
    print(f"{model_name} ({method_name}) Accuracy: {accuracy:.4f}")
    print(f"{model_name} ({method_name}) Precision: {precision:.4f}")
    print(f"{model_name} ({method_name}) Recall: {recall:.4f}")
    print(f"{model_name} ({method_name}) F1 Score: {f1:.4f}")
    return accuracy, precision, recall, f1

causal_model = CausalModel('openai-community/gpt2')
seq2seq_model = SeqModel('google-t5/t5-base')

# 评估所有方法
methods = {
    "Causal LM (Likelihood)": causal_model.lm_likelihood,
    # "Causal LM (Sampled)": causal_model.lm_sampled_completion,
    "Causal LM (Prob Dist)": causal_model.lm_prob_distribution,
    # "Seq2Seq LM (Sampled)": seq2seq_model.lm_sampled_completion,
    "Seq2Seq LM (Prob Dist)": seq2seq_model.lm_prob_distribution
}

# 2.2
for name, (func) in methods.items():
    wrapped_func = lambda p, h: func(p, h)
    data = load_data_2(True)
    evaluate_model(data, wrapped_func, name.split()[0], name.split()[-1])
