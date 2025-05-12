import re

def mapping(text):
    # 使用更精确的正则匹配
    match = re.search(r"\b(entailment|neutral|contradiction)\b", text.lower())
    if match:
        return match.group(1)

    # 保留原有逻辑作为fallback
    text = text.strip().lower()
    if "entail" in text:
        return "entailment"
    elif "neutral" in text:
        return "neutral"
    elif "contradict" in text:
        return "contradiction"
    return "neutral"
