def mapping(text):
    text = text.strip().lower()
    if "entail" in text:
        return "entailment"
    elif "neutral" in text:
        return "neutral"
    elif "contradict" in text:
        return "contradiction"
    else:
        return "neutral"
