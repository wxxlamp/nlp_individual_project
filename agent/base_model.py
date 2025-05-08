from abc import abstractmethod, ABC

import verbalizer

LABELS = ["entailment", "neutral", "contradiction"]


class BaseModel(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def build_prompt(self, premise, hypothesis):
        pass

    def lm_sampled_completion(self, premise, hypothesis):
        prompt = self.build_prompt(premise, hypothesis)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(
            **inputs,
            max_length=100,
            temperature=0.7,
            top_k=50,
            do_sample=True
        )
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return verbalizer.mapping(generated.split("Answer:")[-1])

    @abstractmethod
    def lm_prob_distribution(self, premise, hypothesis):
        pass

    @abstractmethod
    def lm_likelihood(self, premise, hypothesis):
        pass
