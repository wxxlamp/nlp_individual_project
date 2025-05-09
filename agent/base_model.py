from abc import abstractmethod, ABC

LABELS = ["entailment", "neutral", "contradiction"]


class BaseModel(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def build_prompt(self, premise, hypothesis):
        pass

    @abstractmethod
    def lm_sampled_completion(self, premise, hypothesis):
        pass

    @abstractmethod
    def lm_prob_distribution(self, premise, hypothesis):
        pass

    @abstractmethod
    def lm_likelihood(self, premise, hypothesis):
        pass
