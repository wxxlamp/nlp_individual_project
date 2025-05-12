from abc import abstractmethod, ABC

import torch

LABELS = ["entailment", "neutral", "contradiction"]


class BaseModel(ABC):
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

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
