import json
from typing import List

from sklearn import metrics


class Scorer:
    """Class for handling the Scorer module which will act as the Discriminator but
    instead of classifying wether the generated text is generated (fake) or not, this
    module will evaluate how good the response is, based on the two main metrics used
    in Question-Answering applications:

        - Exact Match (EM)
        - F1 Score
    """

    def __init__(self):
        pass

    def get_scores(self, predictions: List[str], batch_answers: List[str]):
        """Method for returning the scores of predictions"""

        em_scores = []
        f1_scores = []
        for pred, answer in zip(predictions, batch_answers):
            em_scores.append(int(pred in answer))
            f1_scores.append(metrics.f1_score([answer], [pred]))
        em_score = sum(em_scores) / len(em_scores)
        f1_score = sum(f1_scores) / len(f1_scores)
        return em_score, f1_score
