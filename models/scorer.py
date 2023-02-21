import json
from typing import List

from sklearn.metrics import f1_score


class SquadScorer:
    """Class for handling the Scorer module which will act as the Discriminator but
    instead of classifying wether the generated text is generated (fake) or not, this
    module will evaluate how good the response is, based on the two main metrics used
    in Question-Answering applications:

        - Exact Match (EM)
        - F1 Score
    """

    def __init__(self, squad_path: str):
        with open(squad_path) as f:
            self.squad_data = json.load(f)["data"]

    def _get_answers(self, squad_qa):
        answers = []
        for qa in squad_qa:
            for answer in qa["answers"]:
                answers.append(answer["text"])
        return answers

    def _get_ground_truth(self, qid):
        for data in self.squad_data:
            for para in data["paragraphs"]:
                for qa in para["qas"]:
                    if qa["id"] == qid:
                        return self._get_answers(qa["answers"])

    def get_scores(self, predictions: List[str], qids: List[str]):
        """Method for returning the scores of predictions"""
        
        em_scores = []
        f1_scores = []
        for pred, qid in zip(predictions, qids):
            gt = self._get_ground_truth(qid)
            if not gt:
                continue
            em = int(pred in gt)
            em_scores.append(em)
            f1_scores.append(f1_score([gt], [pred]))
        em_score = sum(em_scores) / len(em_scores)
        f1_score = sum(f1_scores) / len(f1_scores)
        return em_score, f1_score
