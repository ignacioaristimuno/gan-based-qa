import torch
from torch import Tensor
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import Any, Dict


class Discriminator(nn.Module):
    """Class for handling the Discriminator module of the GAN"""

    def __init__(self, config: Dict[str, Any]) -> None:
        super(Discriminator, self).__init__()
        config = config["Discriminator"]
        self.tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
        self.model = AutoModel.from_pretrained(config["model_name"])
        self.dropout = nn.Dropout(config["dropout_prob"])
        self.fc = nn.Linear(config["hidden_size"], 1)

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        outputs = self.model(input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.fc(pooled_output)
        probs = torch.sigmoid(logits)
        return probs.squeeze()
