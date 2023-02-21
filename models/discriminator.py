import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
import yaml


class Discriminator(nn.Module):
    """Class for handling the Discriminator module of the GAN"""

    def __init__(self, config_file):
        super(Discriminator, self).__init__()
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)['discriminator']
        self.tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
        self.bert = AutoModel.from_pretrained(config['model_name'])
        self.dropout = nn.Dropout(config['dropout_prob'])
        self.fc = nn.Linear(config['hidden_size'], 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.fc(pooled_output)
        probs = torch.sigmoid(logits)
        return probs.squeeze()
