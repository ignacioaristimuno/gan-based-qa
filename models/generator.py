import torch
from torch import Tensor
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import Any, Dict, Tuple


class Generator(torch.nn.Module):
    """Generator class of the GAN for generating answers to the questions given as input
    along with the context (document) and returning the attention mask for the generated answer.

    Example input: [CLS] What is the capital of France? [SEP] Paris is the capital of France. [SEP]
    Example output: The capital of France is Paris.
    """

    def __init__(self, config: Dict[str, Any]):
        super(Generator, self).__init__()
        config = config["Generator"]
        self.tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_name"])
        self.model = AutoModelForSeq2SeqLM.from_pretrained(config["model_name"])
        self.max_length = config["max_length"]

    def forward(
        self, input_ids: Tensor, attention_mask: Tensor
    ) -> Tuple[Tensor, Tensor]:

        print(f"BORRAR: shape input_ids -> {input_ids.shape}")
        print(f"BORRAR: shape attention_mask -> {attention_mask.shape}")

        input_ids = input_ids.squeeze(1)
        attention_mask = attention_mask.squeeze(1)

        print(f"BORRAR: POST shape input_ids -> {input_ids.shape}")
        print(f"BORRAR: POST shape attention_mask -> {attention_mask.shape}")

        output, *_ = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_length,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            num_return_sequences=1,
            decoder_start_token_id=self.tokenizer.cls_token_id,
            eos_token_id=self.tokenizer.sep_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            bos_token_id=self.tokenizer.cls_token_id,
        )

        # Get the attention weights from the last layer of the model
        last_layer_attention = self.model.model.encoder.last_attention

        # Get the attention mask from the input_ids and expand it to match the shape of the attention weights
        expanded_mask = attention_mask.unsqueeze(-1).expand(last_layer_attention.shape)

        # Apply the attention mask to the attention weights to get the average attention per token
        averaged_attention = (last_layer_attention * expanded_mask).mean(1)

        return output, averaged_attention
