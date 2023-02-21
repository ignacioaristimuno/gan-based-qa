import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class Generator(torch.nn.Module):
    """Generator class of the GAN for generating answers to the questions given as input
    along with the context (document).
    
    Example input: [CLS] What is the capital of France? [SEP] Paris is the capital of France. [SEP]
    """

    def __init__(self, model_name, tokenizer_name):
        super(Generator, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask):
        output = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=32,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            num_return_sequences=1,
            decoder_start_token_id=self.tokenizer.cls_token_id,
            eos_token_id=self.tokenizer.sep_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            bos_token_id=self.tokenizer.cls_token_id
        )
        return output


# Example code
# generator = Generator(config_path='configs.yaml')
