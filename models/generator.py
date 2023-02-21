import yaml
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import load_dataset


class Generator:
    """Generator class of the GAN for generating answers to the questions given as input
    along with the context (document)."""

    def __init__(self, config_path: str):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.model_name = self.config['model_name']
        self.tokenizer_name = self.config['tokenizer_name']
        self.max_length = self.config['max_length']
        self.batch_size = self.config['batch_size']
        self.learning_rate = self.config['learning_rate']
        self.warmup_steps = self.config['warmup_steps']
        self.epochs = self.config['epochs']
        self.train_data_path = self.config['train_data_path']
        self.model_output_path = self.config['model_output_path']

    def load_data(self) -> dict:
        dataset = load_dataset('json', data_files=self.train_data_path)['train']
        tokenized_dataset = dataset.map(lambda x: self.tokenize_data(x), batched=True)
        return tokenized_dataset

    def tokenize_data(self, example):
        encoded = self.tokenizer.encode_plus(
            example['context'],
            example['question'],
            max_length=self.max_length,
            truncation='only_second',
            padding='max_length',
            return_tensors='pt'
        )
        encoded['labels'] = self.tokenizer.encode(
            example['answer_text'],
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length
        )
        return encoded

    def train(self) -> None:
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)

        train_dataset = self.load_data()

        training_args = Seq2SeqTrainingArguments(
            output_dir=self.model_output_path,
            num_train_epochs=self.epochs,
            per_device_train_batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            warmup_steps=self.warmup_steps,
            prediction_loss_only=True,
            logging_steps=500,
            save_steps=1000,
            overwrite_output_dir=True,
        )

        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=lambda data: {'input_ids': torch.stack([f['input_ids'] for f in data]),
                                        'attention_mask': torch.stack([f['attention_mask'] for f in data]),
                                        'decoder_input_ids': torch.stack([f['labels'][:-1] for f in data]),
                                        'decoder_attention_mask': torch.stack([torch.ones_like(f['labels'][:-1]) for f in data]),
                                        'labels': torch.stack([f['labels'][1:] for f in data])
                                        },
        )

        trainer.train()

        model.save_pretrained(self.model_output_path)
        tokenizer.save_pretrained(self.model_output_path)


# Example code
# generator = Generator(config_path='configs.yaml')
