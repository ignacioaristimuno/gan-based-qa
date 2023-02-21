from torch.utils.data import Dataset
from .preprocessing import TextPreprocessor


class TextDataset(Dataset):
    """Custom Dataset created for handling the needed preprocessing steps."""
    
    def __init__(self, data, tokenizer, max_seq_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.preprocessor = TextPreprocessor()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        document = sample["document"]
        question = sample["question"]
        answer = sample["answer"]
        label = sample["label"]
        
        # Preprocess the text using the TextPreprocessor class
        document = self.preprocessor.process_text(document)
        question = self.preprocessor.process_text(question)
        
        # Concatenate document and question with special tokens
        inputs = self.tokenizer.encode_plus(
            document,
            question,
            add_special_tokens=True,
            max_length=self.max_seq_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=True,
            return_tensors="pt",
        )

        return inputs, label
