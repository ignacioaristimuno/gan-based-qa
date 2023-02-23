import json

from torch.utils.data import Dataset
from utils.preprocessing import TextPreprocessor


class TextDataset(Dataset):
    """Custom Dataset created for handling the needed preprocessing steps."""

    def __init__(self, file_path, tokenizer, max_seq_length: int = 128) -> None:
        with open(file_path, "r") as f:
            self.data = json.load(f)["data"]
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.preprocessor = TextPreprocessor()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        paragraph = sample["paragraphs"][0]
        document = paragraph["context"]

        question = paragraph["qas"][0]["question"]
        answer = paragraph["qas"][0]["answers"][0]["text"]

        # Preprocess the text using the TextPreprocessor class
        document = self.preprocessor.preprocess_text(document)
        question = self.preprocessor.preprocess_text(question)
        print(f"BORRAR: document -> {document}")
        print(f"BORRAR: question -> {question}")
        print(f"BORRAR: answer -> {answer}")

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
        # label = self.tokenizer.encode(answer, add_special_tokens=False)
        return inputs, answer
