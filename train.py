import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from yaml import safe_load

from models.generator import Generator
from models.discriminator import Discriminator
from models.scorer import Scorer
from data_loading import TextDataset


class GANTrainer:
    """Class for handling the training process of the whole GAN"""

    def __init__(self, config_file: str = "config.yaml"):
        with open(config_file, "r") as file:
            config = safe_load(file)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = config["Training"]["batch_size"]
        self.max_epochs = config["Training"]["max_epochs"]
        self.learning_rate = config["Training"]["learning_rate"]

        # Modules
        self.generator = Generator(config).to(self.device)
        self.discriminator = Discriminator(config).to(self.device)
        self.scorer = Scorer()

        # Optimizers
        self.generator_optimizer = optim.Adam(
            self.generator.parameters(), lr=self.learning_rate
        )
        self.discriminator_optimizer = optim.Adam(
            self.discriminator.parameters(), lr=self.learning_rate
        )

        # Loss function
        self.criterion = nn.BCELoss()

    def train(self):
        """Method for training the GAN model"""

        # Load data
        train_dataset = TextDataset(
            file_path="data/squad_v2/train-v2.0.json",
            tokenizer=self.generator.tokenizer,
            max_seq_length=512,
        )
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4
        )

        # Train loop
        for epoch in range(self.max_epochs):
            self.generator.train()
            self.discriminator.train()

            for i, batch in enumerate(train_loader):

                # Set up input data
                print(f"BORRAR: Batch element -> {batch[0].keys()}")
                input_ids = batch[0]["input_ids"].to(self.device)
                attention_mask = batch[0]["attention_mask"].to(self.device)
                # start_positions = batch["start_positions"].to(self.device)
                # end_positions = batch["end_positions"].to(self.device)
                answers = batch[1]
                print(f"BORRAR: Batch answer -> {answers}")

                # Train generator
                generated_ids, _ = self.generator(input_ids, attention_mask)
                scores = self.scorer(generated_ids, attention_mask)
                generator_loss = -torch.mean(scores)
                self.generator_optimizer.zero_grad()
                generator_loss.backward(retain_graph=True)
                self.generator_optimizer.step()

                # Train discriminator
                tokenized_answers = self.discriminator.tokenizer.encode_plus(
                    answers,
                    add_special_tokens=True,
                    max_length=self.max_seq_length,
                    padding="max_length",
                    truncation=True,
                    return_attention_mask=True,
                    return_token_type_ids=True,
                    return_tensors="pt",
                )
                real_scores = self.discriminator(
                    tokenized_answers["input_ids", tokenized_answers["attention_mask"]]
                )
                fake_scores = self.discriminator(generated_ids, attention_mask)
                real_labels = torch.ones_like(real_scores)
                fake_labels = torch.zeros_like(fake_scores)
                discriminator_loss = self.criterion(
                    real_scores, real_labels
                ) + self.criterion(fake_scores, fake_labels)
                self.discriminator_optimizer.zero_grad()
                discriminator_loss.backward(retain_graph=True)
                self.discriminator_optimizer.step()

                # Get Scorer's scores
                with torch.no_grad():
                    decoded_texts = [
                        self.generator.tokenizer.decode(ids, skip_special_tokens=True)
                        for ids in generated_ids
                    ]
                scores = self.scorer.get_scores(
                    decoded_texts,
                )

                scores = self.scorer(generated_ids, attention_mask)
                scorer_loss = self.criterion(scores, real_labels)
                self.scorer_optimizer.zero_grad()
                scorer_loss.backward()
                self.scorer_optimizer.step()

                # Print losses
                if i % 100 == 0:
                    print(
                        f"Epoch {epoch+1}/{self.max_epochs} Batch {i+1}/{len(train_loader)}:"
                    )
                    print(f"Generator loss: {generator_loss:.4f}")
                    print(f"Discriminator loss: {discriminator_loss:.4f}")
                    print(f"Scorer loss: {scorer_loss:.4f}")

            # Save model after every epoch
            self.save_model(f"models/generator_{epoch+1}.pt", self.generator)
            self.save


trainer = GANTrainer()
trainer.train()

train_dataset = TextDataset(
    file_path="data/squad_v2/train-v2.0.json",
    tokenizer=trainer.generator.tokenizer,
    max_seq_length=512,
)
train_loader = DataLoader(
    train_dataset, batch_size=trainer.batch_size, shuffle=True, num_workers=4
)
sam = next(iter(train_loader))
