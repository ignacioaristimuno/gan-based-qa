import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from generator import Generator
from discriminator import Discriminator
from scorer import Scorer
from dataset import SquadDataset


class GANTrainer:
    """Class for handling the training process of the whole GAN"""

    def __init__(self, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = config["training"]["batch_size"]
        self.max_epochs = config["training"]["max_epochs"]
        self.learning_rate = config["training"]["learning_rate"]

        self.generator = Generator(config).to(self.device)
        self.discriminator = Discriminator(config).to(self.device)
        self.scorer = Scorer(config).to(self.device)

        self.generator_optimizer = optim.Adam(
            self.generator.parameters(), lr=self.learning_rate
        )
        self.discriminator_optimizer = optim.Adam(
            self.discriminator.parameters(), lr=self.learning_rate
        )
        self.scorer_optimizer = optim.Adam(self.scorer.parameters(), lr=self.learning_rate)

        self.criterion = nn.BCELoss()

    def train(self):
        # Load data
        train_dataset = SquadDataset(
            file_path="data/train-v2.0.json",
            tokenizer=self.generator.tokenizer,
            max_length=self.generator.max_length,
        )
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4
        )

        # Train loop
        for epoch in range(self.max_epochs):
            self.generator.train()
            self.discriminator.train()
            self.scorer.train()

            for i, batch in enumerate(train_loader):
                # Set up input data
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                start_positions = batch["start_positions"].to(self.device)
                end_positions = batch["end_positions"].to(self.device)

                # Train generator
                generated_ids = self.generator(input_ids, attention_mask)
                scores = self.scorer(generated_ids, attention_mask)
                generator_loss = -torch.mean(scores)
                self.generator_optimizer.zero_grad()
                generator_loss.backward(retain_graph=True)
                self.generator_optimizer.step()

                # Train discriminator
                real_scores = self.discriminator(start_positions, end_positions)
                fake_scores = self.discriminator(generated_ids, attention_mask)
                real_labels = torch.ones_like(real_scores)
                fake_labels = torch.zeros_like(fake_scores)
                discriminator_loss = (
                    self.criterion(real_scores, real_labels)
                    + self.criterion(fake_scores, fake_labels)
                )
                self.discriminator_optimizer.zero_grad()
                discriminator_loss.backward(retain_graph=True)
                self.discriminator_optimizer.step()

                # Train scorer
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