import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

from data_loading import TextDataset
from generator import Generator
from discriminator import Discriminator

class GANTrainer:
    """Class for handling the training process of the whole GAN"""

    def __init__(self, config):
        self.config = config
        self.generator = Generator(config)
        self.discriminator = Discriminator(config)
        self.train_dataset = TextDataset(config)
        self.train_loader = None
        
    def load_data(self):
        self.train_loader = data.DataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=self.train_dataset.collate_fn
        )
        
    def train(self):
        criterion = nn.BCELoss()
        gen_optimizer = optim.Adam(self.generator.parameters(), lr=self.config.gen_learning_rate)
        dis_optimizer = optim.Adam(self.discriminator.parameters(), lr=self.config.dis_learning_rate)
        for epoch in range(self.config.num_epochs):
            for batch_idx, batch_data in enumerate(self.train_loader):
                # Train the discriminator
                self.discriminator.train()
                self.generator.eval()
                dis_optimizer.zero_grad()
                real_inputs, real_labels = batch_data
                real_labels = torch.ones(real_labels.size(0))
                fake_inputs = self.generator.sample(real_inputs.size(0))
                fake_labels = torch.zeros(fake_inputs.size(0))
                dis_inputs = torch.cat((real_inputs, fake_inputs))
                dis_labels = torch.cat((real_labels, fake_labels))
                dis_outputs = self.discriminator(dis_inputs)
                dis_loss = criterion(dis_outputs.squeeze(), dis_labels)
                dis_loss.backward()
                dis_optimizer.step()

                # Train the generator
                self.generator.train()
                self.discriminator.eval()
                gen_optimizer.zero_grad()
                fake_inputs = self.generator.sample(real_inputs.size(0))
                gen_outputs = self.discriminator(fake_inputs)
                gen_labels = torch.ones(gen_outputs.size(0))
                gen_loss = criterion(gen_outputs.squeeze(), gen_labels)
                gen_loss.backward()
                gen_optimizer.step()
                
                # Print loss and save models
                if (batch_idx+1) % self.config.print_every == 0:
                    print(f"Epoch [{epoch+1}/{self.config.num_epochs}], Batch [{batch_idx+1}/{len(self.train_loader)}], Generator Loss: {gen_loss.item():.4f}, Discriminator Loss: {dis_loss.item():.4f}")
                if (batch_idx+1) % self.config.save_every == 0:
                    self.generator.save_model(f"{self.config.model_dir}/generator_{epoch+1}_{batch_idx+1}.pt")
                    self.discriminator.save_model(f"{self.config.model_dir}/discriminator_{epoch+1}_{batch_idx+1}.pt")
