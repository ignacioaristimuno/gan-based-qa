import torch
from transformers import AutoTokenizer

from generator import Generator
from discriminator import Discriminator
from utils.preprocessing import preprocess_batch


class Evaluator:
    """Evaluation class for hanlding the evaluation of the Generator model"""

    def __init__(self, model_dir, generator_model_name, discriminator_model_name, device="cpu"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(generator_model_name)
        self.generator = Generator(model_dir=model_dir, model_name=generator_model_name, tokenizer_name=generator_model_name, device=device)
        self.discriminator = Discriminator(model_dir=model_dir, model_name=discriminator_model_name, tokenizer_name=generator_model_name, device=device)
    
    def evaluate(self, data_loader):
        total_loss = 0
        total_correct = 0
        total_examples = 0

        self.generator.eval()
        self.discriminator.eval()

        with torch.no_grad():
            for batch in data_loader:
                batch = preprocess_batch(batch, self.tokenizer, self.device)

                # Generate fake answers from the generator
                fake_answers = self.generator.generate(batch['input_ids'], batch['attention_mask'], batch['token_type_ids'])

                # Concatenate real and fake answers for input to the discriminator
                all_answers = torch.cat((batch['answer_input_ids'], fake_answers), dim=0)
                all_attention_masks = torch.cat((batch['answer_attention_mask'], batch['answer_attention_mask']), dim=0)
                all_token_type_ids = torch.cat((batch['answer_token_type_ids'], batch['answer_token_type_ids']), dim=0)

                # Labels for real and fake answers
                real_labels = torch.ones(batch['answer_input_ids'].shape[0], dtype=torch.long).to(self.device)
                fake_labels = torch.zeros(batch['answer_input_ids'].shape[0], dtype=torch.long).to(self.device)
                all_labels = torch.cat((real_labels, fake_labels), dim=0)

                # Evaluate discriminator on real and fake answers
                discriminator_logits = self.discriminator(all_answers, all_attention_masks, all_token_type_ids)

                # Compute discriminator loss and accuracy
                discriminator_loss = torch.nn.functional.binary_cross_entropy_with_logits(discriminator_logits, all_labels.float())
                discriminator_predictions = torch.round(torch.sigmoid(discriminator_logits))
                discriminator_correct = (discriminator_predictions == all_labels).sum().item()

                total_loss += discriminator_loss.item()
                total_correct += discriminator_correct
                total_examples += batch['answer_input_ids'].shape[0]

        mean_loss = total_loss / total_examples
        accuracy = total_correct / (total_examples * 2)

        return mean_loss, accuracy
