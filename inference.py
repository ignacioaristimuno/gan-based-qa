import json
from typing import List
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class AnswersGenerator:
    """Class for wrapping the inference on the Generator model for answering questions based
    on the context provided (document)."""

    def __init__(self, generator_path: str, tokenizer_path: str) -> None:
        self.generator = AutoModelForSeq2SeqLM.from_pretrained(generator_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    def generate_answers(self, prompts: List[str], num_answers: int) -> List[str]:
        """Main method for generating answers based on the given prompts and number of answers."""

        results = []
        for prompt in prompts:
            encoded_prompt = self.tokenizer(
                prompt, padding=True, truncation=True, return_tensors="pt"
            )
            generated_ids, attention_mask = self.generator.generate(
                input_ids=encoded_prompt["input_ids"],
                attention_mask=encoded_prompt["attention_mask"],
                num_beams=num_answers,
                max_length=64,
                no_repeat_ngram_size=2,
                early_stopping=True,
            )
            generated_answers = self.tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True
            )
            results.append(
                {
                    "answer": generated_answers,
                    "attention_mask": list(attention_mask.cpu.numpy()),
                }
            )
        return results

    def run_inference(
        self, prompts_path: str, num_answers: int, output_path: str
    ) -> None:
        """Method for running inferences on prompts stored within a JSON file."""

        with open(prompts_path, "r") as f:
            prompts = json.load(f)

        results = self.generate_answers(prompts, num_answers)
        with open(output_path, "w") as f:
            json.dump(results, f)


# Code samples
# answers_generator = AnswersGenerator(generator_path='models/generator/checkpoint.pth.tar', tokenizer_path='models/tokenizer')
# answers_generator.run_inference(prompts_path='data/prompts.json', num_answers=4, output_path='data/generated_answers.json')
