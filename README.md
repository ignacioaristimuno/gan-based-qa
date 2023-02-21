# GAN-based Question Answering System

This repository contains a GAN-based approach for Question Answering (QA) systems. The model is trained on the **SQuADv2** dataset using Generative Adversarial Networks (GAN) training.


## Relevant Files

The following files are included in this project:

- `config.yaml`: contains the configurations for the model, including hyperparameters and file paths.
- `etl.py`: contains the ETL pipeline for downloading and preprocessing the SQuAD dataset.
- `generator.py`: contains the Generator network for generating answers to given questions.
- `discriminator.py`: contains the Discriminator network for distinguishing between real and generated answers.
- `scorer.py`: contains the Scorer network for evaluating the quality of generated answers.
- `preprocessing.py`: contains utility functions for preprocessing text data.
- `data_loader.py: contains a custom data loader for loading preprocessed SQuAD dataset.
- `train.py`: contains the GANTrainer class for training the GAN model on the SQuAD dataset.
- `inference.py`: contains a script for generating answers to given questions using the trained GAN model.
- `evaluation.py`: contains utility functions for evaluating the performance of the trained model.


## Components

This project is composed of three main components: the Generator, the Discriminator, and the Scorer.

The `Generator` network takes a document and a question as input and generates a new question as output. It uses a pre-trained language model and a custom scoring function to generate questions that are both grammatically correct and relevant to the input document.

The `Discriminator` network takes a question as input and outputs a score that indicates how well it matches the input document. This score is used by the Generator network to learn how to generate better questions.

The `Scorer` is used to evaluate the quality of the generated questions. It computes the F1 score and the Exact Match between the generated questions and the ground truth questions from the Squadv2 dataset.


## Training Flow

During training, the Generator network generates a question given a document as input. The Discriminator network then scores the generated question based on how well it matches the input document. The score is passed to the Generator network, which uses it to update its parameters and generate better questions. This process is repeated until the Generator network can generate questions that are both grammatically correct and relevant to the input document.


## References

- Goodfellow, Ian, et al. "Generative adversarial nets." Advances in neural information processing systems. 2014.