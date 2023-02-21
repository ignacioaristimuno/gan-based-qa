import json
import urllib.request
from tqdm import tqdm


def download_squad_v2(url, path):
    """Function for retrieving the Squad-v2 dataset"""
    urllib.request.urlretrieve(url, path)


def preprocess_squad_v2(path):
    """Function for preprocessing and storing the Squad-v2 dataset for model training"""

    with open(path, 'r', encoding='utf-8') as f:
        squad_data = json.load(f)
        
    # Extract the relevant fields from the Squadv2 data
    documents = []
    questions = []
    answers = []
    for article in tqdm(squad_data['data'], desc='Extracting data'):
        for paragraph in article['paragraphs']:
            context = paragraph['context']
            for qa in paragraph['qas']:
                question = qa['question']
                answer_text = qa['answers'][0]['text']
                start_char = qa['answers'][0]['answer_start']
                end_char = start_char + len(answer_text)
                answer = {'answer_start': start_char, 'answer_end': end_char, 'text': answer_text}
                documents.append(context)
                questions.append(question)
                answers.append(answer)

    # Save the extracted data to a JSON file
    with open('data/squad_v2/processed_data.json', 'w', encoding='utf-8') as f:
        data = {'documents': documents, 'questions': questions, 'answers': answers}
        json.dump(data, f)


if __name__ == '__main__':
    squad_v2_url = 'https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json'
    squad_v2_path = 'data/squad_v2/train-v2.0.json'
    download_squad_v2(squad_v2_url, squad_v2_path)
    preprocess_squad_v2(squad_v2_path)
