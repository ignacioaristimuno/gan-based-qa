import re

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer


class TextPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
    def preprocess_text(self, text):
        # Remove punctuations and digits
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        
        # Tokenize the text into sentences
        sentences = sent_tokenize(text)
        
        # Tokenize each sentence into words, remove stop words, and lemmatize
        preprocessed_sentences = []
        for sentence in sentences:
            words = word_tokenize(sentence)
            words = [word for word in words if word not in self.stop_words]
            words = [self.lemmatizer.lemmatize(word) for word in words]
            preprocessed_sentences.append(words)
        
        return preprocessed_sentences
