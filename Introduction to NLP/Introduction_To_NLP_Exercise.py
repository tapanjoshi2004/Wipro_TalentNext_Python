# 1. .....................................................................
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def load_dataset(file_path):
    data = pd.read_csv(file_path, sep='\t', names=['label', 'message'])
    return data

def preprocess_text(text):

    tokens = word_tokenize(text.lower())
    
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
    
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return ' '.join(tokens)

file_path = 'SMSSpamCollection' 
data = load_dataset(file_path)
data['processed_message'] = data['message'].apply