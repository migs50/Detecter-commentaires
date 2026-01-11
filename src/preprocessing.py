import pandas as pd
import re

# Hardcoded common English stopwords
STOPWORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd",
    'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers',
    'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
    'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been',
    'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if',
    'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between',
    'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out',
    'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
    'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
    'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should',
    "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't",
    'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't",
    'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't",
    'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"
}

def clean_text(text):
    """
    Perform basic text cleaning:
    - Lowercase
    - Remove punctuation
    - Remove special characters/numbers
    - Tokenization (Regex)
    - Remove stopwords (Hardcoded)
    """
    if not isinstance(text, str):
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Remove punctuation and special characters/numbers using regex
    text = re.sub(r'[^a-z\s]', '', text)
    
    # Simple whitespace tokenization
    tokens = text.split()
    
    # Remove stopwords
    tokens = [t for t in tokens if t not in STOPWORDS]
    
    return " ".join(tokens)

def preprocess_dataframe(df, text_column='comment_text'):
    """
    Apply preprocessing to the entire dataframe:
    - Handle missing values
    - Clean text column
    """
    # Create a copy to avoid modifying the original
    df = df.copy()
    
    # Handle missing values in text column
    df[text_column] = df[text_column].fillna("")
    
    # Apply cleaning
    print("Cleaning text data... this may take a moment.")
    df['cleaned_text'] = df[text_column].apply(clean_text)
    
    # Binary classification: if any toxicity label is 1, then toxic = 1
    toxicity_labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    df['is_toxic'] = df[toxicity_labels].max(axis=1)
    
    return df
