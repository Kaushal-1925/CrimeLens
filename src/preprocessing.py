"""
Crime Detection NLP - Preprocessing Module
Provides reproducible, modular text cleaning routines that preserve key negation words
and crime terminology for TF-IDF feature extraction and model inference.
"""

import re
import string
import unicodedata
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure required NLTK resources are available
for resource in ['stopwords', 'wordnet', 'omw-1.4', 'punkt']:
    try:
        nltk.data.find(f'corpora/{resource}' if resource in ['stopwords', 'wordnet', 'omw-1.4'] else f'tokenizers/{resource}')
    except LookupError:
        nltk.download(resource, quiet=True)

# Build custom stopwords set preserving negation words
# Negation words significantly alter the semantic valence of incident reports
NEGATION_WORDS = {
    'no', 'not', 'nor', 'neither', 'never', 'none', 'nobody',
    'nowhere', 'hardly', 'scarcely', 'barely', 'without', 'against'
}

STANDARD_STOPWORDS = set(stopwords.words('english'))
CUSTOM_STOPWORDS = STANDARD_STOPWORDS - NEGATION_WORDS
LEMMATIZER = WordNetLemmatizer()


def clean_text(text: str) -> str:
    """
    Standardized NLP cleaning pipeline for news headlines and incident reports.
    
    Steps:
    1. Handle non-string / NaN values safely.
    2. Unicode normalization (NFKD) to strip accented characters and weird glyphs.
    3. Lowercase conversion.
    4. Remove URLs, web links, and email formats.
    5. Remove punctuation while preserving words and intra-word boundaries.
    6. Tokenize by whitespace.
    7. Remove non-negation stopwords.
    8. Lemmatize words to their dictionary root form.
    9. Filter single characters (except meaningful numerals if desired) and collapse whitespace.
    """
    if not isinstance(text, str):
        return ""
    
    # 1. Unicode normalization
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    
    # 2. Lowercase
    text = text.lower()
    
    # 3. Strip URLs
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    
    # 4. Strip punctuation and symbols (replace with space)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', text)
    
    # 5. Tokenize, remove stopwords (preserving negations), and lemmatize
    tokens = text.split()
    processed_tokens = [
        LEMMATIZER.lemmatize(word)
        for word in tokens
        if word not in CUSTOM_STOPWORDS and len(word) > 1 and not word.isdigit()
    ]
    
    return " ".join(processed_tokens)

