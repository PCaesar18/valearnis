import nltk
import re
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords


def preprocess(text):
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)

    # Tokenize
    text = nltk.word_tokenize(text)

    # Remove stopwords
    text = [word for word in text if word not in stopwords.words('english')]

    # Stemming
    text = [PorterStemmer().stem(word) for word in text]

    # Lemmatization
    text = [WordNetLemmatizer().lemmatize(word) for word in text]

    # Join tokens back into string
    text = ' '.join(text)

    return text
