import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


lemmatizer = WordNetLemmatizer()

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

ignore_words = set(stopwords.words('english'))

def preprocess_question(question):
    # Tokenize the word into its keywords
    title_tokens = set(nltk.word_tokenize(question))

    # Remove stopwords from the set of tokens
    title_tokens.difference_update(ignore_words)

    # Converting all words into their pure forms
    return set(map(lambda word: lemmatizer.lemmatize(word.lower()), title_tokens))