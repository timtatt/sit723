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

    # Remove punctuation
    alphanumeric_tokens = filter(lambda word: word.isalpha(), title_tokens)

    # Converting all words into their pure forms
    lemmatized_tokens = map(lambda word: lemmatizer.lemmatize(word.lower()), alphanumeric_tokens)

    return set(lemmatized_tokens)
