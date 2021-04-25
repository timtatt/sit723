import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import csv
import contractions
import json
import re

lemmatizer = WordNetLemmatizer()

nltk.download('punkt')
nltk.download('stopwords')

ignore_tokens = list(stopwords.words('english'))
ignore_tokens += ['.', '?', '(', ')', '/', '``', ',', '\'\'','""', ':', '-', '\'', '{', '}', '`', '[', ']', ';', '//', '<', '>', 'i','^','~']

ignore_patterns = [] #[r'^[\.\_\^]+$', r'//{0,1}.+', r'\\.+', r'^={3,}$', r'\d{4}-\d{2}-\d{2}', r'^www\.', r'^[a-z]$', r'\d{2}:\d{2}:\d{2}', r'[^\x00-\x7F]']

synonym_refs = {}

with open('programming_synonyms.csv', 'r') as programming_synonyms_file:
    synonym_groups = list(csv.reader(programming_synonyms_file, delimiter=','))
    for row_index in range(len(synonym_groups)):
        for word in synonym_groups[row_index]:
            synonym_refs[word] = row_index

def not_matches_a_pattern(token):
    for pattern in ignore_patterns:
        if re.search(pattern, token) is not None:
            return False
    return True

def preprocess_html(html):
    # Remove html tags
    html_without_code = re.sub(r'(<code>.*<\/code>)', '', html, flags=re.DOTALL)

    # Remove html tags
    html_sanitized_question = re.sub(re.compile('<.*?>'), '', html_without_code)

    return html_sanitized_question

def preprocess_text(question):
    # Expand contractions
    expanded_question = ' '.join(list(map(lambda word: contractions.fix(word), question.split(' '))))

    # Tokenize the word into its keywords
    title_tokens = nltk.word_tokenize(expanded_question)

    # Lowercase tokens
    lowercase_tokens = map(lambda token: token.lower(), title_tokens)

    # Remove preceding '
    # sanitized_tokens = map(lambda token: token[1:] if token[0] in ['\'','|'] else token, lowercase_tokens)

    # Filter out stopwords and punctuation
    filtered_tokens = filter(lambda token: token not in ignore_tokens, lowercase_tokens)
    # filtered_tokens = filter(not_matches_a_pattern, filtered_tokens)

    # Converting all words into their pure forms
    lemmatized_tokens = map(lambda word: lemmatizer.lemmatize(word.lower()), filtered_tokens)

    return set(lemmatized_tokens)

def token_variation_sets(question):
    tokens = preprocess_text(question)
    token_sets = [tokens]
    for token in tokens:
        additional_sets = []
        for synonym in get_synonyms(token):
            if synonym != token:
                for tset in token_sets:
                    current_set = tset.copy()
                    current_set.remove(token)
                    current_set.add(synonym)
                    additional_sets.append(current_set)

        token_sets.extend(additional_sets)

    return token_sets


def get_synonyms(word):
    return synonym_groups[synonym_refs[word]] if word in synonym_refs else []

if __name__ == "__main__":
    with open("../dataset.json") as dataset_file:
        for (post_id, post) in json.load(dataset_file).items():
            print(post['Title'])
            print(preprocess_text(post['Body']))
            for child in post['Children']:
                print(preprocess_text(child['Body']))