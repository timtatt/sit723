import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import csv
import contractions
import json
from html import unescape
import re
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import text_to_word_sequence

lemmatizer = WordNetLemmatizer()

nltk.download('punkt')
nltk.download('stopwords')

ignore_tokens = list(stopwords.words('english'))
ignore_tokens += ['.', '?', '(', ')', '/', '``', ',', '\'\'','""', ':', '-', '\'', '{', '}', '`', '[', ']', ';', '//', '<', '>', 'i','^','~']

ignore_patterns = [] #[r'^[\.\_\^]+$', r'//{0,1}.+', r'\\.+', r'^={3,}$', r'\d{4}-\d{2}-\d{2}', r'^www\.', r'^[a-z]$', r'\d{2}:\d{2}:\d{2}', r'[^\x00-\x7F]']

synonym_refs = {}

with open('programming_synonyms.csv', 'r') as programming_synonyms_file:
    synonym_groups = list(csv.reader(programming_synonyms_file, delimiter=','))
    for row in synonym_groups:
        replacement = row[0]
        for word in row[1:]:
            synonym_refs[word] = replacement

def group_tokens_by_count(tokens):
    token_group = {}
    for token in tokens:
        synonym_token = replace_with_synonym_if_available(token)
        if synonym_token in token_group:
            token_group[synonym_token] += 1
        else:
            token_group[synonym_token] = 1
    return token_group

def tokenize_body(body):
    processed_html = preprocess_html(body)

    expanded_html = ' '.join(list(map(lambda word: contractions.fix(word), processed_html.split(' '))))

    body_tokens = text_to_word_sequence(expanded_html)

    filtered_tokens = filter(lambda token: token not in ignore_tokens, body_tokens)

    lemmatized_tokens = map(lambda word: lemmatizer.lemmatize(word), filtered_tokens)

    grouped_tokens = group_tokens_by_count(lemmatized_tokens)

    multiple_tokens = {k: v for (k, v) in grouped_tokens.items() if v > 1}

    return list(multiple_tokens.keys())

def not_matches_a_pattern(token):
    for pattern in ignore_patterns:
        if re.search(pattern, token) is not None:
            return False
    return True

def preprocess_html(html):
    unescaped_html = unescape(html)

    # Remove html tags
    html_without_code = re.sub(r'(<code>.*<\/code>)', '', unescaped_html, flags=re.DOTALL)

    # Remove html tags
    html_sanitized_question = re.sub(re.compile('<.*?>'), '', html_without_code)

    return html_sanitized_question

def replace_with_synonym_if_available(word):
    return synonym_refs[word] if word in synonym_refs else word

def preprocess_text(question, use_keras=True):
    # Expand contractions
    expanded_question = ' '.join(list(map(lambda word: contractions.fix(word), question.split(' '))))

    # Tokenize the word into its keywords
    if not use_keras:
        title_tokens = nltk.word_tokenize(expanded_question)
    else:
        title_tokens = text_to_word_sequence(expanded_question)

    # Lowercase tokens
    lowercase_tokens = map(lambda token: token.lower(), title_tokens)

    # Filter out stopwords and punctuation
    filtered_tokens = filter(lambda token: token not in ignore_tokens, lowercase_tokens)
    # filtered_tokens = filter(not_matches_a_pattern, filtered_tokens)

    # Converting all words into their pure forms
    lemmatized_tokens = map(lambda word: lemmatizer.lemmatize(word), filtered_tokens)

    # Replaced synonyms
    synonym_replaced_tokens = map(replace_with_synonym_if_available, lemmatized_tokens)

    return set(synonym_replaced_tokens)

if __name__ == "__main__":
    with open("../dataset.json") as dataset_file:
        for (post_id, post) in json.load(dataset_file).items():
            print(post['Title'])
            print(preprocess_text(post['Body']))
            for child in post['Children']:
                print(preprocess_text(child['Body']))