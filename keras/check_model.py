from keras.models import load_model
import json
import csv
import nltk
import numpy as np
import pickle
import random
import helpers

model = load_model('chatbot_model.h5')
nltk.download('punkt')

intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

def bag_of_words(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = helpers.preprocess_question(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bag_of_words(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if (i['tag'] == tag):
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(msg):
    predictions = predict_class(msg, model)
    return getResponse(predictions, intents) if len(predictions) > 0 else False

if __name__ == "__main__":
    with open('../SampleDataset-ParentAndChildrenWithTitles.csv', 'r') as dataset_file:
        dataset = list(csv.DictReader(dataset_file, skipinitialspace=True))

    successes = 0
    matches = 0
    for row in dataset:
        response = chatbot_response(row['Title'])
        if response is not False:
            matches += 1
            success = response == row['Id']
            print(f"{response}, success: {response == row['Id']}")

            if success:
                successes += 1

    print(f"{successes} successes")
    print(f"{successes / len(dataset)} success rate")

    print(f"{matches} matches")
    print(f"{successes / matches} match success rate")

    print(f"{matches / len(dataset)} match rate")