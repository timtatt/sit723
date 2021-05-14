from keras.models import load_model
import json
import nltk
import numpy as np
import pickle
import helpers

model = load_model('chatbot_model.h5')
nltk.download('punkt')

intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

def bag_of_words(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = helpers.preprocess_text(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    p = bag_of_words(sentence, words, show_details=False)
    predictions = model.predict(np.array([p]))[0]

    # filter out predictions below a threshold
    error_threshold = 0.15
    results = [[i, r] for i, r in enumerate(predictions) if r > error_threshold]

    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)

    return list(map(lambda x: {
        "intent": classes[x[0]],
        "probability": str(x[1])
    }, results))

def chatbot_response(msg):
    predictions = predict_class(msg, model)
    post_ids = list(map(lambda prediction: prediction['intent'], predictions))
    return post_ids if len(predictions) > 0 else False

if __name__ == "__main__":
    with open('../dataset.json', 'r') as dataset_file:
        dataset = json.load(dataset_file)

    successes = 0
    partial_success = 0
    matches = 0
    for (post_id, post) in dataset.items():
        test_title = post['Children'][0]['Title'] + ' ' + ' '.join(post['Children'][0]['Tags'] + post['Children'][0]['BodyTokens'])
        response = chatbot_response(test_title)
        if response is not False:
            matches += 1
            success = response[0] == post_id
            # print(f"{response}, success: {response == row['Id']}")

            if success:
                successes += 1
            elif post_id in response:
                partial_success += 1
            else:
                print(f"{response}, {test_title}, success: {response == post_id}")
                for child in dataset[post_id]['Children']:
                    print(child['Title'], child['Tags'])
                print('-' * 10)
                for child in dataset[response[0]]['Children']:
                    print(child['Title'], child['Tags'])
                print()

        else:
            print(f"Failed to find response {post_id}: '{test_title}'")
            print(helpers.preprocess_text(test_title))


    print(f"{successes} successes")
    print(f"{successes / len(dataset)} success rate")

    print(f"{partial_success} partial successes")
    print(f"{partial_success / (matches - successes)} partial success rate")

    print(f"{matches} matches")
    print(f"{successes / matches} match success rate")

    print(f"{matches / len(dataset)} match rate")