import json
import pickle
import random

import numpy as np
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import SGD

import helpers

words = set()
classes = set()
documents = []

if __name__ == "__main__":
    # Processing CSV file to produce JSON model
    with open('../dataset.json', 'r') as dataset_file:
        dataset = json.load(dataset_file)

    counter = 0

    intents = []
    for (post_id, post) in dataset.items():
        child_titles = list(map(lambda child: child['Title'], post['Children']))
        child_titles.append(post['Title'])

        tag = post['Id']

        intents.append({
            'patterns': child_titles,
            'responses': [post['Id']],
            'tag': tag,
        })

        for title in child_titles:
            title_tokens = helpers.preprocess_text(title)

            title_tokens = title_tokens.union(set(post['Tags']))
            title_tokens = title_tokens.union(set(post['BodyTokens']))

            # Add keywords to set of words
            words.update(title_tokens)

            # Linking words to a tag
            documents.append((title_tokens, tag))

            # Adding the tag to the list of classes
            classes.add(tag)

        counter += 1

    # Storing the intents for further examination
    with open('intents.json', 'w+') as intents_json:
        json.dump({
            'intents': intents,
        }, intents_json, indent=2)

    words = sorted(words)
    classes = sorted(classes)

    print(len(documents), "documents")
    print(len(classes), "classes")
    print(len(words), "unique lemmatized words", words)
    print(len(words), "words")

    pickle.dump(words, open('words.pkl', 'wb'))
    pickle.dump(classes, open('classes.pkl', 'wb'))

    # Producing Training Data
    training = []
    output_empty = [0] * len(classes)
    for (pattern_words, class_name) in documents:
        # initializing bag of words
        bag = []

        # create our bag of words array with 1, if word match found in current pattern
        for w in words:
            # word_count = pattern_words.count(w)
            bag.append(1 if w in pattern_words else 0)

        # output is a '0' for each tag and '1' for current tag (for each pattern)
        output_row = list(output_empty)
        output_row[classes.index(class_name)] = 1

        training.append([bag, output_row])

    # shuffle our features and turn into np.array
    random.shuffle(training)
    training = np.array(training)

    # create train and test lists. X - patterns, Y - intents
    train_x = list(training[:,0])
    train_y = list(training[:,1])
    print("Training data created")

    # Create model - 3 layers. First layer 128 neurons, second layer 64 neurons and 3rd output layer contains number of neurons
    # equal to number of intents to predict output intent with softmax
    model = Sequential()
    model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(train_y[0]), activation='softmax'))

    # Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    # fitting and saving the model
    hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
    model.save('chatbot_model.h5', hist)

    print("model created")