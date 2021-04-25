import json
import helpers

if __name__ == "__main__":
    with open("../dataset.json") as dataset_file:
        for (post_id, post) in json.load(dataset_file).items():
            print(post['Title'])
            print(post['Children'][0]['Title'])
            print(' '.join(helpers.preprocess_text(post['Children'][0]['Title'])))
            print()