from functools import reduce

import whoosh.index as index
import json

import helpers
from index_posts import schema
from whoosh.qparser import QueryParser

ix = index.open_dir("indexdir")
searcher = ix.searcher()

qp = QueryParser('tokens', schema=schema)

if __name__ == "__main__":
    successes = dict(zip(range(0, 20), [0]*20))
    with open('../dataset.json') as dataset_file:
        for (post_id, post) in json.load(dataset_file).items():
            q = qp.parse(' '.join(helpers.preprocess_text(post['Children'][0]['Title'])))
            print(q)
            print(post['Children'][0]['Title'])
            print(post['Title'])
            results = searcher.search(q, limit=20)
            if len(results) > 0:
                for index, result in enumerate(results):
                    if result['post_id'] == post_id:
                        successes[index] += 1
                        print(result['title'])
                        print('Success: True')
                        break
    searcher.close()
    print("Successes", successes)
    print(reduce(lambda a, b: a + b, successes.values()), "/476")