import shutil
from whoosh.fields import Schema, TEXT, KEYWORD, ID, STORED
from whoosh.analysis import StemmingAnalyzer
import os
from functools import reduce
import helpers
import json
from whoosh import index

schema = Schema(post_id=ID(stored=True),
                title=TEXT(stored=True),
                tokens=KEYWORD(stored=True, commas=True, scorable=True))

if __name__ == "__main__":
    if os.path.exists("indexdir"):
        shutil.rmtree("indexdir")

    os.mkdir("indexdir")

    ix = index.create_in("indexdir", schema)
    writer = ix.writer()

    with open('../dataset.json') as dataset_file:
        for (post_id, post) in json.load(dataset_file).items():
            terms = set(post['BodyTokens'])
            terms = terms.union(set(helpers.preprocess_text(post['Title'])))

            child_body_terms = reduce(lambda child_one, child_two: child_one.union(child_two), map(lambda child: set(child['BodyTokens']), post['Children'][1:]))
            terms = terms.union(child_body_terms)

            child_title_terms = reduce(lambda child_one, child_two: child_one.union(child_two), map(lambda child: set(helpers.preprocess_text(child['Title'])), post['Children']))
            terms = terms.union(child_title_terms)

            child_tags = reduce(lambda child_one, child_two: child_one.union(child_two), map(lambda child: set(child['Tags']), post['Children']))
            terms = terms.union(child_tags)
            terms = terms.union(set(post['Tags']))

            writer.add_document(post_id=post_id, title=post['Title'], tokens=','.join(terms))
            print(post['Title'])
        writer.commit()

    print(ix.doc_count())