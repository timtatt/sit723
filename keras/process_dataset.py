import csv
import json
import re
import helpers

if __name__ == '__main__':
    cleaned_dataset = {}

    with open('../SampleDataset-Java-1.csv', 'r') as dataset_file:
        dataset = list(csv.DictReader(dataset_file, skipinitialspace=True))

    for row in dataset:
        if row['ParentPostId'] not in cleaned_dataset:
            cleaned_dataset[row['ParentPostId']] = {
                'Id': row['ParentPostId'],
                'Tags': re.findall("<([^>]+)>", row['ParentTags']),
                'Title': row['ParentPostTitle'],
                'BodyTokens': helpers.tokenize_body(row['Body']),
                'Children': []
            }

        cleaned_dataset[row['ParentPostId']]['Children'].append({
            'Id': row['Id'],
            'Title': row['Title'],
            'Tags': re.findall("<([^>]+)>", row['Tags']),
            'BodyTokens': helpers.tokenize_body(row['Body']),
        })

        print("Processing", row['Id'], row['Title'])
        # print(helpers.preprocess_text(row['Title']))
        # print()

    filtered_dataset = {k: v for (k, v) in cleaned_dataset.items() if len(v['Children']) > 2}

    print(len(filtered_dataset), "master questions")

    with open('../dataset.json', 'w+') as dataset_file:
        json.dump(filtered_dataset, dataset_file, indent=2)