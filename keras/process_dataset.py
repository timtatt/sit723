import csv
import json
import re
import html

if __name__ == '__main__':
    cleaned_dataset = {}

    with open('../SampleDataset-4.csv', 'r') as dataset_file:
        dataset = list(csv.DictReader(dataset_file, skipinitialspace=True))

    for row in dataset:
        if row['ParentPostId'] not in cleaned_dataset:
            cleaned_dataset[row['ParentPostId']] = {
                'Id': row['ParentPostId'],
                'Tags': re.findall("<([^>]+)>", row['ParentTags']),
                'Title': row['ParentPostTitle'],
                'Body': row['ParentBody'],
                'Children': []
            }

        cleaned_dataset[row['ParentPostId']]['Children'].append({
            'Id': row['Id'],
            'Title': row['Title'],
            'Tags': re.findall("<([^>]+)>", row['Tags']),
            'Body': html.unescape(row['Body']),
        })

        print("Processing", row['Id'], row['Title'])

    filtered_dataset = {k: v for (k, v) in cleaned_dataset.items() if len(v['Children']) > 2}

    with open('../dataset.json', 'w+') as dataset_file:
        json.dump(filtered_dataset, dataset_file, indent=2)