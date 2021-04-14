import csv
import re
from chatbot import bot
from chatterbot.trainers import ListTrainer

if __name__ == "__main__":
    with open('../SampleDataset-ParentAndChildrenWithTitles.csv', 'r') as dataset_file:
        dataset = list(csv.DictReader(dataset_file, skipinitialspace=True))

    trainer = ListTrainer(bot)

    for row in dataset:
        row['Tags'] = re.findall("<([^>]+)>", row['Tags'])
        row['ChildTitles'] = row['ChildTitles'].split('||')

        for child_title in row['ChildTitles']:
            trainer.train([
                child_title,
                row['Id']
            ])



