import csv
from chatbot import bot

if __name__ == "__main__":
    results = []
    successes = 0
    with open('../SampleDataset-ParentAndChildrenWithTitles.csv', 'r') as dataset_file:
        dataset = list(csv.DictReader(dataset_file, skipinitialspace=True))

        for row in dataset:
            response = bot.get_response(row['Title'])

            row_results = {
                'row': row['Id'],
                'title': row['Title'],
                'result': response.text,
                'success': response.text == row['Id']
            }

            if response.text == row['Id']:
                successes += 1

            results.append(row_results)

            print(row_results)

        print("Successes " + str(successes))
        print("Success Rate:" + str(len(results) / successes))
