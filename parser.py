# adding a separate field for easier classification

import csv
import pandas as pd
from sklearn.model_selection import train_test_split

def limit_to_one_category(category_to_limit_to):
    input_file = 'data/labeled.csv'
    output_file = f'data/FiveCats/labeled_processed_only_{category_to_limit_to}.csv'

    if category_to_limit_to == 0:
        category_mapping = {
            'ML_3_5': 1,
            'ML_3_4': 1,
            'ML_3_3': 1,
            'ML_3_2': 1,
            'ML_3_1': 1,
            'ML_3': 1,

            'ML_2_7_2': 1,
            'ML_2_7_1': 1,
            'ML_2_6': 1,
            'ML_2_5_1': 1,
            'ML_2_4_2': 1,
            'ML_2_4_1': 1,
            'ML_2_3_1': 1,
            'ML_2_3': 1,
            'ML_2_2': 1,
            'ML_2_1_3': 1,
            'ML_2_1_2': 1,
            'ML_2_1_1': 1,
            'ML_2': 1,

            'ML_1_2': 1,
            'ML_1_1_1': 1,
            'ML_1_1': 1,
            'ML_1': 1,
        }

    elif category_to_limit_to == 1:
        category_mapping = {
            'ML_3_5': 0,
            'ML_3_4': 0,
            'ML_3_3': 0,
            'ML_3_2': 0,
            'ML_3_1': 0,
            'ML_3': 0,

            'ML_2_7_2': 0,
            'ML_2_7_1': 0,
            'ML_2_6': 0,
            'ML_2_5_1': 0,
            'ML_2_4_2': 0,
            'ML_2_4_1': 0,
            'ML_2_3_1': 0,
            'ML_2_3': 0,
            'ML_2_2': 0,
            'ML_2_1_3': 0,
            'ML_2_1_2': 0,
            'ML_2_1_1': 0,
            'ML_2': 0,

            'ML_1_2': 12,
            'ML_1_1_1': 111,
            'ML_1_1': 11,
            'ML_1': 1,
        }

    elif category_to_limit_to == 2:
        category_mapping = {
            'ML_3_5': 0,
            'ML_3_4': 0,
            'ML_3_3': 0,
            'ML_3_2': 0,
            'ML_3_1': 0,
            'ML_3': 0,

            'ML_2_7_2': 272,
            'ML_2_7_1': 271,
            'ML_2_6': 26,
            'ML_2_5_1': 251,
            'ML_2_4_2': 242,
            'ML_2_4_1': 241,
            'ML_2_3_1': 231,
            'ML_2_3': 23,
            'ML_2_2': 22,
            'ML_2_1_3': 213,
            'ML_2_1_2': 212,
            'ML_2_1_1': 211,
            'ML_2': 2,

            'ML_1_2': 0,
            'ML_1_1_1': 0,
            'ML_1_1': 0,
            'ML_1': 0,
        }

    elif category_to_limit_to == 3:
        category_mapping = {
            'ML_3_5': 35,
            'ML_3_4': 34,
            'ML_3_3': 33,
            'ML_3_2': 32,
            'ML_3_1': 31,
            'ML_3': 3,

            'ML_2_7_2': 0,
            'ML_2_7_1': 0,
            'ML_2_6': 0,
            'ML_2_5_1': 0,
            'ML_2_4_2': 0,
            'ML_2_4_1': 0,
            'ML_2_3_1': 0,
            'ML_2_3': 0,
            'ML_2_2': 0,
            'ML_2_1_3': 0,
            'ML_2_1_2': 0,
            'ML_2_1_1': 0,
            'ML_2': 0,

            'ML_1_2': 0,
            'ML_1_1_1': 0,
            'ML_1_1': 0,
            'ML_1': 0,
        }

    index = 0

    with open(input_file, mode='r', newline='') as infile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames + ['event_type']

        with open(output_file, mode='w', newline='') as outfile:
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()

            for row in reader:
                index += 1
                print(index)

                event_type = 0

                for category in category_mapping:
                    if any(row[k] == '1.0' for k in row if k.startswith(category)):
                        event_type = category_mapping[category]
                        break

                row['event_type'] = event_type
                writer.writerow(row)


def four_categories():
    input_file = 'data/labeled.csv'
    output_file = 'data/labeled_processed_4Categories.csv'

    category_mapping = {
        'ML_3_5': 3,
        'ML_3_4': 3,
        'ML_3_3': 3,
        'ML_3_2': 3,
        'ML_3_1': 3,
        'ML_3': 3,

        'ML_2_7_2': 2,
        'ML_2_7_1': 2,
        'ML_2_6': 2,
        'ML_2_5_1': 2,
        'ML_2_4_2': 2,
        'ML_2_4_1': 2,
        'ML_2_3_1': 2,
        'ML_2_3': 2,
        'ML_2_2': 2,
        'ML_2_1_3': 2,
        'ML_2_1_2': 2,
        'ML_2_1_1': 2,
        'ML_2': 2,

        'ML_1_2': 1,
        'ML_1_1_1': 1,
        'ML_1_1': 1,
        'ML_1': 1,
    }

    index = 0

    with open(input_file, mode='r', newline='') as infile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames + ['event_type']

        with open(output_file, mode='w', newline='') as outfile:
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()

            for row in reader:
                event_type = 0

                for category in category_mapping:
                    if any(row[k] == '1.0' for k in row if k.startswith(category)):
                        event_type = category_mapping[category]
                        break

                row['event_type'] = event_type
                writer.writerow(row)

                index += 1
                print(index)


def divide_to_traintest_val(category):
    input_file = f'data/FiveCats/labeled_processed_only_{category}.csv'
    output_traintest_file = f'data/FiveCats/only{category}_traintest.csv'
    output_validation_file = f'data/FiveCats/only{category}_validation.csv'

    labeled_dataset = pd.read_csv(input_file)

    traintest, val = train_test_split(
        labeled_dataset,
        test_size=0.15,
        random_state=1,
        stratify=labeled_dataset['event_type']
    )
    traintest.to_csv(output_traintest_file, index=False)
    val.to_csv(output_validation_file, index=False)

def divide_processed_to_traintest_val():
    input_file = f'data/labeled_processed.csv'
    output_traintest_file = f'data/labeled_processed_traintest.csv'
    output_validation_file = f'data/labeled_processed_validation.csv'

    labeled_dataset = pd.read_csv(input_file)

    traintest, val = train_test_split(
        labeled_dataset,
        test_size=0.15,
        random_state=1,
        stratify=labeled_dataset['event_type']
    )
    traintest.to_csv(output_traintest_file, index=False)
    val.to_csv(output_validation_file, index=False)

divide_processed_to_traintest_val()
#four_categories()
#limit_to_one_category(0)
#limit_to_one_category(1)
#divide_to_traintest_val(1)

#limit_to_one_category(2)
#divide_to_traintest_val(2)

#limit_to_one_category(3)
#divide_to_traintest_val(3)
