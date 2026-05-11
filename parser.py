# adding a separate field for easier classification

import csv

input_file = 'data/labeled.csv'
output_file = 'data/labeled_processed.csv'

CATEGORY_MAPPING = {
    'ML_3_5': 35,
    'ML_3_4': 34,
    'ML_3_3': 33,
    'ML_3_2': 32,
    'ML_3_1': 31,
    'ML_3': 3,

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

    'ML_1_2': 12,
    'ML_1_1_1': 111,
    'ML_1_1': 11,
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

            for category in CATEGORY_MAPPING:
                if any(row[k] == '1.0' for k in row if k.startswith(category)):
                    event_type = CATEGORY_MAPPING[category]
                    break

            row['event_type'] = event_type
            writer.writerow(row)

            index += 1
            print(index)