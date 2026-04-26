# adding a separate field for easier classification

import csv

input_file = 'data/labeled.csv'
output_file = 'data/labeled_processed.csv'

CATEGORY_MAPPING = {
    'ML_3_5': 3.5,
    'ML_3_4': 3.4,
    'ML_3_3': 3.3,
    'ML_3_2': 3.2,
    'ML_3_1': 3.1,
    'ML_3': 3.0,

    'ML_2_7_2': 2.72,
    'ML_2_7_1': 2.71,
    'ML_2_6': 2.6,
    'ML_2_5_1': 2.51,
    'ML_2_4_2': 2.42,
    'ML_2_4_1': 2.41,
    'ML_2_3_1': 2.31,
    'ML_2_3': 2.3,
    'ML_2_2': 2.2,
    'ML_2_1_3': 2.13,
    'ML_2_1_2': 2.12,
    'ML_2_1_1': 2.11,
    'ML_2': 2.0,

    'ML_1_2': 1.2,
    'ML_1_1_1': 1.11,
    'ML_1_1': 1.1,
    'ML_1': 1.0,
}

index = 0

with open(input_file, mode='r', newline='') as infile:
    reader = csv.DictReader(infile)
    fieldnames = reader.fieldnames + ['event_type']

    with open(output_file, mode='w', newline='') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            event_type = 0.0

            for category in CATEGORY_MAPPING:
                if any(row[k] == '1.0' for k in row if k.startswith(category)):
                    event_type = CATEGORY_MAPPING[category]
                    break

            row['event_type'] = event_type
            writer.writerow(row)

            index += 1
            print(index)