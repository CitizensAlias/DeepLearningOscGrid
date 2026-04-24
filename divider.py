# adding a separate field for easier classification

import csv

input_file = 'data/labeled.csv'
output_file = 'data/labeled_processed.csv'
index = 0

with open(input_file, mode='r', newline='') as infile:
    reader = csv.DictReader(infile)
    fieldnames = reader.fieldnames + ['event_type']

    with open(output_file, mode='w', newline='') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:

            event_type = 0.0

            if any(row[k] == '1.0' for k in row if k.startswith('ML_3')):
                event_type = 3.0
            elif any(row[k] == '1.0' for k in row if k.startswith('ML_2')):
                event_type = 2.0
            elif any(row[k] == '1.0' for k in row if k.startswith('ML_1')):
                event_type = 1.0

            row['event_type'] = event_type
            writer.writerow(row)

            index = index + 1
            print(index)
