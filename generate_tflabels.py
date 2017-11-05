"""
3_24_n:     5, 30, 40, 50, 60
3_13_r:     2.5, 4.5
"""

import pandas as pd


def unique_labels(path, labels_names):
    examples = pd.read_csv(path)
    for index, row in examples.iterrows():
        labels_names[row['sign_id']] = row['sign_class']


def main():
    labels = {}
    unique_labels('data/rtsd-train.csv', labels)
    unique_labels('data/rtsd-test.csv', labels)
    labels_string = ''
    for key, value in labels.items():
        labels_string += 'item {\n' \
                            '\tid : ' + str(key) + '\n'\
                            '\tname : \'' + str(value) + '\'\n'\
                         '}\n'
    with open('rtsd-labels.pbtxt', 'w') as f:
        f.write(labels_string)
    print('Successfully created the labels for TFRecords: ' + len(labels))


main()