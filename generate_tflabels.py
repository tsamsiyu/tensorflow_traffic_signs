import pandas as pd


def main():
    examples = pd.read_csv('materials/full-gt-with-ids.csv')
    labels = {}
    labels_string = ''
    for key, row in examples.iterrows():
        if row['sign_class_id'] not in labels:
            labels_string += """
item {
    id : %s
    name : '%s'
}""" % (row['sign_class_id'], row['sign_class'])
            labels[row['sign_class_id']] = 1

    with open('all-labels.pbtxt', 'w') as f:
        f.write(labels_string)
    print('Successfully created the labels for TFRecords: ' + str(len(labels)))


main()