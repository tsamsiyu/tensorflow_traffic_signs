from datetime import datetime
import re
import pandas as pd


labels = {}
ind = 1
img_name_regex = re.compile('^autosave(\d\d)_(\d\d)_(\d\d\d\d)_(\d\d)_(\d\d)_(\d\d).*$')


def map_sign_class_id(sign_class):
    global ind
    if sign_class not in labels:
        labels[sign_class] = ind
        ind = ind + 1
    return labels[sign_class]


def map_timestamp(filename):
    (day, month, year, hour, minute, second) = img_name_regex.match(filename).groups()
    return datetime(int(year), int(month), int(day), int(hour), int(minute), int(second)).timestamp()


def main():
    examples = pd.read_csv('materials/full-gt.csv')
    examples['sign_class_id'] = examples['sign_class'].map(map_sign_class_id)
    examples['timestamp'] = examples['filename'].map(map_timestamp)
    examples.to_csv('materials/full-gt-with-ids.csv', index=False)


main()