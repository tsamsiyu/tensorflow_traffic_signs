import pandas as pd


def main():
    inputs = pd.read_csv('materials/full-gt-with-ids.csv')
    counts = {}
    for key, row in inputs.iterrows():
        if row['sign_class'] not in counts:
            counts[row['sign_class']] = 1
        else:
            counts[row['sign_class']] += 1
    for key in counts:
        if counts[key] > 1000:
            print(str(key) + " : " + str(counts[key]))


main()