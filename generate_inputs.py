import sys
import pandas as pd


def main():
    times = {}
    inputs = pd.read_csv('materials/full-gt-with-ids.csv')
    drop_indexes = []
    step = int(sys.argv[1])
    for key, row in inputs.iterrows():
        tmst = int(row['timestamp'])
        if any(key in times for key in range(tmst - step, tmst + step)):
            drop_indexes.append(key)
        else:
            times[tmst] = True
    inputs = inputs.drop(drop_indexes)
    inputs = inputs.sample(frac=1)
    inputs.to_csv('materials/optimized' + str(step) + '.csv', index=False)


main()