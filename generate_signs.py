import pandas as pd


signs = {
    '2_1': {'name' : 'main_road', 'id': 1},
    '5_19_1': {'name': 'crosswalk', 'id' : 2},
    '3_27': {'name' : 'stop_is_forbidden', 'id' : 3},
    '2_4': {'name' : 'give_a_way', 'id': 4},
    '5_16': {'name': 'bus_stop', 'id' : 5},
}


def main():
    inputs = pd.read_csv('materials/full-gt-with-ids.csv')
    inputs = inputs.sample(frac=1)

    columns = ['filename', 'xmin', 'xmax', 'ymin', 'ymax', 'sign_id', 'sign_class']
    trains = []
    evals = []

    trains_indexes = {}
    evals_indexes = {}

    for key, row in inputs.iterrows():
        if row['sign_class'] in signs:
            if row['sign_class'] not in trains_indexes:
                trains_indexes[row['sign_class']] = 0
            if row['sign_class'] not in evals_indexes:
                evals_indexes[row['sign_class']] = 0
            if evals_indexes[row['sign_class']] < 500:
                evals.append([
                    str(row['filename']),
                    int(row['x_from']),
                    int(row['x_from'] + row['width']),
                    int(row['y_from']),
                    int(row['y_from'] + row['height']),
                    str(signs[row["sign_class"]]['id']),
                    str(signs[row["sign_class"]]['name'])
                ])
                evals_indexes[row['sign_class']] += 1
            else:
                if trains_indexes[row['sign_class']] < 4000:
                    trains.append([
                        str(row['filename']),
                        int(row['x_from']),
                        int(row['x_from'] + row['width']),
                        int(row['y_from']),
                        int(row['y_from'] + row['height']),
                        str(signs[row["sign_class"]]['id']),
                        str(signs[row["sign_class"]]['name'])
                    ])
                    trains_indexes[row['sign_class']] += 1
    print(trains_indexes)
    print(evals_indexes)
    df = pd.DataFrame(trains, columns=columns)
    df = df.sample(frac=1)
    df.to_csv('materials/optimized_freq_train.csv', index=False)
    df = pd.DataFrame(evals, columns=columns)
    df = df.sample(frac=1)
    df.to_csv('materials/optimized_freq_eval.csv', index=False)


main()