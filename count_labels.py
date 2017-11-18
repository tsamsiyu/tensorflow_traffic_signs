import pandas as pd
import tensorflow as tf
from collections import namedtuple, OrderedDict

flags = tf.app.flags
flags.DEFINE_string('name', '', 'Path to the CSV input')
FLAGS = flags.FLAGS


def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def main():
    examples = pd.read_csv('materials/' + FLAGS.name + '.csv')
    grouped = split(examples, 'sign_class')
    print("Count:" + str(len(grouped)))

main()
