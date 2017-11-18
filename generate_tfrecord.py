"""
Usage:
  python generate_tfrecord.py --name=sm_train_5k
"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import io
import re
import pandas as pd
import tensorflow as tf

from PIL import Image
from research.object_detection.utils import dataset_util
from collections import namedtuple, OrderedDict

flags = tf.app.flags
flags.DEFINE_string('name', '', 'Data name')
FLAGS = flags.FLAGS
img_name_regex = re.compile('^autosave(\d\d)_(\d\d)_(\d\d\d\d)_(.+\..+)$')
to_base_dir = 'images'


def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def file_location(filename):
    (day, month, year, new_file_name) = img_name_regex.match(filename).groups()
    return to_base_dir + '/' + year + '/' + month + '/' + day + '/' + new_file_name


def create_tf_example(group):
    path = file_location(group.filename)
    with tf.gfile.GFile(path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['x_from'] / width)
        xmaxs.append(row['x_from'] + row['width'] / width)
        ymins.append(row['y_from'] / height)
        ymaxs.append(row['y_from'] + row['height'] / height)
        classes_text.append(row['sign_class'].encode('utf8'))
        classes.append(row['sign_class_id'])

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def main(_):
    writer = tf.python_io.TFRecordWriter('records/' + FLAGS.name + '.record')
    examples = pd.read_csv('materials/' + FLAGS.name + '.csv')
    grouped = split(examples, 'filename')
    for group in grouped:
        tf_example = create_tf_example(group)
        writer.write(tf_example.SerializeToString())
    writer.close()
    output_path = os.path.join(os.getcwd(), 'records/' + FLAGS.name + '.record')
    print('Successfully created the TFRecords: {}'.format(output_path))


if __name__ == '__main__':
    tf.app.run()