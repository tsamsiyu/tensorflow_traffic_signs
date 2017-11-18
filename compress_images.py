import re
from PIL import Image
import pandas as pd
import os
import threading


from_dir = '/media/tsamsiyu/985A6DDF5A6DBAA0/Users/Dmitry/Downloads/full-frames.tar/full-frames/rtsd-frames'
to_base_dir = 'images'
meta_csv_path = 'materials/full-gt.csv'
img_name_regex = re.compile('^autosave(\d\d)_(\d\d)_(\d\d\d\d)_(.+\..+)$')
use_threads = 3


def compress(from_file_path, to_file_path):
    img = Image.open(from_file_path)
    img.resize((640, 480), Image.ANTIALIAS)
    img.save(to_file_path, optimize=True)


def detect_file_locations(filename):
    (day, month, year, new_file_name) = img_name_regex.match(filename).groups()
    to_file_dir = to_base_dir + '/' + year + '/' + month + '/' + day
    to_file_path = to_file_dir + '/' + new_file_name
    if os.path.exists(to_file_path):
        return None
    if not os.path.isdir(to_file_dir):
        os.makedirs(to_file_dir)
    from_file_path = from_dir + '/' + filename
    return from_file_path, to_file_path


def run_threads(threads):
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()


def main():
    examples = pd.read_csv(meta_csv_path)
    charged_threads = []
    handled = {}
    for key, row in examples.iterrows():
        if row['filename'] in handled:
            continue
        handled[row['filename']] = True
        file_locations = detect_file_locations(row['filename'])
        if file_locations is not None:
            t = threading.Thread(target=compress, args=file_locations)
            charged_threads.append(t)
            if len(charged_threads) == use_threads:
                run_threads(charged_threads)
                charged_threads.clear()


main()