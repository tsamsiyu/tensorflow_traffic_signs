from PIL import Image
import pandas as pd
import os
import threading


from_dir = '/media/tsamsiyu/985A6DDF5A6DBAA0/Users/Dmitry/Downloads/full-frames.tar/full-frames/rtsd-frames'
to_base_dir = 'images/freq'
use_threads = 3


def compress(from_file_path, to_file_path):
    img = Image.open(from_file_path)
    width, height = img.size
    img.resize((int(width / 2), int(height / 2)), Image.ANTIALIAS)
    img.save(to_file_path, optimize=True)


def detect_file_locations(filename):
    to_file_path = to_base_dir + '/' + filename
    if os.path.exists(to_file_path):
        return None
    from_file_path = from_dir + '/' + filename
    return from_file_path, to_file_path


def run_threads(threads):
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()


def compresscsv(csvpath):
    examples = pd.read_csv(csvpath)
    charged_threads = []
    handled = {}
    for key, row in examples.iterrows():
        if row['filename'] in handled:
            continue
        handled[row['filename']] = True
        file_locations = detect_file_locations(row['filename'])
        if file_locations is not None:
            print(file_locations[1])
            t = threading.Thread(target=compress, args=file_locations)
            charged_threads.append(t)
            if len(charged_threads) == use_threads:
                run_threads(charged_threads)
                charged_threads.clear()


def main():
    compresscsv('./materials/optimized_freq_train.csv')
    compresscsv('./materials/optimized_freq_eval.csv')

main()