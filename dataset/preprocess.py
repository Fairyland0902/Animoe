import argparse
import os
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--directory", type=str)
args = parser.parse_args()


def preprocess():
    dir_list = os.listdir(args.directory)
    for dir in dir_list:
        path = os.path.join(args.directory, dir, "")
        print("Processing directory:", path)
        files = [name for name in os.listdir(path)]
        for file in files:
            file = os.path.join(path, file)
            result = subprocess.run(['magick', 'identify', '-format', '"%k"', file], stdout=subprocess.PIPE)
            value = result.stdout.decode('utf-8').replace('"', '')
            if int(value) < 257:
                print('x', file)
                os.remove(file)


if __name__ == '__main__':
    preprocess()
