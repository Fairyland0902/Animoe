import pickle
import json
import argparse
import os
import utils
import cv2

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="../dataset")
parser.add_argument("--output", type=str, default="../dataset/avatar_with_tag.dat")
opt = parser.parse_args()
dataset = opt.dataset
output = opt.output


def get_avatar_with_tag(dataset, file):
    tags = []
    tag_file = os.path.join(dataset, 'tag', '{}.json'.format(file))
    img_file = os.path.join(dataset, 'avatar', '{}.png'.format(file))
    with open(tag_file, 'r') as fin:
        for k in json.load(fin).keys():
            if k in utils.tag_map.keys():
                tags.append(k)
    image = cv2.imread(img_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # print(tags)
    # print(utils.get_one_hot_tag(tags))
    return [utils.get_one_hot_tag(tags), image]


def dump_file(obj, dump_filename):
    with open(dump_filename, 'wb') as fout:
        pickle.dump(obj, fout)


if __name__ == '__main__':
    result_list = []
    path = os.path.join(dataset, "tag", "")
    files = [name for name in os.listdir(path)]
    for file in files:
        if os.path.splitext(file)[1] != '.json':
            continue
        result_list.append(get_avatar_with_tag(dataset, os.path.splitext(file)[0]))

    dump_file(result_list, output)
