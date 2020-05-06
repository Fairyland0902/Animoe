import i2v
from PIL import Image
import json
import os
import argparse
import itertools

parser = argparse.ArgumentParser()
parser.add_argument("--directory", type=str, default="../dataset/avatar/")
parser.add_argument("--tags_output", type=str, default="../dataset/tag/")
opt = parser.parse_args()
directory = opt.directory
tags_output = opt.tags_output

illust2vec = i2v.make_i2v_with_chainer("illust2vec_tag_ver200.caffemodel", "tag_list.json")

hair = ['blonde hair', 'brown hair', 'black hair', 'blue hair', 'pink hair',
        'purple hair', 'green hair', 'red hair', 'silver hair', 'white hair', 'orange hair',
        'aqua hair', 'gray hair']
eyes = ['blue eyes', 'red eyes', 'brown eyes',
        'green eyes', 'purple eyes', 'yellow eyes', 'pink eyes', 'aqua eyes', 'black eyes',
        'orange eyes']


def parse():
    path = os.path.join(directory, "")
    print("Processing directory:", path)
    files = [name for name in os.listdir(path)]
    for file in files:
        if os.path.splitext(file)[1] != ".png":
            continue
        img = Image.open(os.path.join(path, file))
        hair_tags = {}
        eyes_tags = {}
        for tagn, tagp in illust2vec.estimate_plausible_tags([img], threshold=0.25)[0]["general"]:
            if tagn in hair:
                hair_tags[tagn] = tagp
            if tagn in eyes:
                eyes_tags[tagn] = tagp
        hair_tags = {k: v for k, v in sorted(hair_tags.items(), key=lambda item: item[1], reverse=True)}
        if len(hair_tags) > 1:
            # only keep the label with the most probability
            hair_tags = dict(itertools.islice(hair_tags.items(), 1))
        eyes_tags = {k: v for k, v in sorted(eyes_tags.items(), key=lambda item: item[1], reverse=True)}
        if len(eyes_tags) > 1:
            # only keep the label with the most probability
            eyes_tags = dict(itertools.islice(eyes_tags.items(), 1))
        hair_tags.update(eyes_tags)
        print(hair_tags)
        if len(hair_tags) < 2:
            continue
        json_file = os.path.splitext(file)[0] + ".json"
        with open(os.path.join(tags_output, json_file), "w") as f:
            json.dump(hair_tags, f)
        print("Parse Image: {}".format(file))


if __name__ == '__main__':
    if not os.path.exists(tags_output):
        os.mkdir(tags_output)
    parse()
