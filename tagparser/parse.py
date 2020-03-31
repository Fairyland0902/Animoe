import i2v
from PIL import Image
import json
import os

illust2vec = i2v.make_i2v_with_chainer("illust2vec_tag_ver200.caffemodel", "tag_list.json")

root = "sample2000/"
entries = os.listdir(root+"data/")
cnt = 0
for entry in entries:
	if entry[-4:]!=".png":
		continue
	cnt += 1
	fid = entry[:-4]
	img = Image.open(root+"data/"+entry)
	tags  = {}
	for tagn, tagp in illust2vec.estimate_plausible_tags([img], threshold=0.5)[0]["general"]:
		tags[tagn] = tagp
	with open('{}tags/{}.json'.format(root, fid), "w") as f:
		json.dump(tags, f)
	print("Parse Image: {}%".format(cnt/20))
	# break