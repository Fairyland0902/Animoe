import cv2
import argparse
import os
import glob

parser = argparse.ArgumentParser()
parser.add_argument("--tag_file", type=str)
parser.add_argument("--output_dir", type=str)
args = parser.parse_args()
if not os.path.exists(args.output_dir):
    os.mkdir(args.output_dir)


def detect(filename, output_dir, cascade_file="lbpcascade_animeface.xml"):
    if not os.path.isfile(cascade_file):
        raise RuntimeError("{}: not found".format(cascade_file))

    cascade = cv2.CascadeClassifier(cascade_file)
    image = cv2.imread(filename)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))

    for i, (x, y, w, h) in enumerate(faces):
        face = image[y:y + h, x:x + w, :]
        face = cv2.resize(face, (256, 256))
        save_filename = "{}-{}.jpg".format(os.path.basename(filename).split('.')[0], i)
        cv2.imwrite(output_dir + save_filename, face)


def main():
    with open(args.tag_file, encoding="utf-8") as fp:
        content = fp.readlines()
    tags = [x.strip() for x in content]

    for tag in tags:
        url = "https://danbooru.donmai.us/posts?tags=" + tag
        os.system('gallery-dl --range 1-1000 ' + url)
        download_dir = "./gallery-dl/danbooru/" + tag + "/*"
        output_dir = args.output_dir + tag + "/"
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        file_list = glob.glob(download_dir)
        for file in file_list:
            ext = os.path.splitext(file)[1]
            if ext == ".jpg" or ext == ".png":
                detect(file, output_dir)


if __name__ == '__main__':
    main()
