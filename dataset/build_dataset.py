import cv2
import argparse
import os
import glob
import sqlite3

parser = argparse.ArgumentParser()
parser.add_argument("--tag_file", type=str)
parser.add_argument("--output_dir", type=str)
args = parser.parse_args()
if not os.path.exists(args.output_dir):
    os.mkdir(args.output_dir)


def detect(filename, output_dir, scale_factor, cascade_file="lbpcascade_animeface.xml"):
    if not os.path.isfile(cascade_file):
        raise RuntimeError("{}: not found".format(cascade_file))

    cascade = cv2.CascadeClassifier(cascade_file)
    image = cv2.imread(filename)
    image_h = image.shape[0]
    image_w = image.shape[1]
    scale_factor = (scale_factor - 1.0) / 2.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))

    for i, (x, y, w, h) in enumerate(faces):
        new_x, new_y = max(int(x - scale_factor * w), 0), max(int(y - scale_factor * h), 0)
        new_w = int(w + 2.0 * scale_factor * w) if image_w > (new_x + w + 2.0 * scale_factor * w) else image_w - new_x
        new_h = int(h + 2.0 * scale_factor * h) if image_h > (new_y + h + 2.0 * scale_factor * h) else image_h - new_y
        face = image[new_y:new_y + new_h, new_x:new_x + new_w, :]
        face = cv2.resize(face, (256, 256))
        save_filename = "{}-{}.jpg".format(os.path.basename(filename).split('.')[0], i)
        cv2.imwrite(output_dir + save_filename, face)


def main():
    with open(args.tag_file, encoding="utf-8") as fp:
        content = fp.readlines()
    tags = [x.strip() for x in content]

    conn = sqlite3.connect("filename.db")
    c = conn.cursor()
    cursor = c.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    table_list = [item[0] for item in tables]
    if "images" in table_list:
        c.execute("DROP TABLE images")
    c.execute("CREATE TABLE images (name VARCHAR(200) PRIMARY KEY NOT NULL)")
    print("Create database successfully")

    for tag in tags:
        url = "https://danbooru.donmai.us/posts?tags=" + tag
        os.system('gallery-dl --range 1-1000 ' + url)
        if ':' in tag:
            tag = tag.replace(':', '_')
        download_dir = "./gallery-dl/danbooru/" + tag + "/*"
        output_dir = args.output_dir + tag + "/"
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        file_list = glob.glob(download_dir)
        for file in file_list:
            name = os.path.basename(file)
            # Search whether the image has already been downloaded.
            cursor = c.execute("SELECT * FROM images WHERE name=?", (name,))
            value = cursor.fetchall()
            if len(value) == 0:
                # This is a new image, insert its name into database and process it.
                c.execute("INSERT INTO images (name) VALUES (?)", (name,))
                ext = os.path.splitext(file)[1]
                if ext == ".jpg" or ext == ".png":
                    detect(file, output_dir=output_dir, scale_factor=1.5)

    conn.commit()
    conn.close()


if __name__ == '__main__':
    main()
