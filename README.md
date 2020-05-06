<div align="center">
    <img width="700px" src="https://github.com/Fairyland0902/Animoe/raw/master/misc/Animoe.png">
</div>

### Dataset

The original dataset can be downloaded from the [Anime Face Dataset](https://github.com/SteinsFu/Anime_face_dataset/tree/master/data). We then parse the tags using [Illustration2Vec](https://github.com/rezoo/illustration2vec) and delete those failure cases.


### Web Application 

#### Demo
[LINK](https://drive.google.com/file/d/1NXwHeaoiNIMDaAmbylT8oezc6Px14P1F/view?usp=sharing)

#### Environment
> Python virtual environment is recommended!

```bash
pip install Flask
```

#### Run

```bash
python3 webapp.py {your .pth model file path}
```

Example:
```bash
python3 webapp.py output/netG_epoch_200.pth
```

#### Web App Preview
> open http://127.0.0.1:5000/

![webapp](misc/webapp.png)

#### Notes

1. The anime eye icon in the logo is adapted from [Wikipedia](https://en.wikipedia.org/wiki/File:Anime_eye.svg).
