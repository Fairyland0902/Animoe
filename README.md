<div align="center">
    <img width="700px" src="misc/Animoe.png">
</div>

### Dataset

The original dataset can be downloaded from the [Anime Face Dataset](https://github.com/SteinsFu/Anime_face_dataset/tree/master/data). We then parse the tags using [Illustration2Vec](https://github.com/rezoo/illustration2vec) and delete those failure cases.

### Run the Code

To run the code, we need first run the file `preprocess.py` in the `model` folder:

```bash
python preprocess.py
```

It will dump all of the images and tags into one binary file, which can speedup our training process.

Then just run the code `train.py` with default parameters:

```bash
python train.py
```

The output images and checkpoint models are put in the folder `output` and we can visualize the loss with TensorBoard.


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

