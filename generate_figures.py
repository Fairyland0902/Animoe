import argparse
import random
import os
import PIL.Image
from shutil import copyfile
from model import dcgan
import torch
import torchvision.utils as vutils

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', required=False, help='path to dataset')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG")
opt = parser.parse_args()

randomSeed = random.randint(1, 10000)
print("Random Seed: ", randomSeed)
random.seed(randomSeed)
torch.manual_seed(randomSeed)


def random_sample(sample_number):
    samples = []
    dir_num = len(os.listdir(opt.dataroot))
    for _ in range(0, sample_number):
        dir = str(random.randint(0, dir_num - 1))
        dir = dir.zfill(3)
        path = os.path.join(opt.dataroot, dir, "")
        files = [name for name in os.listdir(path)]
        index = random.randint(0, len(files) - 1)
        samples.append(os.path.join(path, files[index]))

    return samples


def draw_original_images(png, w, h, cols, rows, samples):
    canvas = PIL.Image.new('RGB', (w * cols, h * rows), 'white')
    for row in range(rows):
        for col in range(cols):
            index = row * cols + col
            image = PIL.Image.open(samples[index])
            canvas.paste(image, (w * col, h * row))
    canvas.save(png)


def draw_generated_images(png, cols, rows):
    nz = int(opt.nz)
    ngf = int(opt.ngf)
    nc = 3
    netG = dcgan.Generator(opt.ngpu, nz, nc, ngf)
    netG.load_state_dict(torch.load(opt.netG))
    latents = torch.randn(cols * rows, nz, 1, 1)
    images = netG(latents)
    vutils.save_image(images.detach(), png, normalize=True, nrow=rows, padding=0)


if __name__ == '__main__':
    samples = random_sample(2000)
    # sample_dir = "./samples/"
    # if not os.path.exists(sample_dir):
    #     os.mkdir(sample_dir)
    # for path in samples:
    #     filename = os.path.basename(path)
    #     copyfile(path, sample_dir + filename)
    draw_original_images("./original.png", 128, 128, 10, 10, random_sample(100))
    draw_generated_images("./generated.png", 10, 10)
