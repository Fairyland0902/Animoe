import argparse
import random
import os
import PIL.Image
import model.utils as utils
from model.model import Generator
from model import utils
import torch
import torchvision.utils as vutils
from torch.autograd import Variable

# have GPU or not.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', default='./dataset/avatar', help='path to dataset')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default=None, help="path to netG")
opt = parser.parse_args()

randomSeed = random.randint(1, 10000)
print("Random Seed: ", randomSeed)
random.seed(randomSeed)
torch.manual_seed(randomSeed)

if opt.netG is not None:
    nz = opt.nz
    ngf = opt.ngf
    nc = 3
    # netG = dcgan.Generator(opt.ngpu, nz, nc, ngf)
    netG = Generator(nz, len(utils.hair) + len(utils.eyes)).to(device)
    netG.load_state_dict(torch.load(opt.netG, map_location=lambda storage, loc: storage))


def random_sample(sample_number):
    samples = []
    for _ in range(0, sample_number):
        path = os.path.join(opt.dataroot, "")
        files = [name for name in os.listdir(path)]
        index = random.randint(0, len(files) - 1)
        samples.append(os.path.join(path, files[index]))

    return samples


def draw_grid_images(png, w, h, cols, rows, samples):
    canvas = PIL.Image.new('RGB', (w * cols, h * rows), 'white')
    for row in range(rows):
        for col in range(cols):
            index = row * cols + col
            image = PIL.Image.open(samples[index])
            image = image.resize((w, h), PIL.Image.ANTIALIAS)
            canvas.paste(image, (w * col, h * row))
    canvas.save(png)


def draw_generated_images(png, cols, rows):
    # random tag
    # noises, tags = utils.fake_generator(cols * rows, opt.nz, device=device)
    # assigned tag
    noises, _ = utils.fake_generator(cols * rows, opt.nz, device=device)
    tag = ['pink hair', 'blue eyes']
    tag = utils.get_one_hot_tag(tag)
    tag = torch.FloatTensor(tag).view(1, -1).to(device)
    tags = torch.cat([tag for _ in range(cols * rows)], dim=0)
    print(tags)
    images = netG(noises, tags).detach()
    path = "./generate"
    try:
        os.makedirs(path)
    except OSError:
        pass
    for i, image in enumerate(images):
        vutils.save_image(utils.denorm(image), os.path.join(path, str(i) + ".png"))
    vutils.save_image(utils.denorm(images), png, nrow=cols, padding=0)


def draw_interpolation_images(png, cols, rows):
    z_1 = torch.rand(cols, nz, 1, 1)
    z_2 = torch.rand(cols, nz, 1, 1)
    dz = (z_2 - z_1) / rows
    latents = torch.FloatTensor(cols * rows, nz, 1, 1)
    for i in range(rows):
        latents[cols * i: cols * (i + 1), :, :, :] = z_1 + i * dz
    latents = Variable(latents)
    images = netG(latents)
    vutils.save_image(images, png, normalize=True, nrow=cols, padding=0)


if __name__ == '__main__':
    draw_grid_images("./original.png", 64, 64, 10, 5, random_sample(50))
    if opt.netG is not None:
        draw_generated_images("./generated.png", 10, 5)
    #     draw_interpolation_images("./interpolation.png", 10, 5)
