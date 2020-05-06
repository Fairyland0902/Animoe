import argparse
import os
import random
import torch
import torch.utils.data
import torch.nn as nn
from torch.autograd import Variable, grad
from torchvision.transforms import transforms
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from model.model import Generator, Discriminator
from model.data_loader import AnimeDataset
import model.utils as utils

# have GPU or not.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--avatar_tag_dat_path', type=str, default='./dataset/avatar_with_tag.dat', help='avatar with tag\'s binary file path')
parser.add_argument('--learning_rate_g', type=float, default=0.0002, help='learning rate for the generator')
parser.add_argument('--learning_rate_d', type=float, default=0.0002, help='learning rate for the discriminator')
parser.add_argument('--beta_1', type=float, default=0.5, help='adam optimizer\'s paramenter')
parser.add_argument('--batch_size', type=int, default=128, help='training batch size for each epoch')
parser.add_argument('--max_epoch', type=int, default=200, help='training epoch')
parser.add_argument('--num_workers', type=int, default=3, help='number of data loader processors')
parser.add_argument('--noise_size', type=int, default=100, help='number of G\'s input')
parser.add_argument('--lambda_cls', type=float, default=1.0, help='cls\'s lambda')
parser.add_argument('--lambda_gp', type=float, default=0.5, help='gp\'s lambda')
parser.add_argument('--interval', type=int, default=100, help='intervals for saving intermediate files')
parser.add_argument('--output_path', type=str, default='./output/', help='folder to output images and model checkpoints')
parser.add_argument('--log_dir', type=str, default='./runs/', help='path to logs')

# Load parameters
opt = parser.parse_args()
avatar_tag_dat_path = opt.avatar_tag_dat_path
learning_rate_g = opt.learning_rate_g
learning_rate_d = opt.learning_rate_d
beta_1 = opt.beta_1
batch_size = opt.batch_size
max_epoch = opt.max_epoch
num_workers = opt.num_workers
noise_size = opt.noise_size
lambda_cls = opt.lambda_cls
lambda_gp = opt.lambda_gp
interval = opt.interval
output_path = opt.output_path
model_path = os.path.join(output_path, "model", "")
image_path = os.path.join(output_path, "image", "")
log_dir = opt.log_dir
fix_noise, fix_tag = utils.fake_generator(batch_size, noise_size, device)


def main(writer):
    dataset = AnimeDataset(avatar_tag_dat_path, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    G = Generator(noise_size, len(utils.hair) + len(utils.eyes)).to(device)
    D = Discriminator(len(utils.hair), len(utils.eyes)).to(device)
    G_optim = torch.optim.Adam(G.parameters(), lr=learning_rate_g, betas=(beta_1, 0.999))
    D_optim = torch.optim.Adam(D.parameters(), lr=learning_rate_d, betas=(beta_1, 0.999))
    criterion = nn.BCELoss()

    # training
    iteration = 0
    real_label = torch.ones(batch_size).to(device)
    # real_label = torch.Tensor(batch_size).uniform_(0.9, 1).to(device)  # soft labeling
    fake_label = torch.zeros(batch_size).to(device)
    for epoch in range(max_epoch + 1):
        for i, (real_tag, real_img) in enumerate(data_loader):
            real_img = real_img.to(device)
            real_tag = real_tag.to(device)

            # train D with real images
            D.zero_grad()
            real_score, real_predict = D(real_img)
            real_discrim_loss = criterion(real_score, real_label)
            real_classifier_loss = criterion(real_predict, real_tag)

            # train D with fake images
            z, fake_tag = utils.fake_generator(batch_size, noise_size, device)
            fake_img = G(z, fake_tag).to(device)
            fake_score, fake_predict = D(fake_img)
            fake_discrim_loss = criterion(fake_score, fake_label)

            discrim_loss = (real_discrim_loss + fake_discrim_loss) * 0.5
            classifier_loss = real_classifier_loss * lambda_cls

            # gradient penalty
            alpha_size = [1] * real_img.dim()
            alpha_size[0] = real_img.size(0)
            alpha = torch.rand(alpha_size).to(device)
            x_hat = Variable(alpha * real_img.data + (1 - alpha) * (real_img.data + 0.5 * real_img.data.std() * torch.rand(real_img.size()).to(device)), requires_grad=True).to(device)
            fake_score, fake_tag = D(x_hat)
            gradients = grad(outputs=fake_score, inputs=x_hat, grad_outputs=torch.ones(fake_score.size()).to(device), create_graph=True, retain_graph=True, only_inputs=True)[0].view(x_hat.size(0), -1)
            gradient_penalty = lambda_gp * ((gradients.norm(2, dim=1) - 1) ** 2).mean()

            D_loss = discrim_loss + classifier_loss + gradient_penalty
            D_loss.backward()
            D_optim.step()

            # train G
            G.zero_grad()
            z, fake_tag = utils.fake_generator(batch_size, noise_size, device)
            fake_img = G(z, fake_tag).to(device)
            fake_score, fake_predict = D(fake_img)

            discrim_loss = criterion(fake_score, real_label)
            classifier_loss = criterion(fake_predict, fake_tag) * lambda_cls

            G_loss = discrim_loss + classifier_loss
            G_loss.backward()
            G_optim.step()

            # plot loss curve
            writer.add_scalar('Loss_D', D_loss.item(), iteration)
            writer.add_scalar('Loss_G', G_loss.item(), iteration)
            print('[{}/{}][{}/{}] Iteration: {}'.format(epoch, max_epoch, i, len(data_loader), iteration))

            if iteration % interval == interval - 1:
                fake_img = G(fix_noise, fix_tag)
                vutils.save_image(utils.denorm(fake_img[:64, :, :, :]), os.path.join(image_path, 'fake_image_{}.png'.format(iteration)), padding=0)
                vutils.save_image(utils.denorm(real_img[:64, :, :, :]), os.path.join(image_path, 'real_image_{}.png'.format(iteration)), padding=0)
                grid = vutils.make_grid(utils.denorm(fake_img[:64, :, :, :]), padding=0)
                writer.add_image('generation results', grid, iteration)

            iteration += 1
        # checkpoint
        torch.save(G.state_dict(), os.path.join(model_path, 'netG_epoch_{}.pth'.format(epoch)))
        torch.save(D.state_dict(), os.path.join(model_path, 'netD_epoch_{}.pth'.format(epoch)))


if __name__ == '__main__':
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    if not os.path.exists(image_path):
        os.makedirs(image_path)
    writer = SummaryWriter(log_dir + datetime.now().strftime("%Y%m%d-%H%M%S"))
    main(writer)
