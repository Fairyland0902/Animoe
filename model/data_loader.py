import torch
from torch.utils.data import Dataset
import pickle


class AnimeDataset(Dataset):
    def __init__(self, pickle_path, transform=None, target_transform=None):
        self.tags = []
        self.images = []
        for i in pickle.load(open(pickle_path, 'rb')):
            self.tags.append(torch.tensor(i[0], dtype=torch.float32))
            self.images.append(i[1])
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        tag = self.tags[index]
        image = self.images[index]
        if self.transform is not None:
            image = self.transform(image)
        return tag, image

    def __len__(self):
        return len(self.tags)
