import pathlib
import os
import sys
import os.path as osp
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd
import vaex as vx
import numpy as np


sys.path.insert(0, str(pathlib.Path(__file__).parent.resolve()))


class WikiArt(object):
    def __init__(self, root_dir):
        assert osp.exists(osp.join(root_dir, 'wikiart.csv'))
        self.root_dir = root_dir
        annotations = vx.from_csv(f'{self.root_dir}/wikiart.csv')
        acceptable_artists = list(set(annotations[annotations['split'] == 'database']['artist'].tolist()))
        temprepo = annotations[annotations['artist'].isin(acceptable_artists)]
        self.query_images = temprepo[temprepo['split'] == 'query']['name'].tolist()
        self.val_images = temprepo[temprepo['split'] == 'database']['name'].tolist()
        self.query_db = annotations[annotations['name'].isin(self.query_images)]
        self.val_db = annotations[annotations['name'].isin(self.val_images)]
        self.query_db['name'] = self.query_db['name'].apply(lambda x: '.'.join(x.split('.')[:-1]))
        self.val_db['name'] = self.val_db['name'].apply(lambda x: '.'.join(x.split('.')[:-1]))

    def get_query_col(self, col):
        return np.asarray(self.query_db[col].tolist())

    def get_val_col(self, col):
        return np.asarray(self.val_db[col].tolist())


class WikiArtD(Dataset):
    def __init__(self, root_dir, split, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.split = split
        assert osp.exists(osp.join(root_dir, 'wikiart.csv'))
        annotations = vx.from_csv(f'{self.root_dir}/wikiart.csv')
        acceptable_artists = list(set(annotations[annotations['split'] == 'database']['artist'].tolist()))
        temprepo = annotations[annotations['artist'].isin(acceptable_artists)]
        self.pathlist = temprepo[temprepo['split'] == split]['path'].tolist()

        self.namelist = list(map(lambda x: x.split('/')[-1], self.pathlist))

    def __len__(self):
        return len(self.namelist)

    def __getitem__(self, idx):
        img_loc = self.pathlist[idx]  # os.path.join(self.root_dir, self.split,self.artists[idx] ,self.pathlist[idx])
        image = Image.open(img_loc).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, idx


class WikiArtTrain(Dataset):
    def __init__(self, root_dir, split='database', transform=None, maxsize=None):
        self.root_dir = root_dir
        self.transform = transform
        self.split = split
        assert os.path.exists(os.path.join(root_dir, 'wikiart.csv'))
        annotations = pd.read_csv(f'{self.root_dir}/wikiart.csv')
        acceptable_artists = list(
            set(annotations[annotations['split'] == 'database']['artist'].tolist())
        )
        temprepo = annotations[annotations['artist'].isin(acceptable_artists)]
        self.pathlist = temprepo[temprepo['split'] == split]['path'].tolist()
        self.labels = temprepo[temprepo['split'] == split]['artist'].tolist()

        self.artist_to_index = {artist: i for i, artist in enumerate(acceptable_artists)}
        self.index_to_artist = acceptable_artists

        # Convert labels to one-hot
        self.labels = list(map(lambda x: self.artist_to_index[x], self.labels))
        self.labels = np.eye(len(acceptable_artists))[self.labels].astype(bool)
        self.namelist = list(map(lambda x: x.split('/')[-1], self.pathlist))

        # Select maxsize number of images
        if maxsize is not None:
            ind = np.random.randint(0, len(self.namelist), maxsize)
            self.namelist = [self.namelist[i] for i in ind]
            self.pathlist = [self.pathlist[i] for i in ind]
            self.labels = self.labels[ind]

    def __len__(self):
        return len(self.namelist)

    def __getitem__(self, idx):

        img_loc = self.pathlist[idx]
        image = Image.open(img_loc).convert("RGB")

        if self.transform:
            images = self.transform(image)

        artist = self.labels[idx]
        return images, artist, idx
