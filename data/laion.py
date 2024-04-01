import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm
import pickle
import vaex as vx


def create_laion_cache(root_dir, anno_dir, keys=['artist', 'medium', 'movement']):
    # -projects/diffusion_rep/data/laion_style_subset
    # read all the picke files in the anno_dir
    paths = []
    labels = []  # list of lists since each image can have multiple labels
    labels_to_index = {}  # dictionary that maps each label to an list of image indices

    keys_offset = {k: 1000000 * i for i, k in enumerate(keys)}  # offset each key labels by a large number

    str_to_list = lambda x, offset: [offset + int(a) for a in x.strip().split(',') if len(a) > 0]
    for f in tqdm(os.listdir(anno_dir)):
        if f.endswith('.pkl'):
            with open(os.path.join(anno_dir, f), 'rb') as tmp:
                ann = pickle.load(tmp)

                for i, path in enumerate(ann['key']):
                    cur_label = {
                        k: str_to_list(ann[k][i], keys_offset[k])
                        for k in keys
                    }
                    cur_label = sum(cur_label.values(), [])
                    if len(cur_label) > 0:
                        image_path = os.path.join(root_dir, 'data', path[:5], path + '.jpg')

                        # if not os.path.exists(image_path):
                        #     continue

                        paths.append(image_path)
                        labels.append(set(cur_label))
                        for l in cur_label: labels_to_index.setdefault(l, []).append(i)

    cache_path = os.path.join(anno_dir, '_'.join(keys) + '.cache')
    with open(cache_path, 'wb') as tmp:
        pickle.dump((paths, labels, labels_to_index), tmp)
    return paths, labels, labels_to_index


class LAION(Dataset):
    def __init__(self, root_dir, anno_dir, split='database', transform=None,
                 keys=['artist', 'medium', 'movement'],
                 min_images_per_label=1, max_images_per_label=100000,
                 num_queries_per_label=10, maxsize=None, model_type='dann'):
        # -projects/diffusion_rep/data/laion_style_subset
        self.root_dir = root_dir
        self.transform = transform
        self.model_type = model_type

        # read all the picke files in the anno_dir
        paths = []
        labels = []  # list of lists since each image can have multiple labels
        labels_to_index = {}  # dictionary that maps each label to an list of image indices

        cache_path = os.path.join(anno_dir, '_'.join(keys) + '.cache')
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as tmp:
                paths, labels, labels_to_index = pickle.load(tmp)
        else:
            paths, labels, labels_to_index = create_laion_cache(root_dir, anno_dir, keys)

        maxout_labels = [l for l, v in labels_to_index.items() if len(v) > max_images_per_label]
        maxout_labels.append('')  # Artificially add an empty label
        print(f"Removing {len(maxout_labels)} tags with > {max_images_per_label} images")

        minout_labels = [l for l, v in labels_to_index.items() if len(v) < min_images_per_label]
        print(f"Removing {len(minout_labels)} tags with < {min_images_per_label} images")

        # Get all possible tags
        self.index_to_labels = list(set(labels_to_index.keys()) - set(maxout_labels) - set(minout_labels))
        self.labels_to_index = {l: i for i, l in enumerate(self.index_to_labels)}

        self.pathlist = []
        self.labels = []
        eye = np.eye(len(self.index_to_labels))
        print("Filtering out labels")
        for path, label in tqdm(zip(paths, labels)):
            for l in maxout_labels:
                if l in label:
                    label.remove(l)

            for l in minout_labels:
                if l in label:
                    label.remove(l)

            if len(label) > 0:
                self.pathlist.append(path)
                cur_label = np.sum(eye[[self.labels_to_index[l] for l in label]], axis=0).astype(bool)
                self.labels.append(cur_label)
        self.labels = np.array(self.labels)

        ## Split the dataset into index and query
        keys_offset = {k: 1000000 * i for i, k in enumerate(keys)}
        self.name_to_label = {}
        for k in keys:
            key_labels_path = os.path.join(
                anno_dir, '../clip-interrogator/clip_interrogator/data',
                k + "s_filtered_new.txt")
            with open(os.path.join(key_labels_path)) as f:
                for i, l in enumerate(f.readlines()):
                    self.name_to_label[l.strip().replace("@", " ")] = keys_offset[k] + i

        with open(os.path.join(anno_dir, 'top612_artists_shortlist_400.txt'), 'r') as f:
            q_names = [l.lower().strip() for l in f.readlines()]
            q_labels = [self.name_to_label[n] for n in q_names]
            q_index = [self.labels_to_index[l] for l in q_labels]

        query_ind = np.unique(np.concatenate(
            [np.where(self.labels[:, i])[0][:num_queries_per_label]
             for i in q_index]))

        if split == "database":
            self.pathlist = [self.pathlist[i] for i in range(len(self.pathlist)) if i not in query_ind]
            self.labels = np.delete(self.labels, query_ind, axis=0)
        else:
            self.pathlist = [self.pathlist[i] for i in query_ind]
            self.labels = self.labels[query_ind]

        self.namelist = list(map(lambda x: x.split('/')[-1], self.pathlist))
        # Select maxsize number of images
        if maxsize is not None:
            ind = np.random.randint(0, len(self.pathlist), maxsize)
            self.pathlist = [self.pathlist[i] for i in ind]
            self.labels = self.labels[ind]
            self.namelist = [self.namelist[i] for i in ind]

    def __len__(self):
        return len(self.pathlist)

    def __getitem__(self, idx):
        img_loc = self.pathlist[idx]
        image = Image.open(img_loc).convert("RGB")

        if self.transform:
            images = self.transform(image)

        style = self.labels[idx]
        if self.model_type == 'dann':
            return images, style, idx
        else:
            return images, idx


def create_laion_dedup_cache(dedup_dir):
    # -projects/diffusion_rep/data/laion_style_subset/dedup_info
    keys = None
    labels = None
    rejects = None
    matching_info = None

    files = [f for f in os.listdir(dedup_dir) if f.endswith('.parquet')]
    for f in tqdm(sorted(files, key=lambda x: int(x.split('_')[2]))):
        # Load dedup info
        df = vx.open(os.path.join(dedup_dir, f))
        if keys is None:
            keys = df['name'].tolist()

        # Updating reject information
        cur_reject = df['matched'].to_numpy()
        if rejects is not None:
            rejects += cur_reject
        else:
            rejects = cur_reject

        # Load labels
        cur_labels = np.load(os.path.join(dedup_dir, f.replace('parquet', 'npz').replace('val_db', 'multilabel')))
        cur_labels = cur_labels["arr_0"]
        if labels is not None:
            labels += cur_labels
        else:
            labels = cur_labels

        # Load matching info
        cur_matching_info = pickle.load(
            open(os.path.join(dedup_dir, f.replace('parquet', 'pkl').replace('val_db', 'matching_info')), 'rb'))
        if matching_info is not None:
            matching_info.extend(cur_matching_info)
        else:
            matching_info = cur_matching_info

    # Propagating labels
    for i in tqdm(range(len(matching_info) - 1, -1, -1)):
        labels[i] += np.sum(labels[matching_info[i], :], axis=0, dtype=bool)

    cache_path = os.path.join(dedup_dir, 'joined.cache')
    with open(cache_path, 'wb') as tmp:
        pickle.dump((keys, labels, rejects), tmp)
    return keys, labels, rejects


class LAIONDedup(Dataset):
    def __init__(self, root_dir, anno_dir, transform=None, model_type='dann', eval_mode=False, artist_mode=False):
        self.root_dir = root_dir
        self.transform = transform
        self.model_type = model_type

        dedup_dir = os.path.join(anno_dir, 'dedup_info')
        cache_path = os.path.join(dedup_dir, 'joined.cache')
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as tmp:
                keys, labels, rejects = pickle.load(tmp)
        else:
            keys, labels, rejects = create_laion_dedup_cache(dedup_dir)

        keys = np.array(keys)[~rejects]
        self.pathlist = [os.path.join(root_dir, 'data', key[:5], key + '.jpg') for key in keys]
        self.labels = labels[~rejects]
        self.namelist = list(map(lambda x: x.split('/')[-1], self.pathlist))

        if eval_mode:
            q_dset = LAION(root_dir, anno_dir, split='query')
            self.query_db = vx.from_arrays(
                name=[x.split('.')[0] for x in q_dset.namelist],
                multilabel=q_dset.labels)

            self.name_to_label = q_dset.name_to_label
            self.labels_to_index = q_dset.labels_to_index
            self.index_to_labels = q_dset.index_to_labels

            self.val_db = vx.from_arrays(
                name=keys.tolist(),
                multilabel=self.labels)

            if artist_mode:
                # Filtering the db to include images which have hit on an artist
                artist_inds = []
                for label, index in self.labels_to_index.items():
                    if label < 1000000:
                        artist_inds.append(index)
                artist_labels = self.labels[:, artist_inds]
                artist_images = np.argwhere(np.sum(artist_labels, axis=1) > 0)
                self.val_db = self.val_db.take(artist_images.squeeze()).extract()

    def __len__(self):
        return len(self.pathlist)

    def __getitem__(self, idx):
        img_loc = self.pathlist[idx]
        image = Image.open(img_loc).convert("RGB")

        if self.transform:
            images = self.transform(image)

        style = self.labels[idx]
        if self.model_type == 'dann':
            return images, style, idx
        else:
            return images, idx

    def get_query_col(self, col):
        return np.asarray(self.query_db[col].tolist())

    def get_val_col(self, col):
        return np.asarray(self.val_db[col].tolist())


class SDSynth400:
    def __init__(self, root_dir, query_split='user_caps', transform=None, eval_mode=False):
        self.root_dir = root_dir
        self.transform = transform
        self.query_split = query_split
        assert query_split in ['user_caps', 'simple_caps', 'woman_caps', 'house_caps', 'dog_caps']
        assert os.path.exists(os.path.join(root_dir, f'{query_split}.csv'))
        annotations = vx.from_csv(f'{self.root_dir}/{query_split}.csv')

        self.pathlist = annotations['filepath'].tolist()
        self.namelist = list(map(lambda x: x.split('/')[-1], self.pathlist))

        # Dummy variables, not actually needed
        self.query_images = []
        self.val_images = []

        if eval_mode:
            data_dir = '-datasets/improved_aesthetics_6plus'
            anno_dir = '-projects/diffusion_rep/data/laion_style_subset'
            val_dset = LAIONDedup(data_dir, anno_dir, transform=transform, eval_mode=True, artist_mode=True)
            # val_dset = LAION(data_dir, anno_dir, transform=transform)
            # Needed for search code
            filenames = [f.split('.')[0] for f in self.namelist]
            q_names = [[l.lower().strip() for l in eval(label)] for label in annotations['labels'].tolist()]
            q_labels = [[val_dset.name_to_label[n] for n in names if n in val_dset.name_to_label] for names in q_names]
            q_index = [[val_dset.labels_to_index[l] for l in labels if l in val_dset.labels_to_index] for labels in
                       q_labels]

            eye = np.eye(len(val_dset.index_to_labels))
            q_binlabels = [np.sum(eye[ind], axis=0).astype(bool) for ind in q_index]
            self.query_db = vx.from_arrays(
                name=filenames, multilabel=q_binlabels)
            self.val_db = val_dset.val_db

    def __len__(self):
        return len(self.namelist)

    def __getitem__(self, idx):
        img_loc = self.pathlist[idx]
        image = Image.open(img_loc).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, idx

    def get_query_col(self, col):
        return np.asarray(self.query_db[col].tolist())

    def get_val_col(self, col):
        return np.asarray(self.val_db[col].tolist())


if __name__ == "__main__":
    # dset = WikiArt(
    #     "-projects/diffusion_rep/data/wikiart", 'database')

    dset = LAION(
        "-datasets/improved_aesthetics_6plus",
        "-projects/diffusion_rep/data/laion_style_subset",
        split='database')
    print(f"{len(dset)} images in the dataset")

    index_to_labels = []
    index_to_keys = []
    index_to_texts = []
    label_to_name = {v: k for k, v in dset.name_to_label.items()}
    for label in dset.index_to_labels:
        index_to_texts.append(label_to_name[label])
        index_to_labels.append(label)
        if label < 1000000:
            index_to_keys.append('artist')
        elif label < 2000000:
            index_to_keys.append('medium')
        else:
            index_to_keys.append('movement')

    path = "-projects/diffusion_rep/data/laion_style_subset/index_to_labels_keys_texts.pkl"
    with open(path, 'wb') as tmp:
        pickle.dump((index_to_labels, index_to_keys, index_to_texts), tmp)

    # dset = LAION(
    #     "-datasets/improved_aesthetics_6plus",
    #     "-projects/diffusion_rep/data/laion_style_subset",
    #     split='query',
    #     min_images_per_label=10,
    #     max_images_per_label=100000)

    # print(f"{len(dset)} images in the dataset")

    # dset = LAIONDedup(
    #     "-datasets/improved_aesthetics_6plus",
    #     "-projects/diffusion_rep/data/laion_style_subset",
    #     eval_mode=True)
