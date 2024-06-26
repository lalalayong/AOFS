#!/usr/bin/python
# encoding: utf-8
import itertools
import os
import random
import time

import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

from image import *
from cfg import cfg
from util import is_dict
import cv2


def topath(p):
    return p.replace('scratch', 'tmp_scratch/basilisk')


def loadlines(root, checkvalid=True):
    if is_dict(root):
        lines = []
        with open(root, 'r') as f:
            # files = [line.rstrip().split()[-1] for line in f.readlines()]
            files = [line.rstrip().split() for line in f.readlines()]
            if checkvalid:
                files = [topath(line[-1]) for line in files if line[0] in cfg.base_classes]
            else:
                files = [topath(line[-1]) for line in files if line[0] in cfg.classes]
        for file in files:
            with open(file, 'r') as f:
                lines.extend(f.readlines())
        lines = sorted(list(set(lines)))
    else:
        with open(root, 'r') as file:
            lines = file.readlines()

    lines_ = []
    # for line in lines:
    #     lines_.append(line.replace('\n', ''))
    lines_ = [item.rstrip() for item in lines]
    return lines_


def is_valid(imgpath, withnovel=True):
    labpath = listDataset.get_labpath(imgpath.rstrip())
    if os.path.getsize(labpath):
        bs = np.loadtxt(labpath)
        if bs is not None:
            bs = np.reshape(bs, (-1, 5))
            clsset = set(bs[:,0].astype(np.int).tolist())
            if withnovel:
                # Check whether an image contains base objects
                if not clsset.isdisjoint(set(cfg.base_ids)):
                    return True
            else:
                # Check whether an image contains base objects only
                if clsset.isdisjoint(set(cfg.novel_ids)):
                    return True

    return False


def build_dataset(dataopt):
    # Base training dataset
    if not cfg.tuning:
        return loadlines(dataopt['train'], checkvalid=False)

    # Meta tuning dataset
    if cfg.repeat == 1:
        return loadlines(dataopt['meta'])
    else:
        if 'dynamic' not in dataopt or int(dataopt['dynamic']) == 0:
            return loadlines(dataopt['meta']) * cfg.repeat
        else:
            metalist, metacnt = load_metadict(dataopt['meta'], cfg.repeat)
            return build_fewset(dataopt['train'], metalist, metacnt, cfg.shot*cfg.repeat)


def load_metadict(metapath, repeat=1):
    with open(metapath, 'r') as f:
        files = []
        for line in f.readlines():
            pair = line.rstrip().split()
            if len(pair) == 2:
                pass
            elif len(pair) == 4:
                pair = [pair[0]+' '+pair[1], pair[2]+' '+pair[3]]
            else:
                raise NotImplementedError('{} not recognized'.format(pair))
            files.append(pair)
        # files = [line.rstrip().split() for line in f.readlines()]

        metadict = {line[0]: loadlines(line[1]) for line in files}

    pdb.set_trace()
    # Remove base-class images
    for k in metadict.keys():
        if k not in cfg.novel_classes:
            metadict[k] = []
    metalist = set(sum(metadict.values(), []))

    # Count bboxes
    metacnt = {c:0 for c in metadict.keys()}
    for imgpath in metalist:
        labpath = listDataset.get_labpath(imgpath.strip())
        # Load converted annotations
        bs = np.loadtxt(labpath)
        bs = np.reshape(bs, (-1, 5))
        bcls = bs[:,0].astype(np.int).tolist()
        for ci in set(bcls):
            metacnt[cfg.classes[ci]] += bcls.count(ci)

    for c in metacnt.keys():
        metacnt[c] *= repeat

    metalist =  list(metalist) * repeat
    return metalist, metacnt


def build_fewset(imglist, metalist, metacnt, shot, replace=True):
    # Random sample bboxes for base classes
    if isinstance(imglist, str):
        with open(imglist) as f:
            names = f.readlines()
    elif isinstance(imglist, list):
        names = imglist.copy()
    else:
        raise NotImplementedError('imglist type not recognized')

    while min(metacnt.values()) < shot:
        imgpath = random.sample(names, 1)[0]
        labpath = listDataset.get_labpath(imgpath.strip())
        # Remove empty annotation
        if not os.path.getsize(labpath):
            names.remove(imgpath)
            continue

        # Load converted annotations
        bs = np.loadtxt(labpath)
        bs = np.reshape(bs, (-1, 5))
        bcls = bs[:,0].astype(np.int).tolist()

        if bs.shape[0] > 3:
            continue

        # Remove images contatining novel objects
        if not set(bcls).isdisjoint(set(cfg.novel_ids)):
            names.remove(imgpath)
            continue

        # Check total number of bbox per class so far
        overflow = False
        for ci in set(bcls):
            if metacnt[cfg.classes[ci]] + bcls.count(ci) > shot:
                overflow = True
                break
        if overflow:
            names.remove(imgpath)
            continue

        # Add current imagepath to the file lists
        for ci in set(bcls):
            metacnt[cfg.classes[ci]] += bcls.count(ci)
        metalist.append(imgpath)

        # To avoid duplication
        if not replace:
            names.remove(imgpath)

    random.shuffle(metalist)
    return metalist

class listDataset(Dataset):

    def __init__(self, root,
            shape=None,
            shuffle=True,
            transform=None,
            target_transform=None,
            train=False, seen=0,
            batch_size=64,
            num_workers=4):
        self.train = train

        if isinstance(root, list):
            self.lines = root
        elif is_dict(root):
            lines = []
            with open(root, 'r') as f:
                files = [line.rstrip().split()[-1] for line in f.readlines()]
            for file in files:
                with open(file, 'r') as f:
                    lines.extend(f.readlines())
            self.lines = sorted(list(set(lines)))
        else:
            with open(root, 'r') as file:
                self.lines = [topath(l) for l in file.readlines()]

        # Filter out images not in base classes
        print("===> Number of samples (before filtring): %d" % len(self.lines))
        if self.train and not isinstance(root, list):
            self.lines = [l for l in self.lines if self.is_valid(l)]
        print("===> Number of samples (after filtring): %d" % len(self.lines))

        if shuffle:
            random.shuffle(self.lines)

        self.nSamples = len(self.lines)
        self.transform = transform
        self.target_transform = target_transform
        self.shape = shape
        self.seen = seen
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.init_width = shape[0]

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        imgpath = self.lines[index].rstrip()

        seen_step = 4000 * 16
        n = self.init_width // 32
        if self.train and index % self.batch_size == 0 and cfg.multiscale:
            if self.seen < seen_step:
                width = n * 32
                self.shape = (width, width)
            elif self.seen < 2 * seen_step:
                width = (random.randint(-1, 1) + n) * 32
                self.shape = (width, width)
            elif self.seen < 3 * seen_step:
                width = (random.randint(-2, 2) + n) * 32
                self.shape = (width, width)
            elif self.seen < 4 * seen_step:
                width = (random.randint(-3, 3) + n) * 32
                self.shape = (width, width)
            else:
                width = (random.randint(-4, 4) + n) * 32
                self.shape = (width, width)

        jitter = 0.2
        hue = 0.1
        saturation = 1.5
        exposure = 1.5

        labpath = listDataset.get_labpath(imgpath)
        img, label = load_data_detection(imgpath, labpath, self.shape, jitter, hue, saturation, exposure, data_aug=self.train)
        label = torch.from_numpy(label)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        self.seen = self.seen + self.num_workers
        return (img, label)

    @staticmethod
    def get_labpath(imgpath):
        subdir = 'labels'
        labpath = imgpath.replace('training/images', subdir) \
                         .replace('evaluation/images', subdir) \
                         .replace('JPEGImages', subdir) \
                         .replace('.jpg', '.txt').replace('.png', '.txt')
        return labpath

    @staticmethod
    def is_valid(imgpath):
        labpath = listDataset.get_labpath(imgpath.rstrip())
        if os.path.getsize(labpath):
            bs = np.loadtxt(labpath)
            if bs is not None:
                bs = np.reshape(bs, (-1, 5))
                clsset = set(bs[:,0].astype(np.int).tolist())
                if not clsset.isdisjoint(set(cfg.base_ids)):
                    return True
        return False




class MetaDataset(Dataset):
    def __init__(self,
            metafiles,
            imgsz=1024,
            train=False,
            transform=None,
            target_transform=None,
            num_workers=0,
            ensemble=False,
            with_ids=False):

        # Backup labeled image paths (for meta-model)
        if train:
            self.classes = cfg.base_classes
            factor = 1
            if cfg.data == 'coco':
                factor = 4
        else:
            # self.classes = cfg.base_classes
            if cfg.data == 'coco':
                self.classes = cfg.base_classes
            else:
                self.classes = cfg.classes
                # self.classes = cfg.base_classes
            factor = 10
        print('num of base classes: ', len(self.classes))

        nbatch = factor * 100 * 64 * cfg.num_gpus // cfg.batch_size
        # nbatch = factor * 500 * 64 * 200 * cfg.num_gpus // cfg.batch_size
        if cfg.tuning:
            # nbatch = factor * 500 * 64 * 100 * cfg.num_gpus // cfg.batch_size
            nbatch = factor * 100 * 64 * cfg.num_gpus // cfg.batch_size
        print('-----nbatch-------: ',nbatch)


        metainds = [[]] * len(self.classes)
        with open(metafiles, 'r') as f:
            metafiles = []
            for line in f.readlines():
                pair = line.rstrip().split()
                if len(pair) == 2:
                    pass
                elif len(pair) == 4:
                    pair = [pair[0]+' '+pair[1], pair[2]+' '+pair[3]]
                else:
                    raise NotImplementedError('{} not recognized'.format(pair))
                metafiles.append(pair)
            # metafiles = [tuple(line.rstrip().split()) for line in f.readlines()]
            metafiles = {k: topath(v) for k, v in metafiles}

            self.metalines = [[]] * len(self.classes)
            for i, clsname in enumerate(self.classes):
                with open(metafiles[clsname], 'r') as imgf:
                    lines = [topath(l) for l in imgf.readlines()]
                    self.metalines[i] = lines
                    if ensemble:
                        metainds[i] = list(zip([i]*len(lines), list(range(len(lines)))))
                    else:
                        inds = np.random.choice(range(len(lines)), nbatch).tolist()
                        metainds[i] = list(zip([i] * nbatch, inds))

        # self.inds = sum(metainds, []) if ensemble else sum(list(zip(*metainds)), ())
        self.inds = list(itertools.chain(*metainds)) if ensemble else list(itertools.chain.from_iterable(zip(*metainds)))
        self.inds = tuple(self.inds)

        self.meta_cnts = [len(ls) for ls in self.metalines]
        if cfg.randmeta:
            self.inds = list(self.inds)
            random.shuffle(self.inds)
            self.inds = tuple(self.inds)

        self.with_ids = with_ids
        self.ensemble = ensemble
        self.batch_size = len(self.classes) * cfg.num_gpus
        self.meta_shape = (imgsz, imgsz)
        self.mask_shape = (imgsz, imgsz)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.num_workers = num_workers
        self.meta_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        # if ensemble:
        #     import pickle
        #     l = len(self.inds)
        #     if os.path.exists('inds_{}.pkl'.format(l)):
        #         with open('inds_{}.pkl'.format(l), 'rb') as f:
        #             self.inds = pickle.load(f)
        #             self.inds = tuple(self.inds)
        #     else:
        #         self.inds = self.filter(self.inds)
        #         with open('inds_{}.pkl'.format(l), 'wb') as f:
        #             pickle.dump(self.inds, f)
            # self.inds = self.filter(self.inds)
            # with open('inds.pkl', 'rb') as f:
            #     self.inds = pickle.load(f)

        self.nSamples = len(self.inds)

    def __len__(self):
        return self.nSamples

  
    def get_img_mask(self, img, lab, merge=True):
        w, h = self.mask_shape

        # w, h = img.width, img.height

        points = []
        points.append(lab[0:2])
        points.append(lab[2:4])
        points.append(lab[4:6])
        points.append(lab[6:8])
        points = np.array(points, np.float32)
        points = np.array(points, np.int32)

        mask = np.zeros((h, w, 3))
        cv2.fillPoly(mask, [points], (1, 0, 0))

        mask = np.asarray([mask[:, :, 0]])
        mask = torch.from_numpy(mask)

        img = self.meta_transform(img)

        if merge:
            return torch.cat([img, mask])
        else:
            return img, mask

    def get_metaimg(self, clsid, imgpath):
        if isinstance(imgpath, int):
            imgpath = self.metalines[clsid][imgpath].rstrip()
        elif isinstance(imgpath, str):
            pass
        else:
            raise NotImplementedError("{}: img path not recognized")

        labpath = self.get_labpath(imgpath, self.classes[clsid])
        img, lab = load_data_with_label(imgpath, labpath)
        return img, lab

    def get_metain(self, clsid, metaind):
        meta_img, meta_lab = self.get_metaimg(clsid, metaind)
        if meta_lab:
            for lab in meta_lab:
                # print(lab)
                img, mask = self.get_img_mask(meta_img, lab, merge=False)
                if mask is None:
                    continue
                return (img, mask)

        # In case the selected meta image has only difficult objects
        while True and not self.ensemble:
        # while True:
            meta_imgpath = random.sample(self.metalines[clsid], 1)[0].rstrip()
            meta_img, meta_lab = self.get_metaimg(clsid, meta_imgpath)
            if not meta_lab:
                continue
            for lab in meta_lab:
                img, mask = self.get_img_mask(meta_img, lab, merge=False)
                if mask is None:
                    continue
                return (img, mask)
        return (None, None)

    def filter(self, inds):
        newinds = []
        pbar = enumerate(inds)
        nb = len(inds)
        pbar = tqdm(pbar, total=nb,desc='===>filtering')
        for i,(clsid, metaind) in pbar:
            img, mask = self.get_metain(clsid, metaind)
            if img is not None:
                newinds.append((clsid, metaind))
        print('before filtering:',len(inds))
        print('after filtering:',len(newinds))
        return newinds

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'

        clsid, metaind = self.inds[index]

        img, mask = self.get_metain(clsid, metaind)
        # ipath = self.metalines[clsid][metaind]


        if self.with_ids:
            return (img, mask, clsid)
        else:
            return (img, mask)

    @staticmethod
    def get_labpath(imgpath, cls_name):
        if cfg.tuning:
            labpath = imgpath.replace('training/images', 'labels_1c/{}_{}shot'.format(cls_name, cfg.shot)) \
                .replace('.jpg', '.txt').replace('.png', '.txt')
        else:
            labpath = imgpath.replace('training/images', 'labels_1c/{}'.format(cls_name)) \
                .replace('.jpg', '.txt').replace('.png', '.txt')

        return labpath


# if __name__ == "__main__":
#     lab = ['928.0', '498.0', '858.0', '292.0', '1024.0', '241.0', '1024.0', '466.0']
#     img = np.zeros((1024, 1024, 3))
#     get_img_mask(img, lab, merge=False)