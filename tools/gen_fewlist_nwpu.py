import argparse
import random
import os
import numpy as np
from os import path

classes = ['airplane',
           'ship',
           'storage-tank',
           'baseball-diamond',
           'tennis-court',
           'basketball-court',
           'ground-track-field',
           'harbor',
           'bridge',
           'vehicle']

# few_nums = [1, 10]
# few_nums = [3, 5, 10]
few_nums = [15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
# few_nums = [20]
DROOT = r'C:\dataset'
root = DROOT + '/nwpulist/'
rootfile = DROOT + '/training.txt'
newlabelroot = DROOT + '/labels_1c/{}_{}shot'


def is_valid(imgpath, cls_name):
    imgpath = imgpath.strip()
    labpath = imgpath.replace('training/images', 'labels_1c/{}'.format(cls_name)) \
        .replace('JPEGImages', 'labels_1c/{}'.format(cls_name)) \
        .replace('.jpg', '.txt').replace('.png', '.txt')
    if os.path.getsize(labpath):
        return True
    else:
        return False


def gen_image_fewlist():
    print('-----------------------------------------------------------')
    print('----------- Generating fewlist  (images) ------------------')
    for i, clsname in enumerate(classes):
        print('===> Processing class: {}'.format(clsname))
        with open(path.join(root, '{}_training.txt'.format(clsname)), 'r') as f:
            name_list = f.readlines()
        num = max(few_nums)
        random.seed(i)
        # selected_list = random.sample(name_list, num)
        selected_list = []
        while len(selected_list) < num:
            x = random.sample(name_list, num)[0]
            if not is_valid(x, clsname):
                continue
            selected_list.append(x)

        for n in few_nums:
            with open(path.join(root, '{}shot_{}_train.txt'.format(n, clsname)), 'w') as f:
                for i in range(n):
                    f.write(selected_list[i])


# -------------------------------------------------------------------------------------

def get_bbox_fewlist(rootfile, shot):
    with open(rootfile) as f:
        names = f.readlines()
    random.seed(2018)
    cls_lists = [[] for _ in range(len(classes))]
    cls_counts = [0] * len(classes)

    for cls in classes:
        if not os.path.exists(newlabelroot.format(cls, shot)):
            os.mkdir(newlabelroot.format(cls, shot))

    while min(cls_counts) < shot:
        imgpath = random.sample(names, 1)[0]
        labpath = imgpath.strip().replace('training/images', 'labels') \
            .replace('JPEGImages', 'labels') \
            .replace('.jpg', '.txt').replace('.png', '.txt')
        newlabpath = imgpath.strip().replace('training/images', 'labels_1c/{}_{}shot') \
            .replace('.jpg', '.txt').replace('.png', '.txt')
        # To avoid duplication
        names.remove(imgpath)

        if not os.path.getsize(labpath):
            continue

        with open(labpath, 'r') as in_f:
            objs = [x.strip().split(' ') for x in in_f.readlines()]

        write_boxs = [[] for _ in range(len(classes))]
        label = []
        for b in objs:
            label.append(b[0:8])
            ci = classes.index(str(b[8]))
            if cls_counts[ci] < shot:
                cls_counts[ci] += 1
                write_boxs[ci].append(b)
                if imgpath not in cls_lists[ci]:
                    cls_lists[ci].append(imgpath)

        for i, boxs in enumerate(write_boxs):
            if len(boxs):
                with open(newlabpath.format(classes[i], shot), 'w') as f:
                    for box in boxs:
                        # print(' '.join([str(a) for a in box]) + '\n')
                        f.write(' '.join([str(a) for a in box]) + '\n')

    return cls_lists


def gen_bbox_fewlist():
    print('-----------------------------------------------------------')
    print('----------- Generating fewlist  (bboxes) ------------------')
    for n in few_nums:
        print('===> On {} shot ...'.format(n))
        filelists = get_bbox_fewlist(rootfile, n)
        for i, clsname in enumerate(classes):
            print('   | Processing class: {}'.format(clsname))
            with open(path.join(root, 'box_{}shot_{}_train.txt'.format(n, clsname)), 'w') as f:
                for name in filelists[i]:
                    f.write(name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, default=None, choices=['box', 'img', 'both'])
    args = parser.parse_args()

    if args.type is None or args.type == 'box':
        gen_bbox_fewlist()
    elif args.type == 'img':
        gen_image_fewlist()
    elif args.type == 'both':
        gen_image_fewlist()
        gen_bbox_fewlist()


if __name__ == '__main__':
    main()
