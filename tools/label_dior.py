import os
from PIL import Image


sets = ['training', 'evaluation']


if __name__ == '__main__':
    import sys

    datapath = os.path.abspath(sys.argv[1])


    for set in sets:
        files = os.listdir(os.path.join(datapath, '{}/images'.format(set)))
        image_ids = [x.strip('.jpg') for x in files]
        list_file = os.path.join(datapath, '{}.txt'.format(set))

        with open(list_file, 'w') as out_f:
            for id in image_ids:
                out_f.write('{}/{}/images/{}.jpg\n'.format(datapath, set, id))

