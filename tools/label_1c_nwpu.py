import os

sets = ['training', 'evaluation']

classes = classes = ['airplane',
                     'ship',
                     'storage-tank',
                     'baseball-diamond',
                     'tennis-court',
                     'basketball-court',
                     'ground-track-field',
                     'harbor',
                     'bridge',
                     'vehicle']

if __name__ == '__main__':
    import sys

    datapath = os.path.abspath(sys.argv[1])

    if not os.path.exists(os.path.join(datapath, 'nwpulist')):
        os.mkdir(os.path.join(datapath, 'nwpulist'))

    for class_name in classes:
        for image_set in sets:
            files = os.listdir(os.path.join(datapath, '{}/images'.format(image_set)))
            image_ids = [x.strip('.jpg') for x in files]
            list_file = os.path.join(datapath, 'nwpulist/{}_{}.txt'.format(class_name, image_set))

            label_dir = 'labels_1c/' + class_name
            if not os.path.exists(os.path.join(datapath, '{}/'.format(label_dir))):
                os.makedirs(os.path.join(datapath, '{}/'.format(label_dir)))

            with open(list_file, 'w') as out_f:
                for id in image_ids:
                    in_file = os.path.join(datapath, '{}/labelTxt/{}.txt'.format(image_set, id))
                    out_file = os.path.join(datapath, 'labels_1c/{}/{}.txt'.format(class_name, id))
                    image = os.path.join(datapath, '{}/images/{}.jpg'.format(image_set, id))

                    with open(in_file, 'r') as in_f:
                        objs = [x.strip().split(' ') for x in in_f.readlines()]

                    write_text = []
                    for obj in objs:
                        # if len(obj) < 5:
                        #     continue
                        cls = obj[8]
                        if cls != class_name:
                            continue

                        write_text.append(obj)

                    if len(write_text):
                        with open(out_file, 'w') as f:
                            for txt in write_text:
                                # print(' '.join(txt) + '\n')
                                f.write(' '.join(txt) + '\n')
                        out_f.write('{}/{}/images/{}.jpg\n'.format(datapath, image_set, id))
