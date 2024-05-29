import os
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon, Circle
import numpy as np
import dota_utils as util
from collections import defaultdict
import cv2


def _isArrayLike(obj):
    if type(obj) == str:
        return False
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')


class DOTA:
    def __init__(self, imagePath, labelPath):
        # self.basepath = basepath
        # self.labelpath = os.path.join(basepath, 'labels')
        self.labelpath = labelPath
        self.imagepath = imagePath
        self.imgpaths = util.GetFileFromThisRootDir(self.labelpath)
        self.imglist = [util.custombasename(x) for x in self.imgpaths]
        self.catToImgs = defaultdict(list)
        self.ImgToAnns = defaultdict(list)
        self.createIndex()

    def createIndex(self):
        for filename in self.imgpaths:
            objects = util.parse_dota_poly(filename)
            imgid = util.custombasename(filename)
            self.ImgToAnns[imgid] = objects
            for obj in objects:
                cat = obj['name']
                self.catToImgs[cat].append(imgid)

    def getImgIds(self, catNms=[]):
        """
        :param catNms: category names
        :return: all the image ids contain the categories
        """
        catNms = catNms if _isArrayLike(catNms) else [catNms]
        if len(catNms) == 0:
            return self.imglist
        else:
            imgids = []
            for i, cat in enumerate(catNms):
                if i == 0:
                    imgids = set(self.catToImgs[cat])
                else:
                    imgids |= set(self.catToImgs[cat])
        return list(imgids)

    def loadAnns(self, catNms=[], imgId=None, difficult=None):
        """
        :param catNms: category names
        :param imgId: the img to load anns
        :return: objects
        """
        catNms = catNms if _isArrayLike(catNms) else [catNms]
        objects = self.ImgToAnns[imgId]
        if len(catNms) == 0:
            return objects
        outobjects = [obj for obj in objects if (obj['name'] in catNms)]
        return outobjects

    def showAnns(self, objects, imgId, c):
        """
        :param catNms: category names
        :param objects: objects to show
        :param imgId: img to show
        :param range: display range in the img
        :return:
        """
        img = self.loadImgs(imgId)[0]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.figure()
        plt.imshow(img)
        # plt.show()
        plt.axis('off')
        # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        # plt.margins(0, 0)

        ax = plt.gca()
        ax.set_autoscale_on(False)
        polygons = []
        color = []
        circles = []
        r = 5

        for obj in objects:
            # poly = obj['poly']
            # polygons.append(Polygon(poly))
            # color.append(c[classes.index(obj['name'])])

            # if (obj['name'] in ['airplane', 'baseball-diamond', 'tennis-court']):
            # if (obj['name'] in ['airplane', 'baseballfield', 'tenniscourt', 'trainstation', 'windmill']):
            if (obj['name'] in ['airplane',
                                'airport',
                                'baseballfield',
                                'basketballcourt',
                                'bridge',
                                'chimney',
                                'dam',
                                'Expressway-Service-area',
                                'Expressway-toll-station',
                                'golffield',
                                'groundtrackfield',
                                'harbor',
                                'overpass',
                                'ship',
                                'stadium',
                                'storagetank',
                                'tenniscourt',
                                'trainstation',
                                'vehicle',
                                'windmill']):
                poly = obj['poly']
                polygons.append(Polygon(poly))
                # color.append('#FF0000')
                color.append(c[classes.index(obj['name'])])
            #
            # x_min = min(poly[0][0],poly[1][0],poly[2][0],poly[3][0])
            # y_min = min(poly[0][1],poly[1][1],poly[2][1],poly[3][1])
            #
            # # Add object name to top left corner
            # ax.text(x_min, y_min, obj['name'], fontsize=8, color='black', verticalalignment='top')

            # if(obj['name'] == 'airplane'):
            #     color.append('#FF0000')
            # else:
            #     color.append('#FFFF00')

            # point = poly[0]
            # circle = Circle((point[0], point[1]), r)
            # circles.append(circle)
        # p = PatchCollection(polygons, facecolors=color, linewidths=0, alpha=0.4)
        # ax.add_collection(p)
        p = PatchCollection(polygons, facecolors='none', edgecolors=color, linewidths=1)
        ax.add_collection(p)
        # p = PatchCollection(circles, facecolors='red')
        # ax.add_collection(p)
        plt.savefig(save_path + imgId + '.jpg', bbox_inches='tight', dpi=400, pad_inches=0.0)
        # plt.show()

    def loadImgs(self, imgids=[]):
        """
        :param imgids: integer ids specifying img
        :return: loaded img objects
        """
        print('isarralike:', _isArrayLike(imgids))
        imgids = imgids if _isArrayLike(imgids) else [imgids]
        print('imgids:', imgids)
        imgs = []
        for imgid in imgids:
            filename = os.path.join(self.imagepath, imgid + '.jpg')
            print('filename:', filename)
            img = cv2.imread(filename)
            imgs.append(img)
        return imgs


if __name__ == '__main__':
    nwpu_classes = ['airplane',
                    'ship',
                    'storage-tank',
                    'baseball-diamond',
                    'tennis-court',
                    'basketball-court',
                    'ground-track-field',
                    'harbor',
                    'bridge',
                    'vehicle']
    dior_classes = ['airplane',
                    'airport',
                    'baseballfield',
                    'basketballcourt',
                    'bridge',
                    'chimney',
                    'dam',
                    'Expressway-Service-area',
                    'Expressway-toll-station',
                    'golffield',
                    'groundtrackfield',
                    'harbor',
                    'overpass',
                    'ship',
                    'stadium',
                    'storagetank',
                    'tenniscourt',
                    'trainstation',
                    'vehicle',
                    'windmill']

    classes = dior_classes

    imagePath = r'C:\Users\admin\Desktop\DOTAvisulizationTest\compare_experiment_ablation\images'
    labelPath = r'C:\Users\admin\Desktop\DOTAvisulizationTest\compare_experiment_ablation\labelTxt_MuiltScale'
    save_path = r'C:\Users\admin\Desktop\DOTAvisulizationTest\compare_experiment_ablation\result_MuiltScale/'

    # c = ('#FF3838', '#FF9D97', '#FF701F', '#FFB21D', '#CFD231', '#48F90A', '#92CC17', '#3DDB86', '#1A9334', '#00D4BB',
    #      '#2C99A8', '#00C2FF', '#344593', '#6473FF', '#0018EC', '#8438FF', '#520085', '#CB38FF', '#FF95C8', '#FF37C7')
    c = ('#FF3838', '#FFB21D', '#CFD231', '#00C2FF', '#344593', '#3DDB86', '#1A9334', '#00D4BB', '#48F90A', '#92CC17',
         '#2C99A8', '#CB38FF', '#FF95C8', '#6473FF', '#520085', '#0018EC', '#FF701F', '#8438FF', '#FF9D97', '#FF37C7')
    for name in classes:
        print(name + ":" + c[classes.index(name)])

    examplesplit = DOTA(imagePath, labelPath)

    imgids = examplesplit.getImgIds(catNms=classes)
    img = examplesplit.loadImgs(imgids)
    for imgid in imgids:
        anns = examplesplit.loadAnns(imgId=imgid)
        examplesplit.showAnns(anns, imgid, c)
