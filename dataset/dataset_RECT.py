from torch.utils.data import Dataset
import torch
import codecs
import cv2
import numpy as np
import os
import copy
from torchvision import transforms
import ImageProcess.ImageTransform as ImageTransform
from PIL import Image
import matplotlib.pyplot as plt
import math


class M500Dataset(Dataset):
    """
    the father class of ICDAR2015 dataset reading
    """

    def __init__(self, train=True):
        if train:
            self.image_dir = "dataset/MSRA-TD500/train/"
            self.labels_dir = "dataset/MSRA-TD500/train/"
        else:
            self.image_dir = 'dataset/MSRA-TD500/test/'
            self.labels_dir = 'dataset/MSRA-TD500/test/'

        self.train = train
        self.img_files = None
        self.img_train_size = [1280, 720]
        self.all_labels = self.read_labels()

    def __len__(self):
        return len(self.all_labels)

    def read_image(self, index):
        image_path = self.img_files[index]
        labels = copy.deepcopy(self.all_labels[index])
        img = ImageTransform.ReadImage(image_path)

        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        return img, labels

    def read_labels(self):
        def rec_rotate(xx, yy, width, height, theta):
            def rotate(angle, x, y):
                """
                »ùÓÚÔ­µãµÄ»¡¶ÈÐý×ª

                :param angle:   »¡¶È
                :param x:       x
                :param y:       y
                :return:
                """
                rotatex = math.cos(angle) * x - math.sin(angle) * y
                rotatey = math.cos(angle) * y + math.sin(angle) * x
                return rotatex, rotatey

            def xy_rorate(theta, x, y, centerx, centery):
                """
                Õë¶ÔÖÐÐÄµã½øÐÐÐý×ª

                :param theta:
                :param x:
                :param y:
                :param centerx:
                :param centery:
                :return:
                """
                r_x, r_y = rotate(theta, x - centerx, y - centery)
                return centerx + r_x, centery + r_y

            centerx = xx + width / 2
            centery = yy + height / 2

            x1, y1 = xy_rorate(theta, xx, yy, centerx, centery)
            x2, y2 = xy_rorate(theta, xx + width, yy, centerx, centery)
            x3, y3 = xy_rorate(theta, xx, yy + height, centerx, centery)
            x4, y4 = xy_rorate(theta, xx + width, yy + height, centerx, centery)

            return int(x1), int(y1), int(x2), int(y2), int(x4), int(y4), int(x3), int(y3)

        dirs = os.listdir(self.image_dir)
        img_files = [d for d in dirs if len(d) == 12]
        label_files = [d for d in dirs if len(d) == 11]
        img_files = sorted(img_files, key=lambda x: int(x[4:-4]))
        label_files = sorted(label_files, key=lambda x: int(x[4:-3]))
        self.img_files = [self.image_dir+d for d in img_files]
        label_files = [self.labels_dir+d for d in label_files]
        ls = []
        for label_file in label_files:
            # utf-8_sig for bom_utf-8
            with codecs.open(label_file, encoding="utf-8_sig") as f:
                lines = f.readlines()
                tmp = {}
                tmp['coor'] = []
                tmp['content'] = []
                tmp['ignore'] = []
                tmp['area'] = []
                for line in lines:
                    content = line.strip('\r\n').split(' ')
                    coor = rec_rotate(int(content[2]), int(content[3]), int(content[4]), int(content[5]), float(content[6]))
                    tmp['coor'].append(coor)
                    tmp['content'].append(content[0])
                    if content[1] == '1':
                        tmp['ignore'].append(True)
                    else:
                        tmp['ignore'].append(False)
                    coor = np.array(coor).reshape([4, 2])
                    tmp['area'].append(cv2.contourArea(coor))
                index = np.argsort(np.array(tmp['area']))[::-1]
                tmp['coor'] = (np.array(tmp['coor'])[index]).tolist()
                tmp['content'] = (np.array(tmp['content'])[index]).tolist()
                tmp['area'] = (np.array(tmp['area'])[index]).tolist()
                tmp['ignore'] = (np.array(tmp['ignore'])[index]).tolist()
                ls.append(tmp)
        return ls


class ICDAR2013Dataset(Dataset):
    """
    the father class of ICDAR2015 dataset reading
    """

    def __init__(self, train=True):
        if train:
            self.image_dir = "dataset/ICDAR2013/Challenge2_Training_Task12_Images/"
            self.labels_dir = "dataset/ICDAR2013/Challenge2_Training_Task1_GT/"
        else:
            self.image_dir = 'dataset/ICDAR2013/Challenge2_Test_Task12_Images/'
            self.labels_dir = 'dataset/ICDAR2013/Challenge2_Test_Task1_GT/'

        self.train = train
        self.img_files = None
        self.all_labels = self.read_labels()

    def __len__(self):
        return len(self.all_labels)

    def read_image(self, index):
        image_path = self.img_files[index]
        labels = copy.deepcopy(self.all_labels[index])
        img = ImageTransform.ReadImage(image_path)
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        return img, labels

    def read_labels(self):
        def distance(a, b):
            return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2

        img_files = os.listdir(self.image_dir)
        label_files = os.listdir(self.labels_dir)
        if self.train:
            img_files = sorted(img_files, key=lambda x: int(x[:-4]))
            label_files = sorted(label_files, key=lambda x: int(x[3:-4]))
        else:
            img_files = sorted(img_files, key=lambda x: int(x[4:-4]))
            label_files = sorted(label_files, key=lambda x: int(x[7:-4]))
        self.img_files = [self.image_dir+d for d in img_files]
        label_files = [self.labels_dir+d for d in label_files]
        ls = []
        for label_file in label_files:
            # utf-8_sig for bom_utf-8
            with codecs.open(label_file, encoding="utf-8_sig") as f:
                lines = f.readlines()
                tmp = {}
                tmp['coor'] = []
                tmp['content'] = []
                tmp['ignore'] = []
                tmp['area'] = []
                tmp['min_side'] = []
                for line in lines:
                    if self.train:
                        content = line.split(' ')
                    else:
                        content = line.split(', ')
                    coor = [int(content[0]), int(content[1]), int(content[2]), int(content[1]),
                            int(content[2]), int(content[3]), int(content[0]), int(content[3]), ]
                    tmp['coor'].append(coor)
                    content[-1] = content[-1].strip("\r\n")
                    tmp['content'].append(content[-1])
                    tmp['ignore'].append(False)
                    coor = np.array(coor).reshape([4, 2])
                    tmp['area'].append(cv2.contourArea(coor))
                    dists = []
                    for i in range(4):
                        dists.append(distance(coor[i], coor[(i + 1) % 4]))
                    tmp['min_side'].append(min(dists))
                index = np.argsort(np.array(tmp['area']))[::-1]
                tmp['coor'] = (np.array(tmp['coor'])[index]).tolist()
                tmp['content'] = (np.array(tmp['content'])[index]).tolist()
                tmp['area'] = (np.array(tmp['area'])[index]).tolist()
                tmp['ignore'] = (np.array(tmp['ignore'])[index]).tolist()
                tmp['min_side'] = (np.array(tmp['min_side'])[index]).tolist()
                ls.append(tmp)
        return ls


class ICDAR2015Dataset(Dataset):
    """
    the father class of ICDAR2015 dataset reading
    """

    def __init__(self, train=True):
        if train:
            self.image_dir = "dataset/ICDAR2015/training_images/"
            self.labels_dir = "dataset/ICDAR2015/training_gt/"
        else:
            self.image_dir = 'dataset/ICDAR2015/test_images/'
            self.labels_dir = 'dataset/ICDAR2015/test_gt/'

        self.train = train
        self.img_files = None
        self.all_labels = self.read_labels()

    def __len__(self):
        return len(self.all_labels)

    def read_image(self, index):
        image_path = self.img_files[index]
        labels = copy.deepcopy(self.all_labels[index])
        img = ImageTransform.ReadImage(image_path)
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        return img, labels

    def read_labels(self):
        def distance(a, b):
            return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2
        img_files = os.listdir(self.image_dir)
        label_files = os.listdir(self.labels_dir)
        img_files = sorted(img_files, key=lambda x: int(x[4:-4]))
        label_files = sorted(label_files, key=lambda x: int(x[7:-4]))
        self.img_files = [self.image_dir+d for d in img_files]
        label_files = [self.labels_dir+d for d in label_files]

        ls = []
        for label_file in label_files:
            # utf-8_sig for bom_utf-8
            with codecs.open(label_file, encoding="utf-8_sig") as f:
                lines = f.readlines()
                tmp = {}
                tmp['coor'] = []
                tmp['content'] = []
                tmp['ignore'] = []
                tmp['area'] = []
                tmp['min_side'] = []
                for line in lines:
                    content = line.split(',')
                    coor = [int(n) for n in content[:8]]
                    tmp['coor'].append(coor)
                    content[8] = content[8].strip("\r\n")
                    tmp['content'].append(content[8])
                    if content[8] == '###':
                        tmp['ignore'].append(True)
                    else:
                        tmp['ignore'].append(False)
                    coor = np.array(coor).reshape([4, 2])
                    tmp['area'].append(cv2.contourArea(coor))
                    dists = []
                    for i in range(4):
                        dists.append(distance(coor[i], coor[(i + 1) % 4]))
                    tmp['min_side'].append(min(dists))
                index = np.argsort(np.array(tmp['area']))[::-1]
                tmp['coor'] = (np.array(tmp['coor'])[index]).tolist()
                tmp['content'] = (np.array(tmp['content'])[index]).tolist()
                tmp['area'] = (np.array(tmp['area'])[index]).tolist()
                tmp['ignore'] = (np.array(tmp['ignore'])[index]).tolist()
                tmp['min_side'] = (np.array(tmp['min_side'])[index]).tolist()
                ls.append(tmp)
        return ls


class ICDAR2017Dataset(Dataset):
    """
    the father class of ICDAR2017 dataset reading
    """

    def __init__(self, train=True):
        if train:
            self.image_dir = "dataset/ICDAR2017/ch8_training_images/"
            self.labels_dir = "dataset/ICDAR2017/ch8_training_localization_transcription_gt/"
        else:
            self.image_dir = "dataset/ICDAR2017/ch8_validation_images/"
            self.labels_dir = "dataset/ICDAR2017/ch8_validation_localization_transcription_gt/"
        self.train = train
        self.img_files = None
        self.all_labels = self.read_labels()

    def __len__(self):
        return len(self.all_labels)

    def read_image(self, index):
        image_path = self.img_files[index]
        labels = copy.deepcopy(self.all_labels[index])
        img = ImageTransform.ReadImage(image_path)
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        return img, labels

    def read_labels(self):
        def distance(a, b):
            return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2
        img_files = os.listdir(self.image_dir)
        label_files = os.listdir(self.labels_dir)
        img_files = sorted(img_files, key=lambda x: int(x[4:-4]))
        label_files = sorted(label_files, key=lambda x: int(x[7:-4]))
        self.img_files = [self.image_dir+d for d in img_files]
        label_files = [self.labels_dir+d for d in label_files]
        ls = []
        for label_file in label_files:
            # utf-8_sig for bom_utf-8
            with codecs.open(label_file, encoding="utf-8_sig") as f:
                lines = f.readlines()
                tmp = {}
                tmp['coor'] = []
                tmp['content'] = []
                tmp['ignore'] = []
                tmp['area'] = []
                tmp['min_side'] = []
                for line in lines:
                    content = line.split(',')
                    coor = [int(content[0]), int(content[1]), int(content[2]), int(content[3]),
                            int(content[4]), int(content[5]), int(content[6]), int(content[7]), ]
                    tmp['coor'].append(coor)
                    content[-1] = content[-1].strip("\r\n")
                    tmp['content'].append(content[-1])
                    if tmp['content'][-1]  == "###":
                        tmp['ignore'].append(True)
                    else:
                        tmp['ignore'].append(False)
                    coor = np.array(coor).reshape([4, 2])
                    tmp['area'].append(cv2.contourArea(coor))
                    dists = []
                    for i in range(4):
                        dists.append(distance(coor[i], coor[(i + 1) % 4]))
                    tmp['min_side'].append(min(dists))
                index = np.argsort(np.array(tmp['area']))[::-1]
                tmp['coor'] = (np.array(tmp['coor'])[index]).tolist()
                tmp['content'] = (np.array(tmp['content'])[index]).tolist()
                tmp['area'] = (np.array(tmp['area'])[index]).tolist()
                tmp['ignore'] = (np.array(tmp['ignore'])[index]).tolist()
                tmp['min_side'] = (np.array(tmp['min_side'])[index]).tolist()
                ls.append(tmp)
        for i in range(len(self.img_files)-len(label_files)):
            tmp = {}
            tmp['coor'] = []
            tmp['content'] = []
            tmp['ignore'] = []
            tmp['area'] = []
            ls.append(tmp)
        return ls


# 每次使用前需要确认数据集
INHERIT = ICDAR2015Dataset


class RectDataset(INHERIT):
    """
    dataset reading
    """

    def __init__(self, img_size, crop=False, light=False, filp=False,
                 rotate=False, noise=False, color=False, train=True):

        super(RectDataset, self).__init__(train=train)
        self.train = train
        self.img_size = img_size
        self.scale_size = (720, 1280)
        self.data_aug = ImageTransform.TextDataAugment(crop=crop, light=light, filp=filp, rotate=rotate,
                                                       noise=noise, color=color)

    def __getitem__(self, index):
        if self.train:
            img, ignore_mask, labels = self.train_data(index)
        else:
            img, ignore_mask, labels = self.test_data(index)
        pixel_mask, pixel_weight, img_ignore = self.label_to_corner_mask(labels, list(img.shape[1:]), ignore_mask)


        return {'image': torch.FloatTensor(img), 'pixel_mask': torch.FloatTensor(pixel_mask),
                'pixel_ignore': torch.FloatTensor(img_ignore),
                'pixel_weight': torch.FloatTensor(pixel_weight), 'label': labels}

    def train_data(self, item):
        """
        Data Augmentation Online
        :param item: the serial number of image
        :return:
        """
        img, labels = self.read_image(item)

        labels = copy.deepcopy(self.all_labels[item])
        scale = min(self.scale_size[0], self.scale_size[1])/min(img.shape[0], img.shape[1])
        img = cv2.resize(img, dsize=None, fx=scale, fy=scale)
        labels['coor'] = list(map(list, (np.array(labels['coor'])*scale).astype('int')))

        img, img_ignore, labels = self.data_aug(img, labels, self.img_size)
        # labels = self.filter_labels(labels, method='rai', scale=scale)
        labels = self.filter_labels(labels, method='msi', scale=scale)

        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        img = transforms.ColorJitter(brightness=32.0 / 255, saturation=0.5)(img)

        img = ImageTransform.normalize(np.array(img), mean=(0.485, 0.456, 0.406),
                                       variance=(0.229, 0.224, 0.225))
        # HWC to CHW
        img = img.transpose(2, 0, 1)
        return img, img_ignore, labels

    def test_data(self, item):
        img, labels = self.read_image(item)

        labels = copy.deepcopy(self.all_labels[item])
        scale = 720/min(img.shape[0], img.shape[1])
        img = cv2.resize(img, dsize=None, fx=scale, fy=scale)
        labels['coor'] = list(map(list, (np.array(labels['coor'])*scale).astype('int')))

        img_ignore = np.ones((img.shape[0:2]), dtype=np.uint8)
        img = ImageTransform.normalize(img)
        img = img.transpose(2, 0, 1)
        return img, img_ignore, labels

    @staticmethod
    def filter_labels(labels, method, scale=0):
        """
        :param labels: the labels of region
        :param method: 'msi' for min ignore, 'rai' for remain area ignore
        :return:
        """

        def distance(a, b):
            return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2

        def min_side_ignore(labelm, min_side, scale):
            labelm = np.array(labelm).reshape([4, 2])
            dists = []
            for i in range(4):
                dists.append(distance(labelm[i], labelm[(i + 1) % 4]))
            if min(dists) / (min_side*scale*scale) < 0.34:
                return True  # ignore it
            else:
                return False

        def remain_area_ignore(labelr, origin_area, scale):
            """
            remain label that larger than 0.2 the area of new label / the area of original region
            :param labelr:
            :param origin_area:
            :return:
            """
            labelr = np.array(labelr).reshape([4, 2])
            area = cv2.contourArea(labelr)
            if area / (origin_area*scale*scale) < 0.2:
                return True
            else:
                return False

        if method == 'msi':
            ignore = list(map(min_side_ignore, labels['coor'], labels['min_side'], [scale]*len(labels['coor'])))
        elif method == 'rai':
            ignore = list(map(remain_area_ignore, labels['coor'], labels['area'], [scale]*len(labels['coor'])))
        else:
            ignore = [False] * 8
        # the original 'ignore' is '###'
        labels["ignore"] = list(map(lambda a, b: a or b, labels['ignore'], ignore))
        return labels

    @staticmethod
    def label_to_corner_mask(labels, img_size, ignore_mask, factor=1):

        ignore = labels['ignore']
        labels = labels['coor']
        assert len(ignore) == len(labels)
        labels = np.array(labels).reshape([-1, 1, 4, 2])
        labels = np.array(labels / factor, dtype=int)
        pixel_mask_size = [int(i / factor) for i in img_size]
        pixel_weight = np.ones(pixel_mask_size, dtype=np.float)
        pixel_mask = np.zeros(pixel_mask_size, dtype=np.uint8)
        pixel_guassian_mask = np.zeros(pixel_mask_size, dtype=np.float)
        for i in range(labels.shape[0]):
            if not ignore[i]:
                pixel_mask_tmp = np.zeros(pixel_mask_size, dtype=np.uint8)
                cv2.drawContours(pixel_mask_tmp, labels[i], -1, 1, thickness=-1)
                pixel_mask += pixel_mask_tmp
        pixel_mask[pixel_mask != 1] = 0
        real_box_num = 0
        for i in range(labels.shape[0]):
            if not ignore[i]:
                pixel_mask_tmp = np.zeros(pixel_mask_size, dtype=np.uint8)
                cv2.drawContours(pixel_mask_tmp, labels[i], -1, 1, thickness=-1)
                pixel_mask_tmp *= pixel_mask
                if np.count_nonzero(pixel_mask_tmp) > 0:
                    real_box_num += 1

        if real_box_num == 0:
            return pixel_guassian_mask, pixel_weight, ignore_mask
        labels = (labels / factor).astype(int)
        pixel_mask_area = np.count_nonzero(pixel_mask)
        avg_weight_per_box = pixel_mask_area / real_box_num
        # generate the label of center
        for i in range(labels.shape[0]):
            if not ignore[i]:
                pixel_mask_tmp = np.zeros(pixel_mask_size)
                pixel_mask_tmp = \
                    ImageTransform.keep_mask(labels[i], pixel_mask_size, pixel_mask_tmp)
                pixel_guassian_mask = np.maximum(pixel_guassian_mask, pixel_mask_tmp)

                pixel_weight_tmp = np.zeros(pixel_mask_size, dtype=np.float)
                cv2.drawContours(pixel_weight_tmp, [labels[i]], -1, avg_weight_per_box, thickness=-1)
                pixel_weight_tmp[pixel_guassian_mask != pixel_mask_tmp] = 0

                area = np.count_nonzero(pixel_weight_tmp)
                if area <= 0:
                    continue
                pixel_weight_tmp /= area
                pixel_weight_tmp[pixel_weight_tmp > 0] += 1
                pixel_weight_tmp[pixel_weight_tmp == 0] = 1
                pixel_weight = np.maximum(pixel_weight, pixel_weight_tmp)
        inter_weight = ImageTransform.one_hot2dist(pixel_guassian_mask)
        inter_weight *= pixel_weight

        return pixel_guassian_mask, inter_weight, ignore_mask
