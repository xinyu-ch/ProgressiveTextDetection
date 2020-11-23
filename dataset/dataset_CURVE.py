from torch.utils.data import Dataset
import torch
import codecs
import cv2
import numpy as np
import os
import copy
from torchvision import transforms
import ImageProcess.ImageTransformCurve as ImageTransform
from PIL import Image
import matplotlib.pyplot as plt


class CTW500Dataset(Dataset):
    """
    the father class of ICDAR2015 dataset reading
    """

    def __init__(self, train=True):
        if train:
            self.image_dir = "dataset/CTW1500/train_images/"
            self.labels_dir = "dataset/CTW1500/ctw1500_train_labels/"
        else:
            self.image_dir = 'dataset/CTW1500/test_images/'
            self.labels_dir = 'dataset/CTW1500/gt_ctw1500/'

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
        img_files = os.listdir(self.image_dir)
        label_files = os.listdir(self.labels_dir)
        if self.train:
            img_files = sorted(img_files, key=lambda x: int(x[:-4]))
            label_files = sorted(label_files, key=lambda x: int(x[:-4]))
        else:
            img_files = sorted(img_files, key=lambda x: int(x[:-4]))
            label_files = sorted(label_files, key=lambda x: int(x[:-4]))
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
                    if self.train:
                        content = line.split('#$#')
                    else:
                        content = line.split(',####')
                    coor = list(map(int, content[0].split(',')))
                    tmp['coor'].append(coor)
                    content[-1] = content[-1].strip("\r\n")
                    tmp['content'].append(content[-1])
                    if content[-1] == "###":
                        tmp['ignore'].append(True)
                    else:
                        tmp['ignore'].append(False)
                    coor = np.array(coor).reshape([-1, 2])
                    tmp['area'].append(cv2.contourArea(coor))
                index = np.argsort(np.array(tmp['area']))[::-1]
                tmp['coor'] = (np.array(tmp['coor'])[index]).tolist()
                tmp['content'] = (np.array(tmp['content'])[index]).tolist()
                tmp['area'] = (np.array(tmp['area'])[index]).tolist()
                tmp['ignore'] = (np.array(tmp['ignore'])[index]).tolist()
                ls.append(tmp)
        return ls


# 每次使用前需要确认数据集
INHERIT = CTW500Dataset


class CurveDataset(INHERIT):
    """
    dataset reading
    """

    def __init__(self, img_size, crop=False, light=False, filp=False,
                 rotate=False, noise=False, color=False, train=True):

        super(CurveDataset, self).__init__(train=train)
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
        # labels = self.filter_labels(labels, method='rai')
        # labels = self.filter_labels(labels, method='msi')

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
    def filter_labels(labels, method):
        """
        :param labels: the labels of region
        :param method: 'msi' for min ignore, 'rai' for remain area ignore
        :return:
        """

        def distance(a, b):
            return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2

        def min_side_ignore(labelm):
            labelm = np.array(labelm).reshape([-1, 2])
            dists = []
            for i in range(4):
                dists.append(distance(labelm[i], labelm[(i + 1) % 4]))
            if min(dists) < 25:
                return True  # ignore it
            else:
                return False

        def remain_area_ignore(labelr, origin_area):
            """
            remain label that larger than 0.2 the area of new label / the area of original region
            :param labelr:
            :param origin_area:
            :return:
            """
            labelr = np.array(labelr).reshape([4, 2])
            area = cv2.contourArea(labelr)
            if area < 400:
                return True
            else:
                return False

        if method == 'msi':
            ignore = list(map(min_side_ignore, labels['coor']))
        elif method == 'rai':
            ignore = list(map(remain_area_ignore, labels['coor'], labels['area']))
        else:
            ignore = [False] * 8
        # the original 'ignore' is '###'
        labels["ignore"] = list(map(lambda a, b: a or b, labels['ignore'], ignore))
        return labels

    @staticmethod
    def label_to_corner_mask(labels, img_size, ignore_mask):

        ignore = labels['ignore']
        labels = labels['coor']
        assert len(ignore) == len(labels)
        labels = np.array(list(map(np.array, labels)))
        pixel_mask_size = img_size
        pixel_weight = np.ones(pixel_mask_size, dtype=np.float)
        # ignore_mask = cv2.resize(ignore_mask, tuple(pixel_mask_size))
        pixel_mask = np.zeros(pixel_mask_size, dtype=np.uint8)
        pixel_guassian_mask = np.zeros(pixel_mask_size, dtype=np.float)
        for i in range(labels.shape[0]):
            if not ignore[i]:
                pixel_mask_tmp = np.zeros(pixel_mask_size, dtype=np.uint8)
                cv2.drawContours(pixel_mask_tmp, labels[i].reshape(1, -1, 2), -1, 1, thickness=-1)
                pixel_mask += pixel_mask_tmp
        pixel_mask[pixel_mask != 1] = 0
        real_box_num = 0
        for i in range(labels.shape[0]):
            if not ignore[i]:
                pixel_mask_tmp = np.zeros(pixel_mask_size, dtype=np.uint8)
                cv2.drawContours(pixel_mask_tmp, labels[i].reshape(1, -1, 2), -1, 1, thickness=-1)
                pixel_mask_tmp *= pixel_mask
                if np.count_nonzero(pixel_mask_tmp) > 0:
                    real_box_num += 1

        if real_box_num == 0:
            # print("box num = 0")
            return pixel_guassian_mask, pixel_weight, ignore_mask
        pixel_mask_area = np.count_nonzero(pixel_mask)
        avg_weight_per_box = pixel_mask_area / real_box_num
        # generate the label of center
        for i in range(labels.shape[0]):
            if not ignore[i]:
                pixel_mask_tmp = np.zeros(pixel_mask_size)
                pixel_mask_tmp = ImageTransform.keep_mask(labels[i], pixel_mask_tmp)
                pixel_guassian_mask = np.maximum(pixel_guassian_mask, pixel_mask_tmp)

                pixel_weight_tmp = np.zeros(pixel_mask_size, dtype=np.float)
                cv2.drawContours(pixel_weight_tmp, [labels[i].reshape(1, -1, 2)], -1, avg_weight_per_box, thickness=-1)
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


if __name__ == '__main__':

    img_size = (512, 512)
    dataset = RectDataset(img_size, train=False)
