from PIL import Image
import numpy as np
import random
import copy
import cv2
from skimage import exposure
from scipy import ndimage


class TextDataAugment(object):
    def __init__(self, crop=False, light=False, filp=False, rotate=False, shear=False,
                 rotate_3D=False, noise=False, color=False):
        self.crop = crop
        self.light = light
        self.filp = filp
        self.rotate = rotate
        self.shear = shear
        self.rotate_3D = rotate_3D
        self.noise = noise
        self.color = color

    def _crop_scale(self, imgs):

        if random.random() > 0.8:
            scale = 1
        else:
            scale = random.uniform(0.9, 1.2)
        imgs = imgs.copy()
        for idx in range(len(imgs)):
            if len(imgs[idx].shape) == 3:
                imgs[idx] = cv2.resize(imgs[idx], dsize=None, fx=scale, fy=scale)
            else:
                imgs[idx] = cv2.resize(imgs[idx], dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

        h, w = imgs[0].shape[0:2]
        th, tw = self.img_size
        if w == tw and h == th:
            return imgs

        if random.random() > 3 / 8 and np.max(imgs[1]) > 0:
            tl = np.min(np.where(imgs[1] > 0), axis=1) - self.img_size
            tl[tl < 0] = 0
            br = np.max(np.where(imgs[1] > 0), axis=1) - self.img_size
            br[br < 0] = 0
            br[0] = min(br[0], h - th)
            br[1] = min(br[1], w - tw)

            i = random.randint(tl[0], br[0])
            j = random.randint(tl[1], br[1])
        else:
            i = random.randint(0, h - th)
            j = random.randint(0, w - tw)

        # return i, j, th, tw
        for idx in range(len(imgs)):
            if len(imgs[idx].shape) == 3:
                imgs[idx] = imgs[idx][i:i + th, j:j + tw, :]
            else:
                imgs[idx] = imgs[idx][i:i + th, j:j + tw]
        return imgs

    @staticmethod
    def _horizontal_filp(imgs):
        if random.random() > 0.5:
            for i in range(len(imgs)):
                imgs[i] = cv2.flip(imgs[i], 1).copy()
        else:
            for i in range(len(imgs)):
                imgs[i] = cv2.flip(imgs[i], 0).copy()
        return imgs

    @staticmethod
    def _change_light(imgs):
        img = imgs[0].copy()
        flag = random.uniform(0.5, 1.5)  # flag>1为调暗,小于1为调亮
        img = exposure.adjust_gamma(img, flag)
        imgs[0] = img
        return imgs

    @staticmethod
    def _change_color(imgs):
        img = imgs[0].copy()
        combine = [[0, 1, 2], [0, 2, 1],
                   [1, 0, 2], [1, 2, 0],
                   [2, 1, 0], [2, 0, 1]]
        img = img[:, :, combine[random.randint(0, 5)]]
        imgs[0] = img
        return imgs

    @staticmethod
    def _rotate(imgs):
        if random.random() < 0.1:
            angle = -90
        elif random.random() < 0.2:
            angle = 90
        else:
            max_angle = 15
            angle = random.randint(-max_angle, max_angle)
        for i in range(len(imgs)):
            if abs(angle) < 90:
                img = imgs[i]
                w, h = img.shape[:2]
                rotation_matrix = cv2.getRotationMatrix2D((h / 2, w / 2), angle, 1)
                if i == 0:
                    img_rotation = cv2.warpAffine(img, rotation_matrix, (h, w))
                else:
                    img_rotation = cv2.warpAffine(img, rotation_matrix, (h, w), flags=cv2.INTER_NEAREST)
                imgs[i] = img_rotation
            else:
                img = cv2.transpose(imgs[i])
                if angle == 90:
                    img_rotation = cv2.flip(img, 1)
                else:
                    img_rotation = cv2.flip(img, 0)
                imgs[i] = img_rotation
        return imgs

    @staticmethod
    def _deal_img_mask(imgs):
        image = imgs[0].copy()
        img_label = imgs[1].copy()
        img_ignore = imgs[2].copy()
        img_label[img_ignore == 0] = 0
        return [image, img_label, img_ignore]

    @staticmethod
    def _show_img(imgs):
        import matplotlib.pyplot as plt
        plt.figure(1)
        plt.imshow(imgs[0])
        plt.figure(2)
        plt.imshow(imgs[1])
        plt.figure(3)
        plt.imshow(imgs[2])
        plt.show()

    @staticmethod
    def _resize_img(imgs, img_size):
        for i in range(len(imgs)):
            if len(imgs[i].shape) == 3:
                imgs[i] = cv2.resize(imgs[i], tuple(img_size)).copy()
            else:

                imgs[i] = cv2.resize(imgs[i], tuple(img_size), interpolation=cv2.INTER_NEAREST).copy()
        return imgs

    @staticmethod
    def gaussian_noise(image):
        '''
            添加高斯噪声
            mean : 均值
            var : 方差
        '''
        image = image.copy()
        image = cv2.GaussianBlur(image, (5, 5), 1.5)
        return image

    def __call__(self, img_raw, labels_raw, img_size):

        self.img_size = img_size
        img = img_raw.copy()
        labels_raw = copy.deepcopy(labels_raw)
        coors = labels_raw['coor']
        ignores = labels_raw['ignore']
        color = 10
        img_label = np.zeros((img.shape[0:2]), dtype=np.uint8)
        img_ignore = np.ones_like(img_label, dtype=np.uint8)
        for i, ignore in enumerate(ignores):
            color += 1
            coor = np.array(coors[i]).reshape(-1, 2)
            if ignore:
                cv2.drawContours(img_ignore, [coor], -1, 0, thickness=-1)
            else:
                cv2.drawContours(img_label, [coor], -1, color, thickness=-1)
        imgs = [img, img_label, img_ignore]
        if random.random() > 0.5 and self.rotate:
            imgs = self._rotate(imgs)

        if random.random() > 0.5 and self.light:
            imgs = self._change_light(imgs)

        if random.random() > 0.5 and self.color:
            imgs = self._change_color(imgs)

        if random.random() > 0.5 and self.filp:
            imgs = self._horizontal_filp(imgs)

        if self.crop:
            imgs = self._crop_scale(imgs)

        else:
            imgs = self._resize_img(imgs, self.img_size)
        assert imgs[0].shape[0:2] == img_size
        # self._show_img(imgs)
        img = imgs[0]

        if random.random() > 0.5 and self.noise:
            img = self.gaussian_noise(img)

        img_label = imgs[1]
        img_ignore = imgs[2]
        for i, ignore in enumerate(ignores):
            if not ignore:
                mask = np.zeros(img_label.shape, dtype=np.uint8)
                mask[img_label == i + 11] = 1
                if mask.max() == 0:
                    ignores[i] = True
                    continue
                _, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                # rect = cv2.minAreaRect(contours[0])
                # box = cv2.boxPoints(rect)
                # box = np.int0(box).reshape(-1).tolist()
                coors[i] = contours[0].reshape(-1).tolist()
        labels_raw['coor'] = coors
        labels_raw['ignore'] = ignores
        return img, img_ignore, labels_raw


def ImgOrderFormat(img, from_order="HWC", to_order="CHW"):
    if from_order == "HWC" and to_order == "CHW":
        return img.transpose(2, 1, 0)
    elif from_order == "CHW" and to_order == "HWC":
        return img.transpose(1, 2, 0)
    else:
        raise ValueError("unknown order format %s or %s" % (from_order, to_order))


def ReadImage(filename, out_format='numpy', order='HWC', color_format='RGB'):
    """
    :param filename: filepath+filename.jpg/.png/...
    :param out_format: "pillow" | "numpy"(default)
    :param order: "CWH" | "HWC"(default)
    :param color_format: "RGB"(default) | "BGR"
    :return: Pillow image object| numpy array
    """
    with Image.open(filename) as img:
        image = np.array(img)
        if color_format == "RGB":
            pass
        elif color_format == "BGR":
            image = image[:, :, (2, 1, 0)]
        else:
            ValueError("unknown color_format '{}'".format(color_format))

        if order == "CHW":
            image = image.transpose(2, 0, 1)
        elif order == "HWC":
            pass
        else:
            ValueError("Unknown order '{}'".format(order))

        if out_format == "numpy":
            return image
        elif out_format == "pillow":
            img = img.load()
            return img
        else:
            ValueError("Unknown out_format '{}".format(out_format))

        return


def normalize(in_img, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)):
    # should be RGB order
    img = in_img.copy().astype(np.float32)

    img -= np.array([mean[0] * 255.0, mean[1] * 255.0, mean[2] * 255.0], dtype=np.float32)
    img /= np.array([variance[0] * 255.0, variance[1] * 255.0, variance[2] * 255.0], dtype=np.float32)
    return img


def denormalize(in_img, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)):
    # should be RGB order
    img = in_img.copy()
    img *= variance
    img += mean
    img *= 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def gaussian_func(img):
    size = img.max()
    sigma = 0.3 * ((size - 1) * 0.5 - 1) + 0.8
    sigma *= 3
    img2 = np.exp(-0.5*np.power(img-img.max(), 2)/sigma**2)
    img2 *= img.astype(bool)
    return img2


def keep_mask(label, pixel_mask):
    label = np.squeeze(label).reshape(-1, 2)
    label = np.maximum(label, 0)
    cv2.drawContours(pixel_mask, [label], -1, 1, -1)
    pos = ndimage.distance_transform_edt(pixel_mask)
    pos = gaussian_func(pos)
    return pos


def one_hot2dist(pixel_mask):

    res = np.ones_like(pixel_mask)
    posmask = pixel_mask.astype(np.bool)

    if posmask.any():
        negmask = ~posmask
        pos = ((1-pixel_mask)+0.75)*posmask
        neg = ndimage.distance_transform_edt(negmask)
        neg /= (neg.max()+0.001)
        neg = (1 - neg)*negmask
        neg /= (neg.max()*2+0.001)
        neg = (neg + 0.75) * negmask
        res = neg + pos
    return res
