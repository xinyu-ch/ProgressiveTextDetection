import argparse
import torch
from dataset import CurveDataset
from torch.utils.data import DataLoader
from models import VGGNet
import matplotlib.pyplot as plt
from ImageProcess import ImageTransform
import cv2
import numpy as np
from postprocess_CURVE import detect_peaks, cal_iou_normal

parser = argparse.ArgumentParser()
parser.add_argument('-bs', dest='batch_size', type=int, default=1)
parser.add_argument('-epoch', dest='epoch', type=int, default=200)
parser.add_argument('-dataset', dest='dataset', nargs='?', type=str, default='CTW1500')
use_cuda = torch.cuda.is_available()


def box_process(image, pred_pixel, boxes, score, num):

    image = image.squeeze(0).cpu().numpy()
    image = ImageTransform.ImgOrderFormat(image, from_order="CHW", to_order="HWC")
    image = ImageTransform.denormalize(image)
    # image = ImageFormat.ImageColorFormat(image, from_color='RGB', to_color='BGR')
    image = image.astype(np.uint8)
    pred_pixel = pred_pixel.squeeze(1).cpu()
    pixel = pred_pixel[num].numpy()

    for b, box in enumerate(boxes):
        box = np.ascontiguousarray(np.int0(box))
        image = np.ascontiguousarray(image)
        cv2.drawContours(image, [box], 0, (255, 0, 0), 2)
        cv2.putText(image, '{:.3f}'.format(score[b]), tuple(box[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), thickness=1)

    plt.figure(2, (20, 10))
    plt.subplot(1, 2, 1)
    plt.title('the number is {}'.format(num+1))
    plt.imshow(pixel, cmap='jet')
    plt.subplot(1, 2, 2)
    plt.imshow(image)
    plt.show()


def test(model, dataloader, args):

    print('Start testing')
    with torch.no_grad():
        numGlobalCareGt = 0
        matchedSum = 0
        numGlobalCareDet = 0
        for i, sample in enumerate(dataloader):
            img = sample['image']
            img = img.cuda()
            pixel_mask = sample['pixel_mask'].cuda()
            pred_pixel  = model(img)
            pred_pixel = pred_pixel.squeeze(1)
            for j in range(pixel_mask.size(0)):
                my_labels, score = detect_peaks(pred_pixel[j].cpu().numpy())
                detMatched, numGtCare, numDetCare = cal_iou_normal(my_labels, sample['label'], 0.5)
                matchedSum += detMatched
                numGlobalCareGt += numGtCare
                numGlobalCareDet += numDetCare
                methodRecall = 0 if numGlobalCareGt == 0 else float(matchedSum) / numGlobalCareGt
                methodPrecision = 0 if numGlobalCareDet == 0 else float(matchedSum) / numGlobalCareDet
                methodHmean = 0 if methodRecall + methodPrecision == 0 else 2 * methodRecall * methodPrecision / (
                        methodRecall + methodPrecision)
                print("i: {}, current TP: {}, FP: {}, FN: {}, precision: {}, recall: {}, human: {}"
                      .format(i, detMatched, numDetCare - detMatched, numGtCare - detMatched, methodPrecision,
                              methodRecall, methodHmean))


def run(args):
    img_size = (512, 512)
    batch_size = args.batch_size
    dataset = CurveDataset(img_size, train=False)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = VGGNet()

    if args.dataset == "CTW1500":
        save_model = "./patchs/ctw1500_model/"

    model_static = torch.load(save_model + 'vgg16-{}.mdl'.format(args.epoch))
    model.load_state_dict(model_static)
    if use_cuda:
        model.cuda()

    test(model, dataloader, args)


if __name__ == '__main__':
    args = parser.parse_args()
    run(args)


