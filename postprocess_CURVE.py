import numpy as np
import cv2
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
from skimage.morphology import watershed
from scipy import ndimage
import Polygon as plg
import imutils
import matplotlib.pyplot as plt


def cal_iou_normal(my_labels, gt_labels, iou_thres=0.5):
    def polygon_from_points(points):
        """
        Returns a Polygon object to use with the Polygon2 class from a list of 8 points: x1,y1,x2,y2,x3,y3,x4,y4
        """
        pointMat = points.reshape([-1, 2])
        return plg.Polygon(pointMat)

    def get_union(pD, pG):
        areaA = pD.area()
        areaB = pG.area()
        return areaA + areaB - get_intersection(pD, pG)

    def get_intersection_over_union(pD, pG):
        try:
            return get_intersection(pD, pG) / get_union(pD, pG)
        except:
            return 0

    def get_intersection(pD, pG):
        pInt = pD & pG
        if len(pInt) == 0:
            return 0
        return pInt.area()

    detMatched = 0
    gtPols = []
    detPols = []

    gtPolPoints = []
    detPolPoints = []

    # Array of Ground Truth Polygons' keys marked as don't Care
    gtDontCarePolsNum = []
    # Array of Detected Polygons' matched with a don't Care GT
    detDontCarePolsNum = []

    pairs = []
    detMatchedNums = []

    evaluationLog = ""

    pointsList = list(map(np.array, gt_labels['coor']))
    dontCares = np.array(list((map(np.array, gt_labels['ignore'])))).reshape(-1)

    for n in range(len(pointsList)):
        points = pointsList[n]
        dontCare = dontCares[n]
        gtPol = polygon_from_points(points)
        gtPols.append(gtPol)
        gtPolPoints.append(points)
        if dontCare:
            gtDontCarePolsNum.append(len(gtPols) - 1)

    pointsList = my_labels

    for n in range(len(pointsList)):
        points = np.array(pointsList[n]).reshape(-1)
        detPol = polygon_from_points(points)
        detPols.append(detPol)
        detPolPoints.append(points)
        if len(gtDontCarePolsNum) > 0:
            for dontCarePol in gtDontCarePolsNum:
                dontCarePol = gtPols[dontCarePol]
                intersected_area = get_intersection(dontCarePol, detPol)
                pdDimensions = detPol.area()
                precision = 0 if pdDimensions == 0 else intersected_area / pdDimensions
                if precision > 0.5:
                    detDontCarePolsNum.append(len(detPols) - 1)
                    break

    if len(gtPols) > 0 and len(detPols) > 0:
        # Calculate IoU and precision matrixs
        outputShape = [len(gtPols), len(detPols)]
        iouMat = np.empty(outputShape)
        gtRectMat = np.zeros(len(gtPols), np.int8)
        detRectMat = np.zeros(len(detPols), np.int8)
        for gtNum in range(len(gtPols)):
            for detNum in range(len(detPols)):
                pG = gtPols[gtNum]
                pD = detPols[detNum]
                iouMat[gtNum, detNum] = get_intersection_over_union(pD, pG)

        for gtNum in range(len(gtPols)):
            for detNum in range(len(detPols)):
                if gtRectMat[gtNum] == 0 and detRectMat[detNum] == 0 \
                        and gtNum not in gtDontCarePolsNum and detNum not in detDontCarePolsNum:
                    if iouMat[gtNum, detNum] > iou_thres:
                        gtRectMat[gtNum] = 1
                        detRectMat[detNum] = 1
                        detMatched += 1
                        detMatchedNums.append(detNum)
        for i in range(len(gtRectMat)):
            if gtRectMat[i] == 0 and dontCares[i] == 0:
                print(gt_labels['content'][i])
                pass
    numGtCare = (len(gtPols) - len(gtDontCarePolsNum))
    numDetCare = (len(detPols) - len(detDontCarePolsNum))
    return detMatched, numGtCare, numDetCare


def detect_peaks(image, text_thresh=0.45, low_text=0.1):
    """
    Takes an image and detect the peaks usingthe local maximum filter.
    Returns a boolean mask of the peaks (i.e. 1 when
    the pixel's value is the neighborhood maximum, 0 otherwise)
    """
    img_score = image.copy()
    image = image.copy()
    image[image <= low_text] = 0
    image[image > text_thresh] = 1
    image = (image * 255).astype(np.uint8)

    thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY)[1]

    # 计算从每个二元像素到最近零像素的精确欧几里得距离，然后在距离图中找到峰值
    D = ndimage.distance_transform_edt(thresh)

    # define an 8-connected neighborhood
    neighborhood = generate_binary_structure(2, 2)
    local_max = maximum_filter(image, footprint=neighborhood) == image
    background = (image == 0)
    eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)
    detected_peaks = local_max ^ eroded_background

    # 利用8连通性对局部峰进行连通分量分析，然后应用分水岭算法
    markers = ndimage.label(detected_peaks, structure=np.ones((3, 3)))[0]
    labels = watershed(-D, markers, mask=thresh)
    # print("[INFO] {} unique segments found".format(len(np.unique(labels)) - 1))

    boxes = []
    boxes_score = []
    # 循环显示标签
    for label in np.unique(labels):
        score_mask = np.zeros(image.shape, dtype=np.uint8)

        # 如果该标签为0，则表示其为背景，直接忽略
        if label == 0:
            continue

        # 为标签区域分配内存并将在mask上绘制结果
        mask = np.zeros(image.shape, dtype="uint8")
        mask[labels == label] = 255

        # 在mask上检测轮廓并获得最大的一个轮廓
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(c) < 20:
            continue
        cv2.drawContours(score_mask, cnts, -1, 255, thickness=-1)
        # cv2.imshow('2', score_mask)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        points = img_score[np.nonzero(score_mask)]
        score = np.sum(points) / points.shape[0]
        if max(points) < 0.75 or score < 0.45:
            continue
        # print(score)
        bbox = np.int0(cnts[0].reshape(-1, 2))
        cv2.drawContours(image, [bbox], -1, 127, thickness=1)
        boxes.append(bbox)
        boxes_score.append(score)

    # plt.figure(1)
    # plt.subplot(2, 2, 1)
    # plt.imshow(image, cmap='jet')
    # plt.subplot(2, 2, 2)
    # plt.imshow(image1, cmap='jet')
    # plt.subplot(2, 2, 3)
    # plt.imshow(img_score, cmap='jet')
    # plt.subplot(2, 2, 4)
    # plt.imshow(image3, cmap='jet')
    # plt.show()
    return boxes, boxes_score
