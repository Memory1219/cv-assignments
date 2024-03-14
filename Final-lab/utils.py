import cv2
import numpy as np
import matplotlib.pyplot as plt
from  PIL import Image

def find_seeds_from_image(image, num_seeds=6, max_pixel_value=50000):
    # 确保图像为灰度
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # 调整直方图计算的bins和范围以匹配图像的像素强度范围
    hist = cv2.calcHist([gray], [0], None, [max_pixel_value//256], [0, max_pixel_value])
    peaks = np.argsort(hist[:, 0])[-num_seeds:]

    seeds = []
    for peak in peaks:
        # 由于我们调整了直方图的范围和bins，需要相应调整峰值到实际像素值的映射
        actual_peak = (peak * max_pixel_value / (max_pixel_value//256))
        mask = (gray == actual_peak)
        y, x = np.where(mask)

        # 计算质心作为种子点
        if len(x) > 0 and len(y) > 0:
            centroid = (int(np.mean(x)), int(np.mean(y)))
            seeds.append(centroid)

    return seeds
# def regionGrow(img, seeds, thresh, p=1):
#
#     height, weight = img.shape
#     seedMark = np.zeros(img.shape)
#     seedList = []
#     for seed in seeds:
#         seedList.append(seed)
#     label = 1
#     connects = selectConnects(p)
#     while (len(seedList) > 0):
#         currentPoint = seedList.pop(0)
#
#         seedMark[currentPoint.x, currentPoint.y] = label
#         for i in range(8):
#             tmpX = currentPoint.x + connects[i].x
#             tmpY = currentPoint.y + connects[i].y
#             if tmpX < 0 or tmpY < 0 or tmpX >= height or tmpY >= weight:
#                 continue
#             grayDiff = getGrayDiff(img, currentPoint, Point(tmpX, tmpY))
#             if grayDiff < thresh and seedMark[tmpX, tmpY] == 0:
#                 seedMark[tmpX, tmpY] = label
#                 seedList.append(Point(tmpX, tmpY))
#     return seedMark
# def regionGrow(img, seeds, thresh, p=1):
#     height, weight = img.shape
#     seedMark = np.zeros_like(img)
#     label = 3  # Start label at 0
#     connects = selectConnects(p)
#
#     for seed in seeds:
#         # Check if this seed's location is already processed to avoid overlapping regions
#         if seedMark[seed.x, seed.y] != 0:
#             continue
#         seedList = [seed]
#
#         while len(seedList) > 0:
#             currentPoint = seedList.pop(0)
#             seedMark[currentPoint.x, currentPoint.y] = label
#             for i in range(8):
#                 tmpX = currentPoint.x + connects[i].x
#                 tmpY = currentPoint.y + connects[i].y
#                 if tmpX < 0 or tmpY < 0 or tmpX >= height or tmpY >= weight:
#                     continue
#                 grayDiff = getGrayDiff(img, currentPoint, Point(tmpX, tmpY))
#                 if grayDiff < thresh and seedMark[tmpX, tmpY] == 0:
#                     seedMark[tmpX, tmpY] = label
#                     seedList.append(Point(tmpX, tmpY))
#         label += 1  # Increment label for a new seed
#
#     return seedMark

def region_growing_8(image, seeds, threshold = 2):
    label = 1
    segment_image = np.zeros_like(image)
    for seed in seeds:
        if segment_image[seed] != 0:
            continue
        mark_list = [seed]
        while mark_list:
            mark_point = mark_list.pop(0)
            for x in [-1, 0, 1]:
                for y in [-1, 0, 1]:
                    if x == 0 and y == 0:
                        continue
                    tmp_x, tmp_y = mark_point[0] + x, mark_point[1] + y
                    if 0 <= tmp_x < image.shape[0] and 0 <= tmp_y < image.shape[1]:
                        if abs(image[tmp_x, tmp_y] - image[seed]) <= threshold and segment_image[tmp_x, tmp_y] == 0:
                            segment_image[tmp_x, tmp_y] = label
                            mark_list.append((tmp_x, tmp_y))
        label += 1
    return segment_image

# def region_growing_8(img, seeds, thresh=2):
#     segmented = np.zeros_like(img)
#     label = 1
#     for seed in seeds:
#         if segmented[seed] != 0:
#             continue
#         points_to_check = [seed]
#         while points_to_check:
#             point = points_to_check.pop(0)
#             for dx in [-1, 0, 1]:
#                 for dy in [-1, 0, 1]:
#                     if dx == 0 and dy == 0:
#                         continue
#                     nx, ny = point[0] + dx, point[1] + dy
#                     if 0 <= nx < img.shape[0] and 0 <= ny < img.shape[1]:
#                         if segmented[nx, ny] == 0 and abs(img[nx, ny] - img[seed]) <= thresh:
#                             segmented[nx, ny] = label
#                             points_to_check.append((nx, ny))
#         label += 1
#     return segmented


def region_growing(img, seeds, thresh=2):
    segmented = np.zeros_like(img)
    label = 1
    for seed in seeds:
        if segmented[seed] != 0:
            continue
        points_to_check = [seed]
        while points_to_check:
            point = points_to_check.pop(0)
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = point[0] + dx, point[1] + dy
                    if 0 <= nx < img.shape[0] and 0 <= ny < img.shape[1]:
                        if segmented[nx, ny] == 0 and abs(img[nx, ny] - img[seed]) <= thresh:
                            segmented[nx, ny] = label
                            points_to_check.append((nx, ny))
        label += 1
    return segmented

def calculate_area(segmented):
    unique_labels, counts = np.unique(segmented, return_counts=True)
    return dict(zip(unique_labels, counts))

# 定义合并区域的函数，以减少到目标标签数量
def merge_areas(segmented, target_labels=6):
    while True:
        areas = calculate_area(segmented)
        if 0 in areas:
            del areas[0]
        if len(areas) <= target_labels:
            break
        labels_sorted_by_area = sorted(areas, key=areas.get)
        to_merge = zip(labels_sorted_by_area[:len(labels_sorted_by_area)//2], reversed(labels_sorted_by_area[len(labels_sorted_by_area)//2:]))
        for small, large in to_merge:
            segmented[segmented == small] = large
            break
    return segmented


def get_x_y(image, n):
    im = image
    plt.imshow(im, cmap=plt.get_cmap("gray"))
    pos = plt.ginput(n)
    plt.close()
    return pos  # 得到的pos是列表中包含多个坐标元组

def edgeDetection(image):
    # 使用Canny算法进行边缘检测
    edges = cv2.Canny(image, 100, 200)
    return edges


def normalize_and_detect_edges(img, lower_threshold, upper_threshold):
    # 将图像强度范围从0-50000归一化到0-255
    img_normalized = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    img_normalized = img_normalized.astype(np.uint8)  # 转换数据类型为uint8

    # 应用Canny边缘检测
    edges = cv2.Canny(image=img_normalized, threshold1=lower_threshold, threshold2=upper_threshold)

    return edges