import cv2
import numpy as np
import matplotlib.pyplot as plt
from  PIL import Image
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score


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


def select_connects(mode):
    connects = []
    if mode == 8:
        for x in [1, 0, -1]:
            for y in [1, 0, -1]:
                if x == 0 and y == 0:
                    continue
                connects.append((x, y))
    if mode == 4:
        for x in [1, 0, -1]:
            for y in [1, 0, -1]:
                if abs(x + y) == 1:
                    connects.append((x, y))
    return connects


def get_gray_diff(img, point_1, point_2):
    return abs(int(img(point_1)) - int(img(point_2)))


def region_grow(img, seeds, thresh, mode=1):
    height, weight = img.shape
    seed_mark = np.zeros_like(img)
    label = 1  # Start label at 0
    connects = select_connects(mode)

    for seed in seeds:
        # Check if this seed's location is already processed to avoid overlapping regions
        if seed_mark[seed] != 0:
            continue
        mark_list = [seed]

        while len(mark_list) > 0:
            current_point = mark_list.pop(0)
            seed_mark[current_point] = label
            for i in range(8):
                tmp_x = current_point[0] + connects[i][0]
                tmp_y = current_point[1] + connects[i][1]
                if tmp_x < 0 or tmp_y < 0 or tmp_x >= height or tmp_y >= weight:
                    continue
                gray_diff = get_gray_diff(img, seed, (tmp_x, tmp_y))
                if gray_diff < thresh and seed_mark[tmp_x, tmp_y] == 0:
                    seed_mark[tmp_x, tmp_y] = label
                    mark_list.append((tmp_x, tmp_y))
        label += 1  # Increment label for a new seed

    return seed_mark


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


def evaluate_f1(y_true, y_pred):
    """
    评估分割结果的F1分数。

    参数:
    - y_true: 真实标签的numpy数组。
    - y_pred: 预测结果的numpy数组。

    返回:
    - 无，直接打印每个类别的precision, recall, F1分数以及它们的加权平均值。
    """
    # 计算并打印分类报告
    report = classification_report(y_true, y_pred, labels=[0, 1, 2, 3, 4, 5],
                                   target_names=['区域0', '区域1', '区域2', '区域3', '区域4', '区域5'],
                                   output_dict=True)

    # 打印每个类别的评估结果
    for key, value in report.items():
        if key in ['区域0', '区域1', '区域2', '区域3', '区域4', '区域5']:
            print(f"{key}:")
            print(f"    Precision: {value['precision']:.3f}")
            print(f"    Recall: {value['recall']:.3f}")
            print(f"    F1 Score: {value['f1-score']:.3f}")
            print("")

    # 打印加权平均F1分数
    print(f"加权平均F1分数: {report['weighted avg']['f1-score']:.3f}")


def calculate_f1_score(y_true, y_pred, n_classes=6):

    weighted_f1 = f1_score(y_true.flatten(), y_pred.flatten(), average='weighted', labels=range(n_classes))

    return weighted_f1
