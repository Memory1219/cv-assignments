import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy.io import loadmat

# 加载.mat文件中的数据
data = loadmat('../Brain.mat')
T1 = data['T1']  # T1加权的MRI图像
labels = data['label']  # 真实的标记数据


# 使用OpenCV找到每个切片中的轮廓并根据面积大小进行排序和标记
def segment_and_label_slice(slice_img):
    # 转换为灰度图像
    # gray = cv2.cvtColor(slice_img, cv2.COLOR_BGR2GRAY)
    # 如果图像是32位浮点型，将其转换为8位整型
    slice_img = slice_img.copy()
    if slice_img.dtype == np.float32:
        slice_img = (slice_img * 255).astype(np.uint8)

    # 转换为灰度图像
    # gray = cv2.cvtColor(slice_img, cv2.COLOR_BGR2GRAY).astype(np.uint8) if len(slice_img.shape) == 3 else slice_img
    gray = (255 * slice_img).astype(np.uint8)
    # 应用阈值方法找到轮廓
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 找到轮廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 按轮廓面积排序
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # 创建一个空的标记图像
    label_image = np.zeros_like(gray, dtype=np.uint8)

    # 按面积大小填充标记
    for label_number, contour in enumerate(contours):
        cv2.drawContours(label_image, [contour], -1, label_number, -1)

    return label_image


# 对每个切片应用分割和标记算法
segmented_slices = [segment_and_label_slice(T1[:, :, i]) for i in range(T1.shape[2])]
plt.imshow(segmented_slices[1], cmap='jet')
plt.show()
# 对标记的轮廓进行排序和重新标记，以确保面积最大的轮廓被标记为0，面积最小的轮廓被标记为5
for segmented_slice in segmented_slices:
    # 转换为灰度图像
    # gray = cv2.cvtColor(segmented_slice, cv2.COLOR_BGR2GRAY)
    gray = segmented_slice
    # 找到轮廓并按面积排序
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # 创建一个空的标记图像
    sorted_label_image = np.zeros_like(gray)

    # 按面积从大到小标记轮廓
    for label_number, contour in enumerate(contours):
        cv2.drawContours(sorted_label_image, [contour], -1, 5 - label_number, -1)

    # 替换原来的分割图像
    segmented_slice = sorted_label_image

