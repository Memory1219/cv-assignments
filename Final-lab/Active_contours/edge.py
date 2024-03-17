import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter

# 加载.mat文件中的数据
data = loadmat('../Brain.mat')
T1 = data['T1']  # T1加权的MRI图像
labels = data['label']  # 真实的标记数据

# 加载图像（这里假设img是你的MRI切片图像，已经转换为灰度图）
# img = cv2.imread('path_to_your_image', cv2.IMREAD_GRAYSCALE)
img = T1[:, :, 0]
img = img.astype(np.uint8)
img = gaussian_filter(img, sigma=0.8)
# 应用Canny边缘检测
edges = cv2.Canny(img, 10, 100)

# 寻找轮廓
contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 根据轮廓面积进行排序，获取面积和对应的轮廓索引
# sorted_contours = sorted([(cv2.contourArea(contour), contour) for contour in contours])
sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

# 创建一个与原图像相同大小的零矩阵，用于绘制分割结果
segmented_img = np.zeros_like(img, dtype=np.uint8).copy()

# 如果找到的轮廓少于6个，则取找到的轮廓数量作为最大标签
num_labels = min(len(sorted_contours), 6)

# 根据轮廓面积从小到大标记（0-5）
# for i, (area, contour) in enumerate(sorted_contours[:num_labels]):
#     # 绘制轮廓：标记值为i，根据面积大小顺序从0开始
#     cv2.drawContours(segmented_img, [contour], -1, i + 1, thickness=cv2.FILLED)
for i, contour in enumerate(sorted_contours[:6]):  # 假设只处理面积最大的6个轮廓
    # 在这里，i + 1是用作标记的整数值，确保其为整型
    # 使用cv2.FILLED常量替换thickness参数来填充轮廓
    cv2.drawContours(segmented_img, [contour], -1, color=int(i + 1), thickness=cv2.FILLED)
# 注意：根据实际情况可能需要调整标记值，确保它们在可视化时有区分

# 保存或显示结果
# cv2.imwrite('segmented_output.png', segmented_img)
# 或 cv2.imshow('Segmented Image', segmented_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
plt.imshow(segmented_img, cmap='jet')
plt.show()
