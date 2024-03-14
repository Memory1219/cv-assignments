import cv2
import numpy as np
from scipy.io import loadmat
import sys
import matplotlib.pyplot as plt
from collections import Counter
from skimage.filters import threshold_multiotsu
from scipy.ndimage import gaussian_filter

data = loadmat('Brain.mat')
T1 = data['T1']
labels = data['label']
print(T1.shape)
print(labels.shape)
label = labels[:, :, 0]

# 选择一个代表性的切片进行分割
image_slice = T1[:, :, 0]
image_slice = gaussian_filter(image_slice, sigma=0.8)
# 使用multi-otsu计算多个阈值以分割图像为6个区域
thresholds = threshold_multiotsu(image_slice, classes=6)
regions = np.digitize(image_slice, bins=thresholds)

# 初始化最终分割图像为全零矩阵
final_segmented = np.zeros_like(image_slice)
final_segmented_2 = np.zeros_like(image_slice)
image_data = np.array(label)
regions_data = np.array(regions)

regions_pixel_values = regions_data.flatten().astype(int)
pixel_values = image_data.flatten().astype(int)

counts_segment = Counter(regions_pixel_values)
sort_counts_segment = sorted(counts_segment.items(), key=lambda x: x[1])
print(sort_counts_segment)

counts = Counter(pixel_values)
sorted_count = sorted(counts.items(), key=lambda x: x[1])
print(sorted_count)
results = [(index, element[0]) for index, element in enumerate(sorted_count)]
print(results)
for i, value in results:
    print(i)
    print(value)
    final_segmented[regions == sort_counts_segment[i][0]] = value
final_segmented[regions == sort_counts_segment[3][0]] = 4
final_segmented[regions == sort_counts_segment[4][0]] = 5
class_values = [0, 4, 5, 1, 3, 2]
for i, value in enumerate(class_values):
    final_segmented_2[regions == i] = value

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(image_slice, cmap='gray')
axes[0].set_title('Original Slice')
t1_image = axes[1].imshow(final_segmented, cmap='jet')
axes[1].set_title('Colored Segmentation')
fig.colorbar(t1_image, ax=axes[1], orientation='horizontal')
t1_image_2 = axes[2].imshow(final_segmented_2, cmap='jet')
axes[2].set_title('Colored Segmentation2')
fig.colorbar(t1_image_2, ax=axes[2], orientation='horizontal')
for ax in axes:
    ax.axis('off')
plt.tight_layout()
plt.show()
