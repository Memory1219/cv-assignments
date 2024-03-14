import matplotlib.pyplot as plt
from scipy.io import loadmat
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter
from scipy.ndimage import gaussian_filter
import utils

data = loadmat('Brain.mat')
T1 = data['T1']
labels = data['label']
imgae = T1[:, :, 0]
smoothed_slice = gaussian_filter(imgae, sigma=1)


max_val = np.max(smoothed_slice)
min_val = np.min(smoothed_slice)
T1_normalized = 255 * (smoothed_slice - min_val) / (max_val - min_val)
smoothed_slice = np.round(T1_normalized).astype(int)


seeds=utils.get_x_y(smoothed_slice, n=6) #获取初始种子

print("选取的初始点为：")
print(seeds)
new_seeds=[]
new_seeds2 = []
for seed in seeds:
    print(seed)
    #下面是需要注意的一点
    #第一： 用鼠标选取的坐标为float类型，需要转为int型
    #第二：用鼠标选取的坐标为（W,H），而我们使用函数读取到的图片是（行，列），而这对应到原图是（H,W），所以这里需要调换一下坐标位置，这是很多人容易忽略的一点
    new_seeds.append((int(seed[1]), int(seed[0])))
    # new_seeds2.append(utils.Point(int(seed[1]), int(seed[0])))


# print(new_seeds)
# Apply region growing on the smoothed slice
# seed_points = []
# for x, y in new_seeds:
#     seed_points.append((x, y))
#     seed_points.append(utils.Point(x,y))
# segmented_regions = utils.region_growing(smoothed_slice, new_seeds, thresh=25)
# segmented_regions = utils.merge_areas(segmented_regions)
# Display the result

segmented_regions2 = utils.region_growing_8(smoothed_slice, new_seeds, threshold=25)
segmented_regions = utils.region_growing(smoothed_slice, new_seeds, thresh=25)

# plt.figure(figsize=(6, 6))
# plt.imshow(segmented_regions, cmap='jet')
# plt.title('Region Growing Segmented Slice')
# plt.axis('off')
# plt.show()
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
t1_image2 = axes[0].imshow(segmented_regions, cmap='jet')
axes[0].set_title('Original Slice')
fig.colorbar(t1_image2, ax=axes[0], orientation='horizontal')
t1_image = axes[1].imshow(segmented_regions2, cmap='jet')
axes[1].set_title('Colored Segmentation')
fig.colorbar(t1_image, ax=axes[1], orientation='horizontal')
for ax in axes:
    ax.axis('off')
plt.tight_layout()
plt.show()