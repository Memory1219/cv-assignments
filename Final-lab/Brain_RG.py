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

# Due to the interactive nature of seed selection, here we'll define an arbitrary set of seeds for demonstration
# seed_points = [utils.Point(99, 100), utils.Point(150, 150), utils.Point(200, 200), utils.Point(250, 250), utils.Point(300, 300), utils.Point(150, 200)]  # Example seed points
# seed_points = utils.find_seeds_from_image(smoothed_slice)

# smoothed_slice = utils.normalize_and_detect_edges(smoothed_slice, 10, 50)


max_val = np.max(smoothed_slice)
min_val = np.min(smoothed_slice)
T1_normalized = 60 * (smoothed_slice - min_val) / (max_val - min_val)
smoothed_slice = np.round(T1_normalized).astype(int)

# seeds=utils.get_x_y(smoothed_slice, n=6) #获取初始种子
#
# print("选取的初始点为：")
# new_seeds=[]
# for seed in seeds:
#     print(seed)
#     #下面是需要注意的一点
#     #第一： 用鼠标选取的坐标为float类型，需要转为int型
#     #第二：用鼠标选取的坐标为（W,H），而我们使用函数读取到的图片是（行，列），而这对应到原图是（H,W），所以这里需要调换一下坐标位置，这是很多人容易忽略的一点
#     new_seeds.append((int(seed[1]), int(seed[0])))#


# print(new_seeds)
# Apply region growing on the smoothed slice
seed_points = []
# for x, y in new_seeds:
#     seed_points.append((x, y))
#     seed_points.append(utils.Point(x,y))
seed_points = [(np.random.randint(0, smoothed_slice.shape[0]), np.random.randint(0, smoothed_slice.shape[1])) for _ in range(20)]
newSeeds = []
for x, y in seed_points:
    newSeeds.append(utils.Point(x, y))
segmented_regions = utils.region_grow(smoothed_slice, newSeeds, thresh = 2)
# segmented_regions = utils.merge_areas(segmented_regions)
# Display the result
plt.figure(figsize=(6, 6))
plt.imshow(segmented_regions, cmap='jet')
plt.title('Region Growing Segmented Slice')
plt.axis('off')
plt.show()






