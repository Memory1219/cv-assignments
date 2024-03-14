import matplotlib.pyplot as plt
from scipy.io import loadmat
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter

data = loadmat('Brain.mat')
T1 = data['T1']
labels = data['label']
print(T1.shape)
print(labels.shape)
label = labels[:, :, 0]
image = T1[:, :, 0]
smoothed_slice = image
reshaped_smoothed_slice = smoothed_slice.reshape(-1, 1)
try:
    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=6, random_state=0).fit(reshaped_smoothed_slice)
    segmented_kmeans = kmeans.labels_.reshape(smoothed_slice.shape)
except Exception as e:
    print(f"Error occurred: {e}")



regions_pixel_values = segmented_kmeans.flatten().astype(int)
pixel_values = label.flatten().astype(int)

counts_segment = Counter(regions_pixel_values)
sort_counts_segment = sorted(counts_segment.items(), key=lambda x: x[1])
print(sort_counts_segment)

counts = Counter(pixel_values)
sorted_count = sorted(counts.items(), key=lambda x: x[1])
print(sorted_count)
results = [(index, element[0]) for index, element in enumerate(sorted_count)]
print(results)

final_segmented = np.zeros_like(smoothed_slice)
for i, value in results:
    print(i)
    print(value)
    final_segmented[segmented_kmeans == sort_counts_segment[i][0]] = value
# final_segmented[segmented_kmeans == sort_counts_segment[3][0]] = 4
# final_segmented[segmented_kmeans == sort_counts_segment[4][0]] = 5






fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].imshow(image, cmap='gray')
ax[0].set_title('Original Slice')
ax[0].axis('off')

ax[1].imshow(smoothed_slice, cmap='gray')
ax[1].set_title('Smoothed Slice')
ax[1].axis('off')

ax[2].imshow(final_segmented, cmap='jet')
ax[2].set_title('KMeans Segmented Slice')
ax[2].axis('off')

plt.show()