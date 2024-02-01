# Imports
import skimage
import scipy
import time
from matplotlib import pyplot as plt
import numpy as np
from filters import gaussian_filter_3x3, gaussian_filter_5x5
from utils import show_rgb_image, show_binary_image, sample_gaussian, zero_cross

# Complete Task 1 here
shakey = skimage.io.imread("shakey.jpg")[:,:,0]

# plt.figure(figsize=(15,20))
# plt.imshow(shakey, cmap="gray")
# plt.title("shakey image")
# plt.axis("off")
# plt.show()

print("shakey image shape: ", shakey.shape)
print("shakey image: ", shakey)

shakey_filtered_3 = scipy.signal.convolve2d(shakey, gaussian_filter_3x3)

shakey_filtered_5 = scipy.signal.convolve2d(shakey, gaussian_filter_5x5, mode='full')


threshold = 80
plot_rows = 1
plot_cols = 3
# Create a figure with some sub-figures
fig, sub_figs = plt.subplots(plot_rows, plot_cols, figsize=(15*plot_cols, 20))
sub1 = sub_figs[0]
sub2 = sub_figs[1]
sub3 = sub_figs[2]
sub1.imshow(shakey > threshold, cmap=plt.cm.gray)
sub1.set_title("shakey image", fontsize=50)
sub2.imshow(shakey_filtered_3 > threshold, cmap="gray")
sub2.set_title("gaussian_filter_3x3", fontsize=50)
sub3.imshow(shakey_filtered_5 > threshold, cmap="gray")
sub3.set_title("gaussian_filter_5x5", fontsize=50)
for i in range(plot_cols):
    sub_figs[i].axis('off')
plt.show()