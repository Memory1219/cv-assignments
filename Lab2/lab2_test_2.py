# Imports
import skimage
import scipy
import time
from matplotlib import pyplot as plt
import numpy as np
from filters import gaussian_filter_3x3, gaussian_filter_5x5
from utils import show_rgb_image, show_binary_image, sample_gaussian, zero_cross


shakey = skimage.io.imread('shakey.jpg')[:,:,0]

img_gaussian_filtered = scipy.signal.convolve2d(shakey, gaussian_filter_3x3, mode='full')<70
show_binary_image(image=img_gaussian_filtered)

img_gaussian_filtered_1 = scipy.signal.convolve2d(shakey, gaussian_filter_5x5, mode='full')<70
show_binary_image(img_gaussian_filtered_1)