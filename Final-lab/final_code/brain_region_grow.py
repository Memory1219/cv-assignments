import matplotlib.pyplot as plt
from scipy.io import loadmat
import numpy as np
from collections import Counter
from scipy.ndimage import gaussian_filter
import utils


def brain_region_grow(image, label):
    smoothed_slice = gaussian_filter(imgae, sigma=0.8)

    max_val = np.max(smoothed_slice)
    min_val = np.min(smoothed_slice)
    normalization = 255 * (smoothed_slice - min_val) / (max_val - min_val)
    smoothed_slice = np.round(normalization).astype(int)

    seeds = utils.get_x_y(smoothed_slice, n=5)  # Get Initial Seed


    new_seeds = []
    for seed in seeds:
        print(seed)
        # The coordinates selected with the mouse are float type, which needs to be converted to int type.
        # The coordinates selected with the mouse are (W,H), and the picture we read using the function is (row, column),
        # which corresponds to the original picture is (H,W), so we need to change the coordinate position here.
        new_seeds.append((int(seed[1]), int(seed[0])))

    segmented_regions = utils.region_grow(smoothed_slice, new_seeds, thresh=25)
    return segmented_regions


if __name__ == '__main__':
    data = loadmat('Brain.mat')
    T1 = data['T1']
    labels = data['label']

    final_segmentations = []
    f1_scores = []
    for i in range(T1.shape[2]):
        print(f"begin segment image {i}")
        imgae = T1[:, :, i]
        label = labels[:, :, i]
        final_segmentation = brain_region_grow(imgae, label)
        f1 = utils.calculate_f1_score(label, final_segmentation)
        final_segmentations.append(final_segmentation)
        f1_scores.append(f1)

    total_f1 = 0
    for i, score in enumerate(f1_scores, start=1):
        print(f"segmentation[{i}]:{score:.4f}")
        total_f1 += score

    average_f1 = total_f1 / len(f1_scores)
    print(f"average_f1：{average_f1:.4f}")

    # evaluation
    specificity_list = []
    sensitivity_list = []
    for i in range(T1.shape[2]):
        sensitivity_list.append(utils.calculate_weighted_sensitivity(labels[:, :, i], final_segmentations[i]))
        specificity_list.append(utils.calculate_weighted_specificity(labels[:, :, i], final_segmentations[i]))
    print(len(f1_scores))
    for i in range(len(f1_scores)):
        print(
            f"Result[{i + 1}]  F1:{f1_scores[i]:.4f}  sensitivity:{sensitivity_list[i]:.4f}  specificity:{specificity_list[i]:.4f}")


    # display
    plot_rows = 2
    plot_cols = 5
    fig_size = (6 * plot_cols, 6 * plot_rows)
    image_list = []
    image_list.extend([("segmented" + str(i + 1), seg) for i, seg in enumerate(final_segmentations)])
    font_size = 20
    camp = 'jet'
    fig, sub_figs = utils.create_subplots(plot_rows, plot_cols, fig_size, image_list, font_size, camp)
    plt.tight_layout()
    plt.show()