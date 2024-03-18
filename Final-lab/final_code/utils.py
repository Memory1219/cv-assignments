import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

def calculate_f1_score(y_true, y_pred, n_classes=6):
    weighted_f1 = f1_score(y_true.flatten(), y_pred.flatten(), average='weighted', labels=range(n_classes))
    return weighted_f1


def create_subplots(plot_rows, plot_cols, fig_size, image_list, font_size, camp):
    if len(image_list) > plot_rows * plot_cols | len(image_list) == 0:
        raise ValueError('Number of image error')
    fig, sub_figs = plt.subplots(plot_rows, plot_cols, figsize=fig_size)
    for i in range(plot_rows):
        for j in range(plot_cols):
            index = i * plot_cols + j
            if index < len(image_list):
                image_name, image_data = image_list[index]

                # When there is only one subgraph, sub_figs is not an array. Visit directly.
                if plot_rows == 1 and plot_cols == 1:
                    sub_fig = sub_figs
                elif plot_rows == 1 or plot_cols == 1:
                    # When there is only one line or column, sub_figs is a one-dimensional array.
                    sub_fig = sub_figs[max(i, j)]
                else:
                    # Otherwise, sub_figs is a two-dimensional array.
                    sub_fig = sub_figs[i, j]

                temp_img = sub_fig.imshow(image_data, cmap=camp)
                sub_fig.set_title(image_name, fontsize=font_size)
                cbar = fig.colorbar(temp_img, ax=sub_fig, orientation='horizontal')
                # cbar.ax.tick_params(labelsize=font_size)
                sub_fig.axis('off')

    # When the number of subgraphs is less than the number of grids, hide the redundant subgraphs.
    for index in range(len(image_list), plot_rows * plot_cols):
        i = index // plot_cols
        j = index % plot_cols
        if plot_rows == 1 and plot_cols == 1:
            continue  # There is no need to hide when there is only one subgraph.
        elif plot_rows == 1 or plot_cols == 1:
            sub_fig = sub_figs[max(i, j)]
        else:
            sub_fig = sub_figs[i, j]
        sub_fig.axis('off')
    return fig, sub_figs


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
    return abs(int(img[point_1]) - int(img[point_2]))


def region_grow(img, seeds, thresh, mode=8):
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


def get_x_y(image, n):
    im = image
    plt.imshow(im, cmap=plt.get_cmap("gray"))
    pos = plt.ginput(n)
    plt.close()
    return pos  # The resulting pos is a list of tuples containing multiple coordinates.


def calculate_weighted_specificity(y_true, y_pred, n_classes=6):
    """
    Calculate the weighted specificity of multi-category image segmentation results.
    """
    # Flatten
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    # Initialize variables
    sum_specificity = 0
    sum_weights = 0

    # Calculate the confusion matrix
    cm = confusion_matrix(y_true_flat, y_pred_flat, labels=range(n_classes))

    # Calculate specificity for each category
    for i in range(n_classes):
        TN = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
        FP = cm[:, i].sum() - cm[i, i]
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
        weight = cm[:, i].sum()
        sum_specificity += specificity * weight
        sum_weights += weight

    weighted_specificity = sum_specificity / sum_weights if sum_weights > 0 else 0

    return weighted_specificity


def calculate_weighted_sensitivity(y_true, y_pred, n_classes=6):
    """
    Calculate the weighted sensitivity of multi-category image segmentation results.
    """
    # flatten
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    # Initialize variables
    sum_sensitivity = 0
    sum_weights = 0

    # Calculate the confusion matrix
    cm = confusion_matrix(y_true_flat, y_pred_flat, labels=range(n_classes))

    # Calculate sensitivity for each category
    for i in range(n_classes):

        TP = cm[i, i]
        FN = cm[i, :].sum() - TP

        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
        weight = cm[i, :].sum()
        sum_sensitivity += sensitivity * weight
        sum_weights += weight

    # Calculate the weighted sensitivity
    weighted_sensitivity = sum_sensitivity / sum_weights if sum_weights > 0 else 0

    return weighted_sensitivity
