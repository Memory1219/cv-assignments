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



def calculate_weighted_specificity(y_true, y_pred, n_classes=6):
    """
    Calculate the weighted specificity of multi-category image segmentation results.
    """
    # Fold
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    # Initialize variables
    sum_specificity = 0
    sum_weights = 0

    # Calculate the confusion matrix
    cm = confusion_matrix(y_true_flat, y_pred_flat, labels=range(n_classes))

    # Calculate specificity for each category
    for i in range(n_classes):
        # The true and negative example (TN) is the sum of elements except for the current category.
        TN = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
        # The false positive example (FP) is the sum of the current column minus the elements on the diagonal (TP)
        FP = cm[:, i].sum() - cm[i, i]

        # Calculate the specificity of the current category
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0

        # Accumulated weighted specificity
        weight = cm[:, i].sum()
        sum_specificity += specificity * weight
        sum_weights += weight

    # Calculate weighted specificity
    weighted_specificity = sum_specificity / sum_weights if sum_weights > 0 else 0

    return weighted_specificity


def calculate_weighted_sensitivity(y_true, y_pred, n_classes=6):
    """
    Calculate the weighted sensitivity of multi-category image segmentation results.
    """
    # Flatten
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    # 初始化变量
    sum_sensitivity = 0
    sum_weights = 0

    cm = confusion_matrix(y_true_flat, y_pred_flat, labels=range(n_classes))

    # Calculate sensitivity for each category
    for i in range(n_classes):
        # The true example (TP) is an element on the diagonal.
        TP = cm[i, i]
        # The false negative case (FN) is the sum of the current line minus the elements on the diagonal (TP).
        FN = cm[i, :].sum() - TP

        # Calculate the sensitivity of the current category
        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0

        # Cumulative weighted sensitivity
        weight = cm[i, :].sum()
        sum_sensitivity += sensitivity * weight
        sum_weights += weight

    weighted_sensitivity = sum_sensitivity / sum_weights if sum_weights > 0 else 0

    return weighted_sensitivity
