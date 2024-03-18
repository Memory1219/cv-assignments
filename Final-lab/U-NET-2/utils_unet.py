import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

def deal_pred_img(pred_img):
    # Suppose pred is the output of the network, and the shape is [1, 6, 512, 512]
    # Convert the network output to a numpy array and remove the batch dimension, resulting in a shape of [6, 512, 512]
    pred = pred_img.data.cpu().numpy()[0]

    # For each pixel, find the category index with the highest probability
    pred_class_indices = np.argmax(pred, axis=0)

    # To save the result image, you need to convert the category index to uint8 type.
    image = pred_class_indices.astype(np.uint8)

    return image


def calculate_accuracy(output, label):
    correct_pixels = 0
    total_pixels = 0
    # output = deal_pred_img(output)
    _, predicted = torch.max(output.data, 1)
    correct_pixels += (predicted == label).sum().item()
    total_pixels += label.numel()
    return correct_pixels / total_pixels * 100


def save_model(model, save_path):
    torch.save(model.state_dict(), save_path)


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

    # Hide redundant subgraphs when the number of subgraphs is less than the number of grids
    for index in range(len(image_list), plot_rows * plot_cols):
        i = index // plot_cols
        j = index % plot_cols
        if plot_rows == 1 and plot_cols == 1:
            continue  # No need to hide when there is only one subgraph
        elif plot_rows == 1 or plot_cols == 1:
            sub_fig = sub_figs[max(i, j)]
        else:
            sub_fig = sub_figs[i, j]
        sub_fig.axis('off')
    return fig, sub_figs


def calculate_weighted_specificity(y_true, y_pred, n_classes=6):

    # flatten
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

    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    sum_sensitivity = 0
    sum_weights = 0

    cm = confusion_matrix(y_true_flat, y_pred_flat, labels=range(n_classes))

    for i in range(n_classes):
        TP = cm[i, i]
        FN = cm[i, :].sum() - TP
        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
        weight = cm[i, :].sum()
        sum_sensitivity += sensitivity * weight
        sum_weights += weight

    weighted_sensitivity = sum_sensitivity / sum_weights if sum_weights > 0 else 0

    return weighted_sensitivity