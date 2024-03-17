import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score


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

                # 当只有一个子图时，sub_figs不是数组，直接访问imshow
                if plot_rows == 1 and plot_cols == 1:
                    sub_fig = sub_figs
                elif plot_rows == 1 or plot_cols == 1:
                    # 当只有一行或一列时，sub_figs是一维数组
                    sub_fig = sub_figs[max(i, j)]
                else:
                    # 否则，sub_figs是二维数组
                    sub_fig = sub_figs[i, j]

                temp_img = sub_fig.imshow(image_data, cmap=camp)
                sub_fig.set_title(image_name, fontsize=font_size)
                cbar = fig.colorbar(temp_img, ax=sub_fig, orientation='horizontal')
                # cbar.ax.tick_params(labelsize=font_size)
                sub_fig.axis('off')

    # 当子图数量小于网格数量时，隐藏多余的子图
    for index in range(len(image_list), plot_rows * plot_cols):
        i = index // plot_cols
        j = index % plot_cols
        if plot_rows == 1 and plot_cols == 1:
            continue  # 只有一个子图时不需要隐藏
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
    return pos  # 得到的pos是列表中包含多个坐标元组
