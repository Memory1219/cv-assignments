import matplotlib.pyplot as plt
from scipy.io import loadmat
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter
import utils
from scipy.ndimage import gaussian_filter


def kmeans_segment(image, label):
    # smoothed_slice = gaussian_filter(image, sigma=0.8)
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
    # print(sort_counts_segment)

    counts = Counter(pixel_values)
    sorted_count = sorted(counts.items(), key=lambda x: x[1])
    # print(sorted_count)
    results = [(index, element[0]) for index, element in enumerate(sorted_count)]
    # print(results)

    final_segmented = np.zeros_like(smoothed_slice)
    for i, value in results:
        # print(i)
        # print(value)
        final_segmented[segmented_kmeans == sort_counts_segment[i][0]] = value
    return final_segmented


if __name__ == "__main__":
    # loading data
    data = loadmat('Brain.mat')
    T1 = data['T1']
    labels = data['label']
    print(T1.shape)
    print(labels.shape)

    final_segmentations = []
    f1_scores = []
    for i in range(T1.shape[2]):
        image = T1[:, :, i]
        label = labels[:, :, i]

        final_segmented = kmeans_segment(image, label)
        final_segmentations.append(final_segmented)

        f1 = utils.calculate_f1_score(label, final_segmented)
        f1_scores.append(f1)

    total_f1 = 0
    for i, score in enumerate(f1_scores, start=1):
        print(f"segmentation[{i}]:{score:.4f}")
        total_f1 += score

    average_f1 = total_f1 / len(f1_scores)
    print(f"average_f1：{average_f1:.4f}")

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


