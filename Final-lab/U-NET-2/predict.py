import numpy as np
import torch
import utils_unet
from model import UNet
from scipy.io import loadmat
from dataset import data_process
import matplotlib.pyplot as plt


def predict(net, img):
    # Turn to tensor
    img_tensor = torch.from_numpy(img)
    # Copy the tensor to the device
    img_tensor = img_tensor.to(device=device, dtype=torch.float32)
    print(img_tensor.shape)
    # predict
    pred = net(img_tensor)
    # Processing results
    # Suppose pred is the output of the network, and the shape is [1, 6, 512, 512]
    # Convert the network output to a numpy array and remove the batch dimension. The resulting shape is [6, 512, 512]
    pred = pred.data.cpu().numpy()[0]

    # For each pixel, find the category index with the highest probability.
    pred_class_indices = np.argmax(pred, axis=0)

    # To save the result image, need to convert the category index to uint8 type.
    pred_image = pred_class_indices.astype(np.uint8)

    return pred_image


if __name__ == "__main__":
    # Select device
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    # Load network, picture single channel, classified as 6.
    net = UNet(n_channels=1, n_classes=6)
    # Copy the network to deivce
    net.to(device=device)
    # Load model parameters
    net.load_state_dict(torch.load('model/best_model.pth', map_location=device))
    # Test mode
    net.eval()
    # Read all image paths
    data = loadmat('Brain.mat')
    tests_data = data['T1']
    labels = data['label']

    # predict and evaluation
    final_segmentations = []
    f1_scores = []
    processed_labels = []
    for i in range(tests_data.shape[2]):
        img = tests_data[:, :, i]
        label = labels[:, :, i]
        img, label = data_process(img, label)
        img = img.reshape(1, 1, 512, 512)
        pred_image = predict(net, img)
        f1 = utils_unet.calculate_f1_score(pred_image, label)
        final_segmentations.append(pred_image)
        f1_scores.append(f1)
        processed_labels.append(np.squeeze(label))
    # print(type(final_segmentations[0]))
    # print(final_segmentations[0].shape)
    specificity_list = []
    sensitivity_list = []
    for i in range(tests_data.shape[2]):
        sensitivity_list.append(utils_unet.calculate_weighted_sensitivity(processed_labels[i], final_segmentations[i]))
        specificity_list.append(utils_unet.calculate_weighted_specificity(processed_labels[i], final_segmentations[i]))
    print(len(f1_scores))
    for i in range(len(f1_scores)):
        print(
            f"Result[{i + 1}]  F1:{f1_scores[i]:.4f}  sensitivity:{sensitivity_list[i]:.4f}  specificity:{specificity_list[i]:.4f}")

    # diaplay
    plot_rows = 2
    plot_cols = 5
    fig_size = (6 * plot_cols, 6 * plot_rows)
    image_list = []
    image_list.extend([("segmented" + str(i + 1), seg) for i, seg in enumerate(final_segmentations)])
    font_size = 20
    camp = 'jet'
    fig, sub_figs = utils_unet.create_subplots(plot_rows, plot_cols, fig_size, image_list, font_size, camp)
    plt.tight_layout()
    plt.show()
