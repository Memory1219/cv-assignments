import numpy as np
import torch
from model import UNet3D
from scipy.io import loadmat
import matplotlib.pyplot as plt
import utils

# Instantiated Model
device = torch.device('cpu')
model_loaded = UNet3D(n_channels=1, n_classes=6)
model_loaded.load_state_dict(torch.load('model/best_model.pth', map_location=device))
model_loaded = model_loaded.to(device)
model_loaded.eval()  # Set to evaluation mode

# loading and processing data
data = loadmat('Brain.mat')
tests_data = data['T1']
labels = data['label']

T1_tensor = torch.tensor(tests_data, dtype=torch.float32).unsqueeze(0)

input = T1_tensor.unsqueeze(0)

output = model_loaded(input)

predicted_output = torch.argmax(output, dim=1)
predicted_output = predicted_output.squeeze(0)

# predict and evaluate
specificity_list = []
sensitivity_list = []
f1_score_list = []
final_segmentations = []
for i in range(10):
    image = predicted_output[:, :, i]
    label = labels[:, :, i]
    final_segmentations.append(image)
    f1 = utils.calculate_f1_score(label, image)
    f1_score_list.append(f1)
    specificity = utils.calculate_weighted_specificity(label, image)
    specificity_list.append(specificity)
    sensitivity = utils.calculate_weighted_sensitivity(label, image)
    sensitivity_list.append(sensitivity)


# print and display
for i in range(len(f1_score_list)):
    print(f"Result[{i + 1}]  F1:{f1_score_list[i]:.4f}  sensitivity:{sensitivity_list[i]:.4f}  specificity:{specificity_list[i]:.4f}")

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


# print(predicted_output.shape)
# plt.imshow(predicted_output[:,:,0], cmap='jet')
# plt.show()
