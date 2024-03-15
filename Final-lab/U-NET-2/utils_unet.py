import numpy as np
import torch

def deal_pred_img(pred_img):
    # 处理结果
    # 假设pred是网络的输出，形状为[1, 6, 512, 512]
    # 将网络输出转换为numpy数组，并去掉批次维度，结果形状为[6, 512, 512]
    pred = pred_img.data.cpu().numpy()[0]

    # 对于每个像素点，找到概率最高的类别索引
    pred_class_indices = np.argmax(pred, axis=0)

    # 可选：根据需要将类别索引转换为实际的像素值，这里类别索引本身即为所需像素值（0-5）

    # 保存结果图像，需要将类别索引转换为uint8类型
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