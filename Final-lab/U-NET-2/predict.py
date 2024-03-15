import glob
import numpy as np
import torch
import os
import cv2
from model import UNet
from scipy.io import loadmat
from dataset import data_process
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # 选择设备
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    # 加载网络，图片单通道，分类为1。
    net = UNet(n_channels=1, n_classes=6)
    # 将网络拷贝到deivce中
    net.to(device=device)
    # 加载模型参数
    net.load_state_dict(torch.load('model/best_model.pth', map_location=device))
    # 测试模式
    net.eval()
    # 读取所有图片路径
    data = loadmat('Brain.mat')
    tests_path = data['T1']
    image1 = tests_path[:, :, 1]
    labels = data['label']
    label1 = labels[:, :, 1]
    image1, label1 = data_process(image1, label1)
    img = image1.reshape(1, 1, 512, 512)
    # 转为tensor
    img_tensor = torch.from_numpy(img)
    # 将tensor拷贝到device中，只用cpu就是拷贝到cpu中，用cuda就是拷贝到cuda中。
    img_tensor = img_tensor.to(device=device, dtype=torch.float32)
    print(img_tensor.shape)
    # 预测
    pred = net(img_tensor)
    # 处理结果
    # 假设pred是网络的输出，形状为[1, 6, 512, 512]
    # 将网络输出转换为numpy数组，并去掉批次维度，结果形状为[6, 512, 512]
    pred = pred.data.cpu().numpy()[0]

    # 对于每个像素点，找到概率最高的类别索引
    pred_class_indices = np.argmax(pred, axis=0)

    # 可选：根据需要将类别索引转换为实际的像素值，这里类别索引本身即为所需像素值（0-5）

    # 保存结果图像，需要将类别索引转换为uint8类型
    pred_image = pred_class_indices.astype(np.uint8)

    plt.imshow(pred_image, cmap='jet')

    plt.show()



    # 如果你需要将结果保存为彩色图像，可以根据类别索引映射到不同的颜色
    # 这里只是直接保存了类别索引作为像素值
    # cv2.imwrite(save_res_path, pred_image)
    # 遍历所有图片
    # for test_path in tests_path:
    #     # 保存结果地址
    #     save_res_path = test_path.split('.')[0] + '_res.png'
    #     # 读取图片
    #     img = cv2.imread(test_path)
    #     # 转为灰度图
    #     img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    #     # 转为batch为1，通道为1，大小为512*512的数组
    #     img = img.reshape(1, 1, img.shape[0], img.shape[1])
    #     # 转为tensor
    #     img_tensor = torch.from_numpy(img)
    #     # 将tensor拷贝到device中，只用cpu就是拷贝到cpu中，用cuda就是拷贝到cuda中。
    #     img_tensor = img_tensor.to(device=device, dtype=torch.float32)
    #     # 预测
    #     pred = net(img_tensor)
    #     # 提取结果
    #     pred = np.array(pred.data.cpu()[0])[0]
    #     # 处理结果
    #     pred[pred >= 0.5] = 255
    #     pred[pred < 0.5] = 0
    #     # 保存图片
    #     cv2.imwrite(save_res_path, pred)