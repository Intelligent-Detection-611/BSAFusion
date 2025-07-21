import numpy as np
import cv2
import imageio.v2 as imageio
import os


def calculate_mutual_information(image1, image2, bins=256):
    """
    计算两幅图像之间的互信息(MI)
    
    参数:
        image1: 第一幅图像，numpy数组
        image2: 第二幅图像，numpy数组
        bins: 直方图的箱数
        
    返回:
        mi: 互信息值
    """
    # 确保图像尺寸相同
    if image1.shape != image2.shape:
        raise ValueError("两幅图像的尺寸必须相同")
    
    # 如果是彩色图像，转换为灰度图
    if len(image1.shape) == 3 and image1.shape[2] == 3:
        image1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
    if len(image2.shape) == 3 and image2.shape[2] == 3:
        image2 = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)
    
    # 将图像值归一化到[0, bins-1]范围
    image1_normalized = np.round((image1 - image1.min()) / (image1.max() - image1.min()) * (bins - 1))
    image2_normalized = np.round((image2 - image2.min()) / (image2.max() - image2.min()) * (bins - 1))
    
    # 计算联合直方图
    hist_2d, _, _ = np.histogram2d(
        image1_normalized.flatten(),
        image2_normalized.flatten(),
        bins=bins,
        range=[[0, bins-1], [0, bins-1]]
    )
    
    # 计算互信息
    # 归一化联合直方图，使其成为联合概率分布
    pxy = hist_2d / float(np.sum(hist_2d))
    px = np.sum(pxy, axis=1)  # 边缘概率分布 p(x)
    py = np.sum(pxy, axis=0)  # 边缘概率分布 p(y)
    px_py = px[:, None] * py[None, :]  # 边缘分布的乘积 p(x) * p(y)
    
    # 计算非零概率的索引，避免log(0)
    nzs = pxy > 0
    
    # 计算互信息
    mi = np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))
    
    return mi


def calculate_normalized_mutual_information(image1, image2, bins=256):
    """
    计算两幅图像之间的归一化互信息(NMI)
    
    参数:
        image1: 第一幅图像，numpy数组
        image2: 第二幅图像，numpy数组
        bins: 直方图的箱数
        
    返回:
        nmi_sum: 归一化互信息值 (MI / (H(X) + H(Y)))
        nmi_sqrt: 归一化互信息值 (MI / sqrt(H(X) * H(Y)))
    """
    # 确保图像尺寸相同
    if image1.shape != image2.shape:
        raise ValueError("两幅图像的尺寸必须相同")
    
    # 如果是彩色图像，转换为灰度图
    if len(image1.shape) == 3 and image1.shape[2] == 3:
        image1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
    if len(image2.shape) == 3 and image2.shape[2] == 3:
        image2 = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)
    
    # 将图像值归一化到[0, bins-1]范围
    image1_normalized = np.round((image1 - image1.min()) / (image1.max() - image1.min()) * (bins - 1))
    image2_normalized = np.round((image2 - image2.min()) / (image2.max() - image2.min()) * (bins - 1))
    
    # 计算联合直方图
    hist_2d, _, _ = np.histogram2d(
        image1_normalized.flatten(),
        image2_normalized.flatten(),
        bins=bins,
        range=[[0, bins-1], [0, bins-1]]
    )
    
    # 计算互信息和熵
    # 归一化联合直方图，使其成为联合概率分布
    pxy = hist_2d / float(np.sum(hist_2d))
    px = np.sum(pxy, axis=1)  # 边缘概率分布 p(x)
    py = np.sum(pxy, axis=0)  # 边缘概率分布 p(y)
    px_py = px[:, None] * py[None, :]  # 边缘分布的乘积 p(x) * p(y)
    
    # 计算非零概率的索引，避免log(0)
    nzs = pxy > 0
    
    # 计算互信息
    mi = np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))
    
    # 计算边缘熵
    hx = -np.sum(px * np.log(px + 1e-10))  # 添加小常数避免log(0)
    hy = -np.sum(py * np.log(py + 1e-10))
    
    # 计算归一化互信息
    # MI / sqrt(H(X) * H(Y))
    nmi_sqrt = mi / np.sqrt(hx * hy)
    
    return nmi_sqrt


def main():
    """
    主函数，用于测试两幅图像的互信息
    """
    # 设置图像路径
    image1 = imageio.imread(r'E:\VoxelMorph-torch-test\Result\pair1_10000_warped.png').astype(np.float32)
    image2 = imageio.imread(r'E:\VoxelMorph-torch-test\Result\pair1_1000_fixed.png').astype(np.float32)
    
    # 计算互信息
    mi = calculate_mutual_information(image1, image2)
    nmi_sqrt = calculate_normalized_mutual_information(image1, image2)
    
    # 打印结果
    print("\n计算结果:")
    print(f"互信息 (MI): {mi:.6f}")
    print(f"归一化互信息: {nmi_sqrt:.6f}")


if __name__ == "__main__":
    main()