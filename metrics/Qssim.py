import numpy as np
import cv2
from scipy.ndimage import convolve

def gaussian_kernel(size=11, sigma=1.5):
    """生成二维高斯核"""
    ax = np.arange(-size // 2 + 1., size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    return kernel / np.sum(kernel)

def ssim(img1, img2):
    """
    计算两幅图像的SSIM值，灰度或RGB均可，会自动转换为灰度
    img1,img2: uint8或者float32图像，范围0~255或0~1均可，但需统一
    返回mssim标量
    """

    # 转换为浮点数，尺度归一到0~255
    if img1.dtype != np.float32:
        img1 = img1.astype(np.float32)
    if img2.dtype != np.float32:
        img2 = img2.astype(np.float32)

    if img1.max() <= 1.0:
        img1 = img1 * 255
    if img2.max() <= 1.0:
        img2 = img2 * 255

    # 如果是彩色图，转成灰度
    if img1.ndim == 3 and img1.shape[2] == 3:
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    if img2.ndim == 3 and img2.shape[2] == 3:
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    K1 = 0.01
    K2 = 0.03
    L = 255  # 图像动态范围

    C1 = (K1 * L)**2
    C2 = (K2 * L)**2

    window = gaussian_kernel(11, 1.5)

    # 计算均值
    mu1 = convolve(img1, window, mode='reflect')
    mu2 = convolve(img2, window, mode='reflect')

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    # 计算方差和协方差
    sigma1_sq = convolve(img1 * img1, window, mode='reflect') - mu1_sq
    sigma2_sq = convolve(img2 * img2, window, mode='reflect') - mu2_sq
    sigma12 = convolve(img1 * img2, window, mode='reflect') - mu1_mu2

    # 计算SSIM地图
    numerator1 = 2 * mu1_mu2 + C1
    numerator2 = 2 * sigma12 + C2
    denominator1 = mu1_sq + mu2_sq + C1
    denominator2 = sigma1_sq + sigma2_sq + C2

    ssim_map = (numerator1 * numerator2) / (denominator1 * denominator2)

    return np.mean(ssim_map)


def metrics_ssim(img1, img2, fused):
    """
    对应MATLAB的metricsSsim函数
    img1: 第一幅源图（多通道）
    img2: 第二幅源图（可能单通道）
    fused: 融合结果图（多通道）
    返回融合图与两源图SSIM的和或均值
    """
    fused = fused.astype(np.float32)
    b = fused.shape[2] if fused.ndim == 3 else 1
    b1 = img2.shape[2] if img2.ndim == 3 else 1

    if b == 1:
        # 修复：只计算img1和fused的SSIM
        g = ssim(img1, fused)
        res = g
    elif b1 == 1:
        g = []
        for k in range(b):
            # 修复：分别计算每个通道与img2的SSIM，然后计算fused每个通道与img1对应通道的SSIM
            ssim1 = ssim(img1[:, :, k], fused[:, :, k])  # fused与img1的SSIM
            ssim2 = ssim(img2, fused[:, :, k])  # fused与img2的SSIM
            g.append((ssim1 + ssim2) / 2)  # 取平均
        res = np.mean(g)
    else:
        g = []
        for k in range(b):
            # 修复：分别计算fused与两个源图的SSIM
            ssim1 = ssim(img1[:, :, k], fused[:, :, k])
            ssim2 = ssim(img2[:, :, k], fused[:, :, k])
            g.append((ssim1 + ssim2) / 2)  # 取平均
        res = np.mean(g)
    return res


# 测试示例
if __name__ == "__main__":
    import imageio.v2 as imageio  # 修复：使用v2版本

    # 修复：移除as_gray参数，改用mode参数
    img1 = imageio.imread(r'C:\Users\King\Downloads\BSAFusion-main\CT_result\CT_align\0.png').astype(np.float32)
    img2 = imageio.imread(r'C:\Users\King\Downloads\BSAFusion-main\CT_result\MRI\0.png').astype(np.float32)
    fused = imageio.imread(r'C:\Users\King\Downloads\BSAFusion-main\CT_result\Fusion\0.png').astype(np.float32)

    score = metrics_ssim(img1, img2, fused)
    print("SSIM Metrics Score:", score)