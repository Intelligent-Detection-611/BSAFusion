import numpy as np
from scipy.signal import convolve2d

def fspecial_gaussian(shape=(3, 3), sigma=0.5):
    """
    生成二维高斯滤波器，类似MATLAB fspecial('gaussian', ...)
    shape: tuple，高斯核大小 (height, width)
    sigma: 标准差
    返回归一化后的高斯滤波核
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1, -n:n+1]
    h = np.exp(-(x*x + y*y) / (2.*sigma*sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def vifp_mscale(ref, dist):
    """
    多尺度视觉信息保真度(Visual Information Fidelity)计算
    输入:
        ref: 参考图像，灰度float64数组
        dist: 待评估图像，灰度float64数组
    输出:
        vifp: VIF指标标量，越大越好
    """
    sigma_nsq = 2
    num = 0.0
    den = 0.0
    for scale in range(1, 5):
        N = 2**(4 - scale + 1) + 1  # 窗口大小
        win = fspecial_gaussian((N, N), N / 5)

        if scale > 1:
            ref = convolve2d(ref, win, mode='valid')
            dist = convolve2d(dist, win, mode='valid')
            ref = ref[::2, ::2]     # 下采样2倍
            dist = dist[::2, ::2]

        mu1 = convolve2d(ref, win, mode='valid')
        mu2 = convolve2d(dist, win, mode='valid')
        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = convolve2d(ref * ref, win, mode='valid') - mu1_sq
        sigma2_sq = convolve2d(dist * dist, win, mode='valid') - mu2_sq
        sigma12 = convolve2d(ref * dist, win, mode='valid') - mu1_mu2

        sigma1_sq[sigma1_sq < 0] = 0
        sigma2_sq[sigma2_sq < 0] = 0

        g = sigma12 / (sigma1_sq + 1e-10)
        sv_sq = sigma2_sq - g * sigma12

        # 修正条件
        g[sigma1_sq < 1e-10] = 0
        sv_sq[sigma1_sq < 1e-10] = sigma2_sq[sigma1_sq < 1e-10]
        sigma1_sq[sigma1_sq < 1e-10] = 0

        g[sigma2_sq < 1e-10] = 0
        sv_sq[sigma2_sq < 1e-10] = 0

        sv_sq[g < 0] = sigma2_sq[g < 0]
        g[g < 0] = 0
        sv_sq[sv_sq <= 1e-10] = 1e-10

        num += np.sum(np.log10(1 + (g ** 2) * sigma1_sq / (sv_sq + sigma_nsq)))
        den += np.sum(np.log10(1 + sigma1_sq / sigma_nsq))

    vifp = num / den
    return vifp

def VIF_function(A, B, F):
    """
    基于多尺度VIF计算融合图F相对源图A和B的视觉信息保真度综合评分
    """
    vif_A = vifp_mscale(A, F)
    vif_B = vifp_mscale(B, F)
    VIF = vif_A + vif_B
    return VIF

# 示例用法
if __name__ == "__main__":
    import imageio
    import cv2

    # 读取灰度图像，确保输入为float64
    A = imageio.imread(r'C:\Users\King\Downloads\BSAFusion-main\CT_result\CT_align\0.png').astype(np.float32)
    B = imageio.imread(r'C:\Users\King\Downloads\BSAFusion-main\CT_result\MRI\0.png').astype(np.float32)
    F = imageio.imread(r'C:\Users\King\Downloads\BSAFusion-main\CT_result\Fusion\0.png').astype(np.float32)

    # 转灰度 float64格式
    if A.ndim == 3:
        A = cv2.cvtColor(A, cv2.COLOR_BGR2GRAY)
    if B.ndim == 3:
        B = cv2.cvtColor(B, cv2.COLOR_BGR2GRAY)
    if F.ndim == 3:
        F = cv2.cvtColor(F, cv2.COLOR_BGR2GRAY)

    A = A.astype(np.float64)
    B = B.astype(np.float64)
    F = F.astype(np.float64)

    vif_score = VIF_function(A, B, F)
    print(f"VIF score: {vif_score:.4f}")