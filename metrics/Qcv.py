# import numpy as np
# import cv2
# from scipy.stats import entropy

# def image_entropy(img):
#     """计算单通道图像的熵"""
#     hist, _ = np.histogram(img.flatten(), bins=256, range=(0, 255), density=True)
#     hist = hist + 1e-12  # 防止log(0)
#     return -np.sum(hist * np.log2(hist))

# def local_entropy_map(img, win_size=8):
#     """
#     计算图像局部熵图，窗口大小win_size。
#     返回与图像大小相同的局部熵矩阵（边缘采用镜像padding）
#     """
#     pad = win_size // 2
#     padded = cv2.copyMakeBorder(img, pad, pad, pad, pad, borderType=cv2.BORDER_REFLECT)
#     entropy_map = np.zeros_like(img, dtype=np.float32)

#     for i in range(img.shape[0]):
#         for j in range(img.shape[1]):
#             window = padded[i:i+win_size, j:j+win_size]
#             entropy_map[i,j] = image_entropy(window)
#     return entropy_map

# def local_correlation(img1, img2, win_size=8):
#     """
#     计算两幅图像的局部相关性，窗口大小win_size。
#     返回局部相关性矩阵。
#     """
#     pad = win_size // 2
#     padded1 = cv2.copyMakeBorder(img1, pad, pad, pad, pad, borderType=cv2.BORDER_REFLECT)
#     padded2 = cv2.copyMakeBorder(img2, pad, pad, pad, pad, borderType=cv2.BORDER_REFLECT)
#     corr_map = np.zeros_like(img1, dtype=np.float32)

#     for i in range(img1.shape[0]):
#         for j in range(img1.shape[1]):
#             win1 = padded1[i:i+win_size, j:j+win_size].flatten()
#             win2 = padded2[i:i+win_size, j:j+win_size].flatten()
#             mean1 = np.mean(win1)
#             mean2 = np.mean(win2)
#             numerator = np.sum((win1 - mean1) * (win2 - mean2))
#             denominator = np.sqrt(np.sum((win1 - mean1)**2) * np.sum((win2 - mean2)**2)) + 1e-12
#             corr_map[i,j] = numerator / denominator
#     return corr_map

# def compute_qcv(imgA, imgB, imgF, win_size=8):
#     """
#     计算Qcv指标：
#     imgA, imgB 输入源图像，imgF 融合结果，均为灰度图，uint8或float64范围0-255
#     """
#     # 转float并归一化到0-255
#     imgA = imgA.astype(np.float32)
#     imgB = imgB.astype(np.float32)
#     imgF = imgF.astype(np.float32)

#     # 计算局部熵
#     entA = local_entropy_map(imgA, win_size)
#     entB = local_entropy_map(imgB, win_size)
#     entF = local_entropy_map(imgF, win_size)

#     # 计算局部相关性
#     corrAF = local_correlation(imgA, imgF, win_size)
#     corrBF = local_correlation(imgB, imgF, win_size)

#     # 计算权重wA和wB，基于局部熵
#     wA = entA / (entA + entB + 1e-12)
#     wB = entB / (entA + entB + 1e-12)

#     # 计算区域融合评分q
#     qA = wA * corrAF * entF / (entA + 1e-12)
#     qB = wB * corrBF * entF / (entB + 1e-12)

#     # Qcv取平均负值，越低越好
#     Qcv = np.mean(qA + qB)

#     return Qcv

# if __name__ == "__main__":
#     # 示例读取
#     import matplotlib.pyplot as plt
#     imgA = cv2.imread(r'C:\Users\King\Downloads\BSAFusion-main\CT_result\CT_align\0.png', cv2.IMREAD_GRAYSCALE)
#     imgB = cv2.imread(r'C:\Users\King\Downloads\BSAFusion-main\CT_result\MRI\0.png', cv2.IMREAD_GRAYSCALE)
#     imgF = cv2.imread(r'C:\Users\King\Downloads\BSAFusion-main\CT_result\Fusion\0.png', cv2.IMREAD_GRAYSCALE)

#     qcv_value = compute_qcv(imgA, imgB, imgF, win_size=8)
#     print(f"Qcv value: {qcv_value:.6f}")

#     # 测试不同图像对
#     for i in range(10):
#         try:
#             imgA_test = cv2.imread(f'C:\\Users\\King\\Downloads\\BSAFusion-main\\CT_result\\CT_align\\{i}.png', cv2.IMREAD_GRAYSCALE)
#             imgB_test = cv2.imread(f'C:\\Users\\King\\Downloads\\BSAFusion-main\\CT_result\\MRI\\{i}.png', cv2.IMREAD_GRAYSCALE)
#             imgF_test = cv2.imread(f'C:\\Users\\King\\Downloads\\BSAFusion-main\\CT_result\\Fusion\\{i}.png', cv2.IMREAD_GRAYSCALE)
            
#             score = compute_qcv(imgA_test, imgB_test, imgF_test, win_size=8)
#             print(f"Image {i} Qcv value: {score:.6f}")
#         except:
#             continue

import numpy as np
import cv2
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from skimage.util import view_as_blocks

def normalize1(data):
    data = data.astype(np.float64)
    da = data.max()
    xiao = data.min()
    if da == 0 and xiao == 0:
        return data
    else:
        newdata = (data - xiao) / (da - xiao)
        return np.round(newdata * 255)

def block_process(image, block_size, func):
    """
    类似于 MATLAB blkproc 功能，对图像分块后对每个块使用func计算
    image: 2D ndarray
    block_size: (h, w)
    func: 接收块数据返回标量
    返回：每个块对应的结果组成矩阵
    """
    h, w = image.shape
    bh, bw = block_size
    # 按块切分
    blocks = view_as_blocks(image, block_shape=block_size)
    H, W = blocks.shape[:2]
    res = np.zeros((H, W))
    for i in range(H):
        for j in range(W):
            res[i, j] = func(blocks[i, j])
    return res

def Qcv(im1, im2, fused):
    # 参数设置
    alpha_c = 1
    alpha_s = 0.685
    f_c = 97.3227
    f_s = 12.1653
    windowSize = 16
    alpha = 5  # 幂指数

    # 转double + 归一化到0-255整数
    im1 = normalize1(im1)
    im2 = normalize1(im2)
    fused = normalize1(fused)

    # 转灰度（如果是彩色）
    def to_gray(img):
        if img.ndim == 3 and img.shape[2] == 3:
            return cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float64)
        return img.astype(np.float64)

    im1 = to_gray(im1)
    im2 = to_gray(im2)
    fused = to_gray(fused)

    # Sobel滤波器
    flt1 = np.array([[-1, 0, 1],
                     [-2, 0, 2],
                     [-1, 0, 1]], dtype=np.float64)
    flt2 = np.array([[-1, -2, -1],
                     [0,  0,  0],
                     [1,  2,  1]], dtype=np.float64)

    def filter2(img, kernel):
        return cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_REPLICATE)

    # 提取边缘信息
    fuseX = filter2(fused, flt1)
    fuseY = filter2(fused, flt2)
    fuseG = np.sqrt(fuseX**2 + fuseY**2)
    fuseX[fuseX == 0] = 1e-5  # 防止除0
    fuseA = np.arctan(fuseY / fuseX)

    img1X = filter2(im1, flt1)
    img1Y = filter2(im1, flt2)
    im1G = np.sqrt(img1X**2 + img1Y**2)
    img1X[img1X == 0] = 1e-5
    im1A = np.arctan(img1Y / img1X)

    img2X = filter2(im2, flt1)
    img2Y = filter2(im2, flt2)
    im2G = np.sqrt(img2X**2 + img2Y**2)
    img2X[img2X == 0] = 1e-5
    im2A = np.arctan(img2Y / img2X)

    # 计算局部区域显著性
    hang, lie = im1.shape
    H = hang // windowSize
    L = lie // windowSize

    fun = lambda x: np.sum(x**alpha)  # 块内求和（幂）

    ramda1 = block_process(im1G, (windowSize, windowSize), fun)
    ramda2 = block_process(im2G, (windowSize, windowSize), fun)

    # 相似度测量

    f1 = im1 - fused
    f2 = im2 - fused

    # 频域构建 CSF 滤波器，参考 freqspace
    u = np.linspace(-0.5, 0.5, lie, endpoint=False)
    v = np.linspace(-0.5, 0.5, hang, endpoint=False)
    U, V = np.meshgrid(u, v)

    # 缩放比例(论文中取lie/8, hang/8)
    U *= lie / 8
    V *= hang / 8
    r = np.sqrt(U**2 + V**2)

    # Mannos-Skarison滤波器
    theta_m = 2.6 * (0.0192 + 0.144 * r) * np.exp(-(0.144 * r)**1.1)

    # Daly滤波器
    r_nonzero = r.copy()
    r_nonzero[r_nonzero == 0] = 1  # 避免零除
    buff = 0.008 / (r_nonzero**3) + 1
    buff = buff ** (-0.2)
    buff1 = -0.3 * r * np.sqrt(1 + 0.06 * np.exp(0.3 * r))
    theta_d = buff * (1.42 * r * np.exp(buff1))
    theta_d[r == 0] = 0

    # Ahumada滤波器
    theta_a = alpha_c * np.exp(-(r / f_c)**2) - alpha_s * np.exp(-(r / f_s)**2)

    # 频域滤波
    ff1 = fft2(f1)
    ff2 = fft2(f2)

    # 使用Mannos-Skarison滤波器作为示例
    filtered_ff1 = fftshift(ff1) * theta_m
    filtered_ff2 = fftshift(ff2) * theta_m

    Df1 = np.real(ifft2(ifftshift(filtered_ff1)))
    Df2 = np.real(ifft2(ifftshift(filtered_ff2)))

    # 块均方值
    fun2 = lambda x: np.mean(x**2)
    D1 = block_process(Df1, (windowSize, windowSize), fun2)
    D2 = block_process(Df2, (windowSize, windowSize), fun2)

    # 全局质量计算
    numerator = np.sum(ramda1 * D1 + ramda2 * D2)
    denominator = np.sum(ramda1 + ramda2)

    Q = numerator / denominator if denominator != 0 else 0

    return Q

def metricsQcv(img1, img2, fused):
    """多通道支持"""
    fused = fused.astype(np.float64)
    b = fused.shape[2] if fused.ndim == 3 else 1
    b1 = img2.shape[2] if img2.ndim == 3 else 1

    if b == 1:
        return Qcv(img1, img2, fused)
    elif b1 == 1:
        scores = []
        for k in range(b):
            scores.append(Qcv(img1[..., k], img2, fused[..., k]))
        return np.mean(scores)
    else:
        scores = []
        for k in range(b):
            scores.append(Qcv(img1[..., k], img2[..., k], fused[..., k]))
        return np.mean(scores)


if __name__ == "__main__":
    import imageio

    img1 = imageio.imread(r'C:\Users\King\Downloads\BSAFusion-main\CT_result\CT_align\0.png')
    img2 = imageio.imread(r'C:\Users\King\Downloads\BSAFusion-main\CT_result\MRI\0.png')
    fused = imageio.imread(r'C:\Users\King\Downloads\BSAFusion-main\CT_result\Fusion\0.png')

    score = metricsQcv(img1, img2, fused)
    print(f"Qcv score: {score:.4f}")