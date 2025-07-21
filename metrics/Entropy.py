import numpy as np
import cv2

def Entropy(img1, img2, fused):
    """
    计算融合图像的熵值（信息量）
    img1, img2 在此函数内未被使用，仅为了接口统一保留参数。
    fused: 输入图像，灰度或彩色（uint8 或 float），范围不限，自动处理。
    返回熵的标量值
    """
    h = fused
    # 转灰度
    if h.ndim == 3 and h.shape[2] == 3:
        h1 = cv2.cvtColor(h.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    else:
        h1 = h

    # 转uint8类型，范围0-255
    if h1.dtype != np.uint8:
        h_min = h1.min()
        h_max = h1.max()
        if h_max > h_min:
            h1 = ((h1 - h_min) / (h_max - h_min) * 255).astype(np.uint8)
        else:
            h1 = np.zeros_like(h1, dtype=np.uint8)

    # 统计直方图，256 bins
    hist = np.bincount(h1.flatten(), minlength=256).astype(np.float64)

    # 概率分布
    P = hist / hist.sum()

    # 计算熵，只考虑非零概率项
    P_nonzero = P[P > 0]
    entropy_value = -np.sum(P_nonzero * np.log2(P_nonzero))

    return entropy_value

def metricsEntropy(img1, img2, fused):
    """
    多通道支持：融合图多通道时对每个通道计算熵，取平均
    """
    fused = np.array(fused)
    if fused.ndim == 2:
        # 单通道
        return Entropy(img1, img2, fused)
    elif fused.ndim == 3:
        b = fused.shape[2]
        b1 = img2.shape[2] if img2.ndim == 3 else 1
        if b1 == 1:
            scores = [Entropy(img1[..., k], img2, fused[..., k]) for k in range(b)]
            return np.mean(scores)
        else:
            scores = [Entropy(img1[..., k], img2[..., k], fused[..., k]) for k in range(b)]
            return np.mean(scores)
    else:
        raise ValueError("Unsupported image dimensions.")

# 测试示例：
if __name__ == "__main__":
    import imageio

    img1 = imageio.imread(r'C:\Users\King\Downloads\BSAFusion-main\CT_result\CT_align\0.png')
    img2 = imageio.imread(r'C:\Users\King\Downloads\BSAFusion-main\CT_result\MRI\0.png')
    fused = imageio.imread(r'C:\Users\King\Downloads\BSAFusion-main\CT_result\Fusion\0.png')

    score = metricsEntropy(img1, img2, fused)
    print(f"Entropy score: {score:.4f}")