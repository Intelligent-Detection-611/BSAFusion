import numpy as np
import cv2

def rgb2gray(img):
    """RGB转灰度"""
    if img.ndim == 3 and img.shape[2] == 3:
        return cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    else:
        return img

def mse(a, b):
    """
    均方误差计算，与MATLAB逻辑匹配：
    使用的是 sqrt(sum((a-b)^2)) / (m*n) 与标准MSE定义略有不同
    """
    a_gray = rgb2gray(a).astype(np.float64)
    b_gray = rgb2gray(b).astype(np.float64)

    m, n = a_gray.shape
    diff = a_gray - b_gray
    temp = np.sqrt(np.sum(diff**2))
    res0 = temp / (m * n)
    return res0

def psnr(img1, img2, fused):
    B = 8
    MAX = 2**B - 1

    mes = (mse(img1, fused) + mse(img2, fused)) / 2.0
    if mes == 0:
        return float('inf')  # 完美重合，PSNR无穷大

    psnr_value = 20 * np.log10(MAX / np.sqrt(mes))
    return psnr_value

def metrics_psnr(img1, img2, fused):
    fused = np.array(fused, dtype=np.float64)
    img1 = np.array(img1, dtype=np.float64)
    img2 = np.array(img2, dtype=np.float64)

    if fused.ndim == 2:
        return psnr(img1, img2, fused)
    elif fused.ndim == 3:
        b = fused.shape[2]
        b1 = img2.shape[2] if img2.ndim == 3 else 1

        if b1 == 1:
            scores = [psnr(img1[:, :, k], img2, fused[:, :, k]) for k in range(b)]
        else:
            scores = [psnr(img1[:, :, k], img2[:, :, k], fused[:, :, k]) for k in range(b)]
        return np.mean(scores)
    else:
        raise ValueError("Unsupported image dimensions.")

if __name__ == "__main__":
    import imageio

    img1 = imageio.imread(r'C:\Users\King\Downloads\BSAFusion-main\CT_result\CT_align\0.png')
    img2 = imageio.imread(r'C:\Users\King\Downloads\BSAFusion-main\CT_result\MRI\0.png')
    fused = imageio.imread(r'C:\Users\King\Downloads\BSAFusion-main\CT_result\Fusion\0.png')

    score = metrics_psnr(img1, img2, fused)
    print(f"PSNR score: {score:.4f}")