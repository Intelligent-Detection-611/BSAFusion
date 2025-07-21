import numpy as np
import cv2

def rgb2gray(img):
    """将RGB图像转换为灰度"""
    if len(img.shape) == 3 and img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        return img

def conv2(im, kernel):
    """二维卷积，保持大小不变，与MATLAB conv2(im,kernel,'same')效果"""
    return cv2.filter2D(im, -1, kernel, borderType=cv2.BORDER_REPLICATE)

def Qabf(img1, img2, fused):
    # 参数定义
    L = 1  
    Tg = 0.9994
    kg = -15
    Dg = 0.5
    Ta = 0.9879
    ka = -22
    Da = 0.8

    # Sobel算子(与原MATLAB对应)
    h1 = np.array([[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]], dtype=np.float32)
    h3 = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=np.float32)

    # 转换灰度图
    pA = rgb2gray(img1).astype(np.float64)
    pB = rgb2gray(img2).astype(np.float64)
    pF = rgb2gray(fused).astype(np.float64)

    # 归一化到0-255 float64 (im2double相当于除以255，这里乘以255为了保持和MATLAB近似)
    if pA.max() <= 1.0:
        pA = pA * 255
    if pB.max() <= 1.0:
        pB = pB * 255
    if pF.max() <= 1.0:
        pF = pF * 255

    # 计算梯度响应
    SAx = conv2(pA, h3)
    SAy = conv2(pA, h1)
    gA = np.sqrt(SAx**2 + SAy**2)

    SBx = conv2(pB, h3)
    SBy = conv2(pB, h1)
    gB = np.sqrt(SBx**2 + SBy**2)

    SFx = conv2(pF, h3)
    SFy = conv2(pF, h1)
    gF = np.sqrt(SFx**2 + SFy**2)

    # 计算梯度方向，避免除零
    def gradient_angle(Sx, Sy):
        angle = np.zeros_like(Sx)
        mask_zero = (Sx == 0)
        angle[mask_zero] = np.pi / 2
        angle[~mask_zero] = np.arctan(Sy[~mask_zero] / Sx[~mask_zero])
        return angle

    aA = gradient_angle(SAx, SAy)
    aB = gradient_angle(SBx, SBy)
    aF = gradient_angle(SFx, SFy)

    # 初始化Q矩阵
    M, N = gA.shape
    QAF = np.zeros((M, N))
    QBF = np.zeros((M, N))

    # 计算QAF
    for i in range(M):
        for j in range(N):
            # GAF 计算
            if gA[i, j] > gF[i, j]:
                GAF = gF[i, j] / gA[i, j]
            elif gA[i, j] == gF[i, j]:
                GAF = gF[i, j]
            else:
                GAF = gA[i, j] / gF[i, j]

            # AAF 计算
            AAF = 1 - abs(aA[i, j] - aF[i, j]) / (np.pi / 2)
            
            QgAF = Tg / (1 + np.exp(kg * (GAF - Dg)))
            QaAF = Ta / (1 + np.exp(ka * (AAF - Da)))

            QAF[i, j] = QgAF * QaAF

    # 计算QBF
    for i in range(M):
        for j in range(N):
            if gB[i, j] > gF[i, j]:
                GBF = gF[i, j] / gB[i, j]
            elif gB[i, j] == gF[i, j]:
                GBF = gF[i, j]
            else:
                GBF = gB[i, j] / gF[i, j]

            ABF = 1 - abs(aB[i, j] - aF[i, j]) / (np.pi / 2)

            QgBF = Tg / (1 + np.exp(kg * (GBF - Dg)))
            QaBF = Ta / (1 + np.exp(ka * (ABF - Da)))

            QBF[i, j] = QgBF * QaBF

    deno = np.sum(gA + gB)
    nume = np.sum(QAF * gA + QBF * gB)

    Qabf_value = nume / deno if deno != 0 else 0

    return Qabf_value


def metrics_qabf(img1, img2, fused):
    """
    对应matlab的metricsQabf函数，支持多通道情况
    """
    fused = fused.astype(np.float64)
    b = fused.shape[2] if fused.ndim == 3 else 1
    b1 = img2.shape[2] if img2.ndim == 3 else 1

    if b == 1:
        res = Qabf(img1, img2, fused)
    elif b1 == 1:
        scores = []
        for k in range(b):
            scores.append(Qabf(img1[:, :, k], img2, fused[:, :, k]))
        res = np.mean(scores)
    else:
        scores = []
        for k in range(b):
            scores.append(Qabf(img1[:, :, k], img2[:, :, k], fused[:, :, k]))
        res = np.mean(scores)
    return res


# 简单示例调用：
if __name__ == "__main__":
    import imageio
    img1 = imageio.imread(r'C:\Users\King\Downloads\BSAFusion-main\CT_result\CT_align\1.png')
    img2 = imageio.imread(r'C:\Users\King\Downloads\BSAFusion-main\CT_result\MRI\1.png')
    fused = imageio.imread(r'C:\Users\King\Downloads\BSAFusion-main\CT_result\Fusion\1.png')

    score = metrics_qabf(img1, img2, fused)
    print(f"Qabf score: {score:.4f}")

    #测试不同图像对
    for i in range(10):
        try:
            imgA_test = cv2.imread(f'C:\\Users\\King\\Downloads\\BSAFusion-main\\CT_result\\CT_align\\{i}.png', cv2.IMREAD_GRAYSCALE).astype(np.float64)
            imgB_test = cv2.imread(f'C:\\Users\\King\\Downloads\\BSAFusion-main\\CT_result\\MRI\\{i}.png', cv2.IMREAD_GRAYSCALE).astype(np.float64)
            imgF_test = cv2.imread(f'C:\\Users\\King\\Downloads\\BSAFusion-main\\CT_result\\Fusion\\{i}.png', cv2.IMREAD_GRAYSCALE).astype(np.float64)
            
            score = metrics_qabf(imgA_test, imgB_test, imgF_test)
            print(f"Image {i} Q_AB/F: {score:.4f}")
        except:
            continue