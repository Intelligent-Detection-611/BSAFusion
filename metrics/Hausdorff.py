import numpy as np
import imageio.v2 as imageio
from scipy.ndimage import distance_transform_edt
from skimage import measure

def hausdorff_distance(img1, img2, threshold=0.5, percentile=95):
    """
    计算两个图像的Hausdorff距离
    
    参数:
        img1: 第一个图像（通常是配准后的图像）
        img2: 第二个图像（通常是参考图像/标签）
        threshold: 二值化阈值
        percentile: 百分位数，用于计算鲁棒Hausdorff距离（默认95%）
    
    返回:
        hausdorff距离（像素单位）
    """
    # 确保图像是二值图像
    if img1.max() > 1.0:
        img1 = img1 / 255.0
    if img2.max() > 1.0:
        img2 = img2 / 255.0
    
    # 二值化
    binary_img1 = img1 > threshold
    binary_img2 = img2 > threshold
    
    # 计算边缘
    edges1 = measure.find_contours(binary_img1, 0.5)
    edges2 = measure.find_contours(binary_img2, 0.5)
    
    # 如果没有边缘，返回无穷大
    if len(edges1) == 0 or len(edges2) == 0:
        return float('inf')
    
    # 合并所有边缘点
    points1 = np.vstack([edge for edge in edges1])
    points2 = np.vstack([edge for edge in edges2])
    
    # 计算点集之间的距离
    def _compute_distances(p1, p2):
        distances = np.zeros((len(p1), len(p2)))
        for i, point1 in enumerate(p1):
            for j, point2 in enumerate(p2):
                distances[i, j] = np.sqrt(np.sum((point1 - point2) ** 2))
        return distances
    
    # 计算从points1到points2的最小距离
    distances1to2 = np.array([np.min(_compute_distances(np.array([p1]), points2)) for p1 in points1])
    distances2to1 = np.array([np.min(_compute_distances(np.array([p2]), points1)) for p2 in points2])
    
    # 标准Hausdorff距离
    forward_hausdorff = np.max(distances1to2)
    backward_hausdorff = np.max(distances2to1)
    hausdorff = max(forward_hausdorff, backward_hausdorff)
    
    # 鲁棒Hausdorff距离（使用百分位数）
    robust_forward = np.percentile(distances1to2, percentile)
    robust_backward = np.percentile(distances2to1, percentile)
    robust_hausdorff = max(robust_forward, robust_backward)
    
    return hausdorff, robust_hausdorff

# 更高效的实现（使用距离变换）
def hausdorff_distance_edt(img1, img2, threshold=0.5, percentile=95):
    """
    使用欧氏距离变换计算Hausdorff距离（更高效）
    """
    # 确保图像是二值图像
    if img1.max() > 1.0:
        img1 = img1 / 255.0
    if img2.max() > 1.0:
        img2 = img2 / 255.0
    
    # 二值化
    binary_img1 = img1 > threshold
    binary_img2 = img2 > threshold
    
    # 计算距离变换
    dist_map1 = distance_transform_edt(~binary_img1)
    dist_map2 = distance_transform_edt(~binary_img2)
    
    # 计算Hausdorff距离
    max_dist1 = np.max(dist_map1[binary_img2])
    max_dist2 = np.max(dist_map2[binary_img1])
    hausdorff = max(max_dist1, max_dist2)
    
    # 计算鲁棒Hausdorff距离
    robust_dist1 = np.percentile(dist_map1[binary_img2], percentile)
    robust_dist2 = np.percentile(dist_map2[binary_img1], percentile)
    robust_hausdorff = max(robust_dist1, robust_dist2)
    
    return hausdorff, robust_hausdorff

# 示例用法
if __name__ == "__main__":
    # 加载图像
    warped_img = imageio.imread(r'C:\Users\King\Downloads\BSAFusion-main\CT_result\CT_align\1.png').astype(np.float32)
    reference_img = imageio.imread(r'C:\Users\King\Downloads\BSAFusion-main\CT_result\MRI\1.png').astype(np.float32)
    
    # 计算Hausdorff距离
    hd, robust_hd = hausdorff_distance_edt(warped_img, reference_img)
    print(f"Hausdorff距离: {hd:.4f} 像素")
    print(f"95%鲁棒Hausdorff距离: {robust_hd:.4f} 像素")