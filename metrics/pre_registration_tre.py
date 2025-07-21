import os
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import cv2
import torch
from scipy.spatial.distance import euclidean
import pickle
from tqdm import tqdm

# 导入BSAFusion_test.py中的模型和函数
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils_2d.warp import Warper2d
from modal_2d.RegFusion_lite import Encoder, ModelTransfer_lite, RegNet_lite, FusionNet_lite
from utils_2d.utils import project


def mark_points_manually(image, num_points=5, window_name="标记特征点"):
    """
    手动在图像上标记特征点
    
    参数:
        image: 输入图像
        num_points: 要标记的点数
        window_name: 窗口名称
    
    返回:
        points: 标记的点坐标列表 [(x1,y1), (x2,y2), ...]
    """
    points = []
    temp_image = image.copy()
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal points, temp_image
        
        if event == cv2.EVENT_LBUTTONDOWN:
            # 添加点并在图像上绘制
            points.append((x, y))
            cv2.circle(temp_image, (x, y), 5, (0, 255, 0), -1)
            cv2.putText(temp_image, str(len(points)), (x+10, y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.imshow(window_name, temp_image)
            
            print(f"标记点 {len(points)}: ({x}, {y})")
            
            if len(points) >= num_points:
                print(f"已标记 {num_points} 个点，按任意键继续...")
    
    # 创建窗口并设置鼠标回调
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)
    
    # 显示图像
    cv2.imshow(window_name, temp_image)
    print(f"请在图像上标记 {num_points} 个特征点（点击鼠标左键）")
    
    # 等待直到标记了足够的点或按下ESC键
    while len(points) < num_points:
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC键
            break
    
    cv2.destroyWindow(window_name)
    return np.array(points)


def transform_points_with_flow(points, flow):
    """
    使用光流场变换点坐标
    
    参数:
        points: 原始点坐标列表 [(x1,y1), (x2,y2), ...]
        flow: 光流场，形状为 [1, 2, H, W]
    
    返回:
        transformed_points: 变换后的点坐标列表
    """
    transformed_points = []
    flow = flow.squeeze().cpu().numpy()  # [2, H, W]
    
    for x, y in points:
        # 确保坐标在图像范围内
        x, y = int(round(x)), int(round(y))
        x = max(0, min(x, flow.shape[2] - 1))
        y = max(0, min(y, flow.shape[1] - 1))
        
        # 获取该点的位移向量
        dx = flow[0, y, x]  # 注意OpenCV坐标系是y,x而非x,y
        dy = flow[1, y, x]
        
        # 计算变换后的坐标
        new_x = x + dx
        new_y = y + dy
        
        transformed_points.append((new_x, new_y))
    
    return np.array(transformed_points)


def calculate_tre(reference_points, transformed_points):
    """
    计算目标配准误差 (Target Registration Error, TRE)
    
    参数:
        reference_points: 参考图像中的特征点坐标
        transformed_points: 变换后的特征点坐标
    
    返回:
        mean_tre: 平均TRE
        max_tre: 最大TRE
        std_tre: TRE的标准差
        tre_values: 每个点对的TRE值列表
    """
    if len(reference_points) != len(transformed_points):
        raise ValueError("参考点和变换点的数量必须相同")
    
    if len(reference_points) == 0:
        raise ValueError("至少需要一个特征点")
    
    # 计算每对点之间的欧氏距离
    tre_values = [euclidean(ref, trans) for ref, trans in zip(reference_points, transformed_points)]
    
    # 计算统计量
    mean_tre = np.mean(tre_values)
    max_tre = np.max(tre_values)
    std_tre = np.std(tre_values)
    
    return mean_tre, max_tre, std_tre, tre_values


def visualize_tre(reference_image, registered_image, reference_points, transformed_points, tre_values):
    """
    可视化TRE结果
    
    参数:
        reference_image: 参考图像
        registered_image: 配准后图像
        reference_points: 参考图像中的特征点
        transformed_points: 变换后的特征点
        tre_values: 每个点对的TRE值
    """
    # 创建彩色图像用于可视化
    if len(reference_image.shape) == 2:
        ref_vis = cv2.cvtColor(reference_image.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        reg_vis = cv2.cvtColor(registered_image.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    else:
        ref_vis = reference_image.copy()
        reg_vis = registered_image.copy()
    
    # 在图像上绘制特征点
    for i, (ref_pt, trans_pt, tre) in enumerate(zip(reference_points, transformed_points, tre_values)):
        # 参考图像上的点
        cv2.circle(ref_vis, (int(ref_pt[0]), int(ref_pt[1])), 5, (0, 255, 0), -1)
        cv2.putText(ref_vis, f"{i+1}: {tre:.2f}", (int(ref_pt[0])+10, int(ref_pt[1])), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # 配准图像上的点
        cv2.circle(reg_vis, (int(trans_pt[0]), int(trans_pt[1])), 5, (0, 255, 0), -1)
        cv2.putText(reg_vis, f"{i+1}: {tre:.2f}", (int(trans_pt[0])+10, int(trans_pt[1])), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # 显示图像
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.imshow(cv2.cvtColor(ref_vis, cv2.COLOR_BGR2RGB))
    plt.title("参考图像上的特征点")
    plt.axis('off')
    
    plt.subplot(122)
    plt.imshow(cv2.cvtColor(reg_vis, cv2.COLOR_BGR2RGB))
    plt.title("配准图像上的变换后特征点")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig("pre_registration_tre_visualization.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # 绘制TRE分布直方图
    plt.figure(figsize=(8, 4))
    plt.hist(tre_values, bins=10, alpha=0.7, color='blue')
    plt.axvline(np.mean(tre_values), color='red', linestyle='dashed', linewidth=2, label=f'平均值: {np.mean(tre_values):.2f}')
    plt.xlabel('TRE (像素)')
    plt.ylabel('频率')
    plt.title('TRE分布直方图')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig("pre_registration_tre_histogram.png", dpi=300, bbox_inches='tight')
    plt.show()


def load_models(modal='CT', device=torch.device('cpu')):
    """
    加载预训练模型
    """
    checkpoint_path = r'C:\Users\King\Downloads\BSAFusion-main\checkpoint'
    checkpoint = torch.load(os.path.join(checkpoint_path, f'BSAFusion_{modal}.pkl'), map_location=device)
    
    encoder = Encoder().to(device)
    transfer = ModelTransfer_lite(num_vit=2, num_heads=4, img_size=[256, 256]).to(device)
    reg_net = RegNet_lite().to(device)
    fusion_net = FusionNet_lite().to(device)
    
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    transfer.load_state_dict(checkpoint['transfer_state_dict'])
    reg_net.load_state_dict(checkpoint['reg_net_state_dict'])
    fusion_net.load_state_dict(checkpoint['fusion_net_state_dict'])
    
    return encoder, transfer, reg_net, fusion_net


def get_registration_flow(img1, img2, encoder, transfer, reg_net, device):
    """
    获取配准的光流场
    """
    image_warp = Warper2d()
    
    # 确保图像是单通道的
    if len(img1.shape) == 3 and img1.shape[2] == 3:
        # 将彩色图像转换为灰度图
        img1_gray = np.dot(img1[...,:3], [0.299, 0.587, 0.114])
        img1 = img1_gray[np.newaxis, np.newaxis, :, :]
    elif len(img1.shape) == 2:
        img1 = img1[np.newaxis, np.newaxis, :, :]
    else:
        raise ValueError("不支持的图像格式")
        
    if len(img2.shape) == 3 and img2.shape[2] == 3:
        # 将彩色图像转换为灰度图
        img2_gray = np.dot(img2[...,:3], [0.299, 0.587, 0.114])
        img2 = img2_gray[np.newaxis, np.newaxis, :, :]
    elif len(img2.shape) == 2:
        img2 = img2[np.newaxis, np.newaxis, :, :]
    else:
        raise ValueError("不支持的图像格式")
    
    img1 = torch.from_numpy(img1).float().to(device)
    img2 = torch.from_numpy(img2).float().to(device)
    
    H, W = img1.shape[2], img1.shape[3]
    
    # 前向传播获取光流场
    with torch.no_grad():
        AS_F, feature1 = encoder(img1)
        BS_F, feature2 = encoder(img2)
        pre1, pre2, feature_pred1, feature_pred2, feature1, feature2, AU_F, BU_F = transfer(feature1, feature2)
        
        feature1 = project(feature1, [H, W]).to(device)
        feature2 = project(feature2, [H, W]).to(device)
        
        _, _, flows, flow, _, _ = reg_net(feature1, feature2)
    
    return flow


def main():
    # 设置参数
    modal = 'CT'
    device = torch.device('cpu')
    num_points = 5  # 要标记的特征点数量
    
    # 加载模型
    encoder, transfer, reg_net, fusion_net = load_models(modal, device)
    encoder.eval()
    transfer.eval()
    reg_net.eval()
    fusion_net.eval()
    
    # 加载图像
    reference_img_path = r'C:\Users\King\Downloads\BSAFusion-main\data\testData\CT\MRI\0.png'
    moving_img_path = r'C:\Users\King\Downloads\BSAFusion-main\data\testData\CT\CT\0.png'
    
    reference_img = imageio.imread(reference_img_path).astype(np.float32) / 255.0
    moving_img = imageio.imread(moving_img_path).astype(np.float32) / 255.0
    
    # 显示原始图像用于标记
    ref_display = (reference_img * 255).astype(np.uint8)
    mov_display = (moving_img * 255).astype(np.uint8)
    
    # 标记特征点
    print("\n在参考图像(MRI)上标记特征点:")
    reference_points = mark_points_manually(ref_display, num_points, window_name="参考图像(MRI)")
    
    print("\n在待配准图像(CT)上标记对应的特征点:")
    moving_points = mark_points_manually(mov_display, num_points, window_name="待配准图像(CT)")
    
    # 保存标记的点
    points_data = {
        'reference_points': reference_points,
        'moving_points': moving_points
    }
    with open('marked_points.pkl', 'wb') as f:
        pickle.dump(points_data, f)
    print("\n特征点已保存到 marked_points.pkl")
    
    # 获取配准的光流场
    print("\n正在计算配准光流场...")
    flow = get_registration_flow(reference_img, moving_img, encoder, transfer, reg_net, device)
    
    # 使用光流场变换待配准图像上的特征点
    transformed_points = transform_points_with_flow(moving_points, flow)
    
    # 计算TRE
    mean_tre, max_tre, std_tre, tre_values = calculate_tre(reference_points, transformed_points)
    
    # 打印结果
    print("\nTRE结果:")
    print(f"平均TRE: {mean_tre:.4f} 像素")
    print(f"最大TRE: {max_tre:.4f} 像素")
    print(f"TRE标准差: {std_tre:.4f} 像素")
    print(f"各点TRE: {[f'{v:.4f}' for v in tre_values]}")
    
    # 可视化结果
    # 获取配准后的图像用于可视化
    image_warp = Warper2d()
    
    # 确保 moving_img 是单通道的
    if len(moving_img.shape) == 3 and moving_img.shape[2] == 3:
        # 将彩色图像转换为灰度图
        moving_img_gray = np.dot(moving_img[...,:3], [0.299, 0.587, 0.114])
        img2_tensor = torch.from_numpy(moving_img_gray[np.newaxis, np.newaxis, :, :]).float().to(device)
    else:
        img2_tensor = torch.from_numpy(moving_img[np.newaxis, np.newaxis, :, :]).float().to(device)
    
    # 确保 flow 的维度正确
    if flow.shape[0] != 1:
        flow = flow.unsqueeze(0)  # 添加批次维度
    
    warped_image = image_warp(flow, img2_tensor)
    warped_image = warped_image.squeeze().cpu().numpy()
    warped_display = (warped_image * 255).astype(np.uint8)
    
    visualize_tre(ref_display, warped_display, reference_points, transformed_points, tre_values)


if __name__ == "__main__":
    main()