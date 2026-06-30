import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# ==========================================
# 1. 配置参数与工具类
# ==========================================
VIDEO_PATH = r'D:\xxxxxxxxxxxxxxxxx.mp4' 
OUTPUT_DIR = r'D:\xxxxxxxxxxxxxxxxxxxx'
OUTPUT_POINTS_FILE = os.path.join(OUTPUT_DIR, 'combined_point_cloud_clean3.npy')
OUTPUT_DATA_FILE = os.path.join(OUTPUT_DIR, 'bamboo_metrics_32_clean3.csv')
OUTPUT_IMAGE_FILE = os.path.join(OUTPUT_DIR, 'bamboo_analysis_32_clean3.png')

# 激光颜色阈值
LOWER_RED = np.array([0, 0, 252])   
UPPER_RED = np.array([255, 255, 255])
MIN_LASER_PIXELS = 100 

class ManualCalibrator:
    def __init__(self):
        self.points = []

    def select_points(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.points) < 4:
                self.points.append((x, y))
                cv2.circle(param, (x, y), 5, (0, 0, 255), -1)
                cv2.putText(param, str(len(self.points)), (x+10, y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                cv2.imshow('Manual Calibration', param)

# ==========================================
# 2. 核心功能函数
# ==========================================
def detect_laser(frame):
    mask = cv2.inRange(frame, LOWER_RED, UPPER_RED)
    if np.count_nonzero(mask) > MIN_LASER_PIXELS:
        return True, mask
    return False, mask

def get_homography_manual(frame):
    print("\n" + "="*50)
    print("【系统提示】检测到激光，开始进行坐标系标定！")
    print("请在弹出的窗口中，依次点击靠近画面中心的 2x2 方格的 4 个角点。")
    print("点击顺序：1.左上 -> 2.右上 -> 3.右下 -> 4.左下")
    print("="*50 + "\n")

    calibrator = ManualCalibrator()
    temp_img = frame.copy()
    cv2.namedWindow('Manual Calibration')
    cv2.setMouseCallback('Manual Calibration', calibrator.select_points, temp_img)
    cv2.imshow('Manual Calibration', temp_img)
    
    while True:
        k = cv2.waitKey(20) & 0xFF
        if len(calibrator.points) == 4:
            pts = np.array(calibrator.points, np.int32)
            cv2.polylines(temp_img, [pts], True, (0, 255, 0), 2)
            cv2.imshow('Manual Calibration', temp_img)
            print("点选取完毕，按任意键继续...")
            cv2.waitKey(0)
            break
        if k == 27: 
            cv2.destroyAllWindows()
            return None
            
    cv2.destroyAllWindows()
    src_points = np.array(calibrator.points, dtype=np.float32)
    half_len = 15.0 
    dst_points = np.array([
        [-half_len, -half_len], [ half_len, -half_len],
        [ half_len,  half_len], [-half_len,  half_len]
    ], dtype=np.float32)
    H, _ = cv2.findHomography(src_points, dst_points)
    return H

def extract_and_map_laser_points(laser_mask, H):
    y_coords, x_coords = np.where(laser_mask > 0)
    if len(x_coords) == 0: return np.array([])
    pixel_coords = np.vstack((x_coords, y_coords, np.ones_like(x_coords)))
    P_mm = H @ pixel_coords
    X_mm = P_mm[0, :] / P_mm[2, :]
    Y_mm = P_mm[1, :] / P_mm[2, :]
    points = np.vstack((X_mm, Y_mm)).T
    valid_mask = (points[:, 1] >= 50) & (points[:, 1] <= 200)
    return points[valid_mask]

# ==========================================
# ✅ 新增：智能去噪函数 (连通域分析)
# ==========================================
def remove_isolated_noise(points, resolution=100.0, padding=10):
    """
    使用连通域分析去除孤立的噪声团。
    :param points: 输入点云 (N, 2)
    :param resolution: 栅格化分辨率 (像素/mm)，越大越精细
    :param padding: 图像边缘留白 (像素)
    """
    if len(points) == 0: return points
    print("正在进行智能去噪 (去除孤立噪点团)...")

    # 1. 计算边界并将点云映射到图像坐标系
    min_x, min_y = np.min(points, axis=0)
    max_x, max_y = np.max(points, axis=0)
    
    width_mm = max_x - min_x
    height_mm = max_y - min_y
    
    w_px = int(width_mm * resolution) + 2 * padding
    h_px = int(height_mm * resolution) + 2 * padding
    
    # 创建空白图像
    grid_img = np.zeros((h_px, w_px), dtype=np.uint8)
    
    # 坐标转换函数
    def to_img_coords(pts):
        ix = ((pts[:, 0] - min_x) * resolution).astype(int) + padding
        iy = ((pts[:, 1] - min_y) * resolution).astype(int) + padding
        return ix, iy

    ix, iy = to_img_coords(points)
    
    # 边界检查
    ix = np.clip(ix, 0, w_px - 1)
    iy = np.clip(iy, 0, h_px - 1)
    
    # 在图像上画点
    grid_img[iy, ix] = 255
    
    # 2. 形态学膨胀 (连接邻近点，使竹子环成为一个整体)
    # 假如扫描有点稀疏，这一步很重要
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated = cv2.dilate(grid_img, kernel, iterations=2)
    
    # 3. 寻找轮廓 (连通域)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return points # 没找到轮廓，返回原数据
    
    # 4. 找到面积最大的轮廓 (即竹子主体)
    max_cnt = max(contours, key=cv2.contourArea)
    
    # 5. 创建掩膜
    mask_img = np.zeros_like(grid_img)
    cv2.drawContours(mask_img, [max_cnt], -1, 255, thickness=cv2.FILLED)
    
    # 稍微膨胀掩膜，以免边缘的点被误删
    mask_img = cv2.dilate(mask_img, kernel, iterations=1)
    
    # 6. 过滤原始点
    # 检查原始点的坐标映射回图像后，是否在掩膜白色区域内
    is_valid = mask_img[iy, ix] > 0
    
    cleaned_points = points[is_valid]
    print(f"去噪完成：移除 {len(points) - len(cleaned_points)} 个噪点。")
    
    return cleaned_points

# ==========================================
# 3. 竹子几何分析与可视化
# ==========================================
def analyze_and_visualize_bamboo(point_cloud, output_dir):
    # >>>>> 第一步：调用去噪函数 <<<<<
    point_cloud = remove_isolated_noise(point_cloud, resolution=5.0) # 5像素/mm分辨率
    
    if len(point_cloud) == 0:
        print("无有效点云数据。")
        return

    # 1. 计算几何中心
    center_x = np.mean(point_cloud[:, 0])
    center_y = np.mean(point_cloud[:, 1])
    print(f"Calculated Center: ({center_x:.2f}, {center_y:.2f})")

    # 2. 预计算极坐标
    dx = point_cloud[:, 0] - center_x
    dy = point_cloud[:, 1] - center_y
    distances = np.sqrt(dx**2 + dy**2)
    angles = np.arctan2(dy, dx)

    # 3. 32个方向射线探测
    NUM_RAYS = 32
    step_angle = 2 * np.pi / NUM_RAYS 
    max_detect_dist = 100.0
    
    results = [] 
    inner_poly_pts = [] 
    outer_poly_pts = [] 
    ray_lines_data = [] 

    print(f"\nStarting {NUM_RAYS}-Direction Analysis...")
    
    for i in range(NUM_RAYS):
        target_angle_rad = i * step_angle
        normalized_target = target_angle_rad
        if normalized_target > np.pi:
            normalized_target -= 2 * np.pi
            
        half_wedge = step_angle / 2
        angle_diff = np.abs(np.arctan2(np.sin(angles - normalized_target), np.cos(angles - normalized_target)))
        mask = (angle_diff <= half_wedge) & (distances <= max_detect_dist)
        
        sector_dists = distances[mask]
        
        inner_r = np.nan
        thickness = np.nan
        outer_r = np.nan
        
        if len(sector_dists) > 5: 
            sorted_dists = np.sort(sector_dists)
            
            # 即使去噪了，保留前后1.5%剔除逻辑作为双重保险
            n_points = len(sorted_dists)
            trim_count = int(n_points * 0.015)
            if trim_count > 0:
                valid_dists = sorted_dists[trim_count : -trim_count]
            else:
                valid_dists = sorted_dists
            
            if len(valid_dists) > 0:
                inner_r = valid_dists[0]
                outer_r = valid_dists[-1]
                thickness = outer_r - inner_r
        
        deg = np.degrees(target_angle_rad)
        results.append({
            'index': i+1,
            'angle_deg': deg,
            'inner_radius': inner_r,
            'wall_thickness': thickness
        })

        ray_end_x = center_x + max_detect_dist * np.cos(target_angle_rad)
        ray_end_y = center_y + max_detect_dist * np.sin(target_angle_rad)
        
        label_text = ""
        if not np.isnan(inner_r):
            # 几何修正点
            in_x = center_x + inner_r * np.cos(target_angle_rad)
            in_y = center_y + inner_r * np.sin(target_angle_rad)
            inner_poly_pts.append((in_x, in_y))
            
            out_x = center_x + outer_r * np.cos(target_angle_rad)
            out_y = center_y + outer_r * np.sin(target_angle_rad)
            outer_poly_pts.append((out_x, out_y))
            
            label_text = f"R:{inner_r:.1f}\nT:{thickness:.1f}"

        ray_lines_data.append({
            'start': (center_x, center_y),
            'end': (ray_end_x, ray_end_y),
            'angle_deg': deg,
            'text': label_text
        })

    # 4. 输出数据
    with open(OUTPUT_DATA_FILE, 'w') as f:
        f.write("Direction_Index,Angle_Deg,Inner_Radius(mm),Wall_Thickness(mm)\n")
        for res in results:
            r_val = f"{res['inner_radius']:.3f}" if not np.isnan(res['inner_radius']) else "N/A"
            t_val = f"{res['wall_thickness']:.3f}" if not np.isnan(res['wall_thickness']) else "N/A"
            f.write(f"{res['index']},{res['angle_deg']:.1f},{r_val},{t_val}\n")

    # 5. 画图
    plt.figure(figsize=(14, 14))
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    
    plt.plot(point_cloud[:, 0], point_cloud[:, 1], 'r.', markersize=0.5, alpha=0.15, label='Cleaned Point Cloud')
    plt.plot(center_x, center_y, 'go', markersize=8, zorder=10, label='Center')
    
    for item in ray_lines_data:
        sx, sy = item['start']
        ex, ey = item['end']
        
        plt.plot([sx, ex], [sy, ey], color='gray', linestyle='--', linewidth=0.5, alpha=0.4)
        
        # 角度文字
        angle_text_x = sx + 105 * np.cos(np.radians(item['angle_deg']))
        angle_text_y = sy + 105 * np.sin(np.radians(item['angle_deg']))
        rot_angle = item['angle_deg']
        if 90 < rot_angle <= 270: rot_angle += 180
        plt.text(angle_text_x, angle_text_y, f"{int(item['angle_deg'])}°", 
                 fontsize=7, color='black', ha='center', va='center', rotation=rot_angle)

        # 数据文字
        if item['text']:
            t_x = sx + 115 * np.cos(np.radians(item['angle_deg']))
            t_y = sy + 115 * np.sin(np.radians(item['angle_deg']))
            plt.text(t_x, t_y, item['text'], 
                     fontsize=6, color='blue', ha='center', va='center', rotation=rot_angle)

    if len(inner_poly_pts) > 0:
        pts_in = np.array(inner_poly_pts + [inner_poly_pts[0]])
        plt.plot(pts_in[:, 0], pts_in[:, 1], 'b-', linewidth=1.5, marker='.', markersize=4, label='Inner Wall')
        
    if len(outer_poly_pts) > 0:
        pts_out = np.array(outer_poly_pts + [outer_poly_pts[0]])
        plt.plot(pts_out[:, 0], pts_out[:, 1], color='purple', linestyle='-', linewidth=1.5, marker='.', markersize=4, label='Outer Wall')

    plt.title(f"Bamboo Analysis (Cleaned & 32 Directions)\nCenter: ({center_x:.1f}, {center_y:.1f})", fontsize=14)
    plt.xlabel("X (mm)")
    plt.ylabel("Y (mm)")
    plt.legend(loc='upper right')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.xlim(center_x - 130, center_x + 130)
    plt.ylim(center_y - 130, center_y + 130)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE_FILE, dpi=300)
    print(f"Image saved: {OUTPUT_IMAGE_FILE}")
    plt.show()

# ==========================================
# 4. 主程序
# ==========================================
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"❌ 错误：无法打开视频 {VIDEO_PATH}")
        exit()

    print("🎥 视频处理开始...")
    frame_count = 0
    H_matrix = None 
    combined_point_cloud = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break 

        has_laser, laser_mask = detect_laser(frame)
        if not has_laser:
            frame_count += 1
            continue
        
        if H_matrix is None:
            H_matrix = get_homography_manual(frame)
            if H_matrix is None: break

        if H_matrix is not None:
            points = extract_and_map_laser_points(laser_mask, H_matrix)
            if len(points) > 0:
                combined_point_cloud.append(points)
                if frame_count % 10 == 0:
                    print(f"-> 帧 {frame_count}: 提取 {len(points)} 点")
        frame_count += 1

    cap.release()
    print("\n✅ 扫描结束。")

    if combined_point_cloud:
        final_point_cloud = np.concatenate(combined_point_cloud, axis=0)
        np.save(OUTPUT_POINTS_FILE, final_point_cloud)
        # 执行分析
        analyze_and_visualize_bamboo(final_point_cloud, OUTPUT_DIR)
    else:
        print("❌ 未提取到点云。")