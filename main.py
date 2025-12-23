import cv2
import os
import sys
import numpy as np
from ultralytics import YOLO
from glob import glob
from PIL import Image, ImageDraw, ImageFont

# 添加plate_yolov8路径到系统路径
sys.path.insert(0, r'C:\Users\leoli\Desktop\plate_yolov8')

# 全局变量：模板字典
TEMPLATES = {
    'chinese': {},  # 省份简称
    'char': {}      # 字母和数字
}

def load_templates():
    """加载所有字符模板"""
    print("正在加载字符模板...")
    
    # 加载中文省份简称模板
    chinese_dir = 'chars/annCh'
    if os.path.exists(chinese_dir):
        for province_dir in os.listdir(chinese_dir):
            province_path = os.path.join(chinese_dir, province_dir)
            if os.path.isdir(province_path):
                # 提取省份名称（zh_cuan -> 川）
                province_name = province_dir.replace('zh_', '')
                name_map = {
                    'cuan': '川', 'e': '鄂', 'gan': '赣', 'gui': '贵', 
                    'hei': '黑', 'hu': '沪', 'ji': '冀', 'jin': '津',
                    'jing': '京', 'jl': '吉', 'liao': '辽', 'lu': '鲁',
                    'meng': '蒙', 'min': '闽', 'ning': '宁', 'qing': '青',
                    'qiong': '琼', 'shan': '陕', 'su': '苏', 'sx': '晋',
                    'wan': '皖', 'xiang': '湘', 'xin': '新', 'yu': '豫',
                    'yu1': '渝', 'yue': '粤', 'yun': '云', 'zang': '藏',
                    'zhe': '浙', 'gan1': '赣', 'gui1': '桂'
                }
                char_label = name_map.get(province_name, province_name)
                
                # 读取该省份的所有模板图片（改进：加载多张模板）
                template_files = glob(os.path.join(province_path, '*.jpg'))
                if template_files and char_label not in TEMPLATES['chinese']:
                    # 加载多张模板（最多10张）以提高识别率
                    templates_list = []
                    for template_file in template_files[:10]:
                        template_img = cv2.imread(template_file, cv2.IMREAD_GRAYSCALE)
                        if template_img is not None:
                            # 汉字使用更大的尺寸保留细节（30x30）
                            template_img = cv2.resize(template_img, (30, 30))
                            # 应用形态学处理增强特征
                            template_img = cv2.adaptiveThreshold(
                                template_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 11, 2
                            )
                            templates_list.append(template_img)
                    
                    if templates_list:
                        TEMPLATES['chinese'][char_label] = templates_list
    
    # 加载字母和数字模板
    char_dir = 'chars/annGray'
    if os.path.exists(char_dir):
        for char_folder in os.listdir(char_dir):
            char_path = os.path.join(char_dir, char_folder)
            if os.path.isdir(char_path):
                # 文件夹名就是字符标签
                char_label = char_folder
                
                # 读取该字符的所有模板图片（改进：加载多张模板）
                template_files = glob(os.path.join(char_path, '*.jpg'))
                if template_files:
                    # 加载多张模板（最多5张）
                    templates_list = []
                    for template_file in template_files[:5]:
                        template_img = cv2.imread(template_file, cv2.IMREAD_GRAYSCALE)
                        if template_img is not None:
                            # 字母数字使用20x20
                            template_img = cv2.resize(template_img, (20, 20))
                            # 二值化处理
                            _, template_img = cv2.threshold(
                                template_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
                            )
                            templates_list.append(template_img)
                    
                    if templates_list:
                        TEMPLATES['char'][char_label] = templates_list
    
    print(f"  已加载 {len(TEMPLATES['chinese'])} 个省份模板")
    print(f"  已加载 {len(TEMPLATES['char'])} 个字符模板")
    print()


def recognize_char(char_img, position):
    """
    使用模板匹配识别单个字符
    
    参数:
        char_img: 字符图像
        position: 字符位置 (0=省份简称, 1=字母, 2-6=字母或数字)
    
    返回:
        识别的字符
    """
    if char_img is None or char_img.size == 0:
        return '?'
    
    # 转换为灰度图
    if len(char_img.shape) == 3:
        gray = cv2.cvtColor(char_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = char_img.copy()
    
    # 根据位置选择模板库和处理尺寸
    if position == 0:
        # 第一个字符：省份简称（使用更大尺寸）
        templates = TEMPLATES['chinese']
        target_size = (30, 30)
        # 汉字使用自适应阈值处理
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
    else:
        # 其他字符：字母和数字
        templates = TEMPLATES['char']
        target_size = (20, 20)
        # 字母数字使用OTSU二值化
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 调整大小
    resized = cv2.resize(binary, target_size)
    
    if not templates:
        return '?'
    
    # 模板匹配（改进：对每个字符的多个模板取最高分）
    best_score = -1
    best_char = '?'
    
    for char_label, template_list in templates.items():
        # 遍历该字符的所有模板
        max_score_for_char = -1
        
        for template in template_list:
            # 使用多种匹配方法并综合评分
            # 方法1: 归一化相关系数
            result1 = cv2.matchTemplate(resized, template, cv2.TM_CCOEFF_NORMED)
            score1 = result1[0][0]
            
            # 方法2: 归一化平方差（取反使得越小越好变为越大越好）
            result2 = cv2.matchTemplate(resized, template, cv2.TM_SQDIFF_NORMED)
            score2 = 1 - result2[0][0]
            
            # 综合评分（加权平均）
            combined_score = 0.7 * score1 + 0.3 * score2
            
            if combined_score > max_score_for_char:
                max_score_for_char = combined_score
        
        # 取该字符所有模板的最高分
        if max_score_for_char > best_score:
            best_score = max_score_for_char
            best_char = char_label
    
    return best_char

def correct_plate_tilt(plate_img):
    """
    校正倾斜的车牌图像并二次裁剪
    
    参数:
        plate_img: 车牌图像
        
    返回:
        corrected_img: 校正后的车牌图像
    """
    if plate_img is None or plate_img.size == 0:
        return plate_img
    
    h, w = plate_img.shape[:2]
        
        # 转换为灰度图
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY) if len(plate_img.shape) == 3 else plate_img
        
        # 二值化
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 边缘检测
    edges = cv2.Canny(binary, 50, 150, apertureSize=3)
        
        # 使用霍夫变换检测直线
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=w*0.3, maxLineGap=10)
        
    if lines is not None and len(lines) > 0:
            # 计算所有直线的角度
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            # 只考虑接近水平的线（-45到45度）
            if -45 < angle < 45:
                angles.append(angle)
            
        if angles:
            # 使用中位数角度作为倾斜角度
            tilt_angle = np.median(angles)
                
            # 如果倾斜角度很小，不需要校正
            if abs(tilt_angle) < 0.5:
                    return plate_img
                
            # 计算旋转矩阵
            center = (w // 2, h // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, tilt_angle, 1.0)
                
            # 计算旋转后的图像尺寸
            cos = np.abs(rotation_matrix[0, 0])
            sin = np.abs(rotation_matrix[0, 1])
            new_w = int((h * sin) + (w * cos))
            new_h = int((h * cos) + (w * sin))
                
            # 调整旋转矩阵以适应新尺寸
            rotation_matrix[0, 2] += (new_w / 2) - center[0]
            rotation_matrix[1, 2] += (new_h / 2) - center[1]
                
            # 执行旋转
            rotated = cv2.warpAffine(plate_img, rotation_matrix, (new_w, new_h), 
                                        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
                
            # 二次裁剪：去除旋转后的黑边
            gray_rotated = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY) if len(rotated.shape) == 3 else rotated
            _, thresh = cv2.threshold(gray_rotated, 10, 255, cv2.THRESH_BINARY)
                
            # 找到非零像素的边界
            coords = cv2.findNonZero(thresh)
            if coords is not None:
                x, y, w_crop, h_crop = cv2.boundingRect(coords)
                # 添加小的边距
                margin = 2
                x = max(0, x - margin)
                y = max(0, y - margin)
                w_crop = min(new_w - x, w_crop + 2 * margin)
                h_crop = min(new_h - y, h_crop + 2 * margin)
                    
                cropped = rotated[y:y+h_crop, x:x+w_crop]
                return cropped
                
            return rotated
        
        # 如果没有检测到足够的直线，尝试使用轮廓方法
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # 找到最大的轮廓
            largest_contour = max(contours, key=cv2.contourArea)
            
            # 获取最小外接矩形
            rect = cv2.minAreaRect(largest_contour)
            angle = rect[2]
            
            # 调整角度
            if angle < -45:
                angle += 90
            elif angle > 45:
                angle -= 90
            
            if abs(angle) > 0.5:
                # 旋转图像
                center = (w // 2, h // 2)
                rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                
                cos = np.abs(rotation_matrix[0, 0])
                sin = np.abs(rotation_matrix[0, 1])
                new_w = int((h * sin) + (w * cos))
                new_h = int((h * cos) + (w * sin))
                
                rotation_matrix[0, 2] += (new_w / 2) - center[0]
                rotation_matrix[1, 2] += (new_h / 2) - center[1]
                
                rotated = cv2.warpAffine(plate_img, rotation_matrix, (new_w, new_h),
                                        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
                
                # 二次裁剪
                gray_rotated = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY) if len(rotated.shape) == 3 else rotated
                _, thresh = cv2.threshold(gray_rotated, 10, 255, cv2.THRESH_BINARY)
                coords = cv2.findNonZero(thresh)
                if coords is not None:
                    x, y, w_crop, h_crop = cv2.boundingRect(coords)
                    margin = 2
                    x = max(0, x - margin)
                    y = max(0, y - margin)
                    w_crop = min(new_w - x, w_crop + 2 * margin)
                    h_crop = min(new_h - y, h_crop + 2 * margin)
                    cropped = rotated[y:y+h_crop, x:x+w_crop]
                    return cropped
                
                return rotated
    return plate_img


def segment_characters(plate_img, output_path):
    """
    字符分割函数 - 基于行投影去除上下边框，再进行列投影分割
    参考CSDN博客方法
    """
    if plate_img is None or plate_img.size == 0:
        return []
    
    try:
        # 1. Resize到固定尺寸 (320, 100)
        img_resized = cv2.resize(plate_img, (320, 100))
        
        # 2. 灰度化和二值化
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 3. 水平投影 - 去除上下边框
        row_histogram = np.sum(binary, axis=1)  # 每行像素和
        row_min = np.min(row_histogram)
        row_average = np.sum(row_histogram) / binary.shape[0]
        row_threshold = (row_min + row_average) / 2
        
        # 找波峰 - 确定上下边界
        def find_waves(threshold, histogram):
            up_point = -1
            is_peak = False
            if histogram[0] > threshold:
                up_point = 0
                is_peak = True
            wave_peaks = []
            for i, x in enumerate(histogram):
                if is_peak and x < threshold:
                    if i - up_point > 2:
                        is_peak = False
                        wave_peaks.append((up_point, i))
                elif not is_peak and x >= threshold:
                    is_peak = True
                    up_point = i
            if is_peak and up_point != -1 and i - up_point > 4:
                wave_peaks.append((up_point, i))
            return wave_peaks
        
        wave_peaks = find_waves(row_threshold, row_histogram)
        
        # 选择跨度最大的波峰作为车牌主体区域
        if not wave_peaks:
            return []
        
        wave_span = 0
        selected_wave = wave_peaks[0]
        for wave_peak in wave_peaks:
            span = wave_peak[1] - wave_peak[0]
            if span > wave_span:
                wave_span = span
                selected_wave = wave_peak
        
        # 裁剪上下边框
        binary_cropped = binary[selected_wave[0]:selected_wave[1], :]
        
        # 4. 垂直投影 - 分割字符
        height, width = binary_cropped.shape
        white_list = []  # 记录每列白色像素总数
        black_list = []  # 记录每列黑色像素总数
        white_max = 0
        black_max = 0
        
        for i in range(width):
            line_white = 0
            line_black = 0
            for j in range(height):
                if binary_cropped[j][i] == 255:
                    line_white += 1
                else:
                    line_black += 1
            white_max = max(white_max, line_white)
            black_max = max(black_max, line_black)
            white_list.append(line_white)
            black_list.append(line_black)
        
        # 判断是黑底白字还是白底黑字
        is_black_bg = True  # True表示黑底白字
        if black_max < white_max:
            is_black_bg = False
        
        # 5. 查找字符边界
        def find_end(start, is_black_bg, black_list, white_list, width, black_max, white_max):
            end = start + 1
            for m in range(start + 1, width - 1):
                if (black_list[m] if is_black_bg else white_list[m]) > (0.95 * black_max if is_black_bg else 0.95 * white_max):
                    end = m
                    break
            return end
        
        # 分割字符
        char_bboxes = []
        n = 1
        while n < width - 2:
            n += 1
            # 检测字符起始位置
            if (white_list[n] if is_black_bg else black_list[n]) > (0.05 * white_max if is_black_bg else 0.05 * black_max):
                start = n
                end = find_end(start, is_black_bg, black_list, white_list, width, black_max, white_max)
                n = end
                
                # 初步检测，只做基本的宽度检查，不过滤
                char_width = end - start
                
                if char_width > 3 and char_width < width * 0.5:
                    # 映射回原图坐标
                    x1 = start
                    y1 = selected_wave[0]
                    x2 = end
                    y2 = selected_wave[1]
                    char_bboxes.append([x1, y1, x2, y2])
        
        # 6. 合并过近的字符（解决左右结构汉字被分割的问题）
        if len(char_bboxes) > 0:
            merged_bboxes = []
            i = 0
            
            while i < len(char_bboxes):
                x1, y1, x2, y2 = char_bboxes[i]
                
                # 检查是否需要与下一个字符合并
                while i + 1 < len(char_bboxes):
                    next_x1, next_y1, next_x2, next_y2 = char_bboxes[i + 1]
                    
                    gap = next_x1 - x2  # 间距
                    current_width = x2 - x1
                    next_width = next_x2 - next_x1
                    
                    # 合并条件（更加保守）：
                    # 只合并间距非常小且字符非常窄的情况（明显的错误分割）
                    should_merge = False
                    
                    # 条件1: 间距极小（小于3像素）且至少有一个字符很窄
                    if gap < 3 and (current_width < 10 or next_width < 10):
                        should_merge = True
                    
                    # 条件2: 两个字符都非常窄（可能是一个汉字被分成两部分）
                    if current_width < 8 and next_width < 8 and gap < 8:
                        should_merge = True
                    
                    # 防止过度合并：
                    # 1. 合并后宽度不能太大
                    merged_width = next_x2 - x1
                    if merged_width > width * 0.25:  # 单个字符不超过25%宽度
                        should_merge = False
                    
                    # 2. 如果当前已经是正常宽度的字符，不要合并
                    if current_width > 15 and gap > 2:
                        should_merge = False
                    
                    if should_merge:
                        x2 = next_x2  # 扩展右边界
                        i += 1
                    else:
                        break
                
                merged_bboxes.append((x1, y1, x2, y2))
                i += 1
            
            char_bboxes = merged_bboxes
        
        # 7. 打分策略：对每个字符区域打分，只保留分数最高的7个
        scored_bboxes = []
        plate_height = selected_wave[1] - selected_wave[0]
        
        for x1, y1, x2, y2 in char_bboxes:
            char_width = x2 - x1
            char_height = y2 - y1
            
            # 基础过滤：过滤掉明显不合理的区域
            if char_width < 5 or char_height < plate_height * 0.2:
                continue
            
            score = 0
            
            # 1. 宽度得分（10-40像素为最佳）
            if 15 <= char_width <= 35:
                score += 30
            elif 10 <= char_width <= 45:
                score += 20
            elif char_width > 5:
                score += 10
            
            # 2. 高度得分（占车牌高度60%-90%为最佳）
            height_ratio = char_height / plate_height
            if 0.6 <= height_ratio <= 0.9:
                score += 30
            elif 0.4 <= height_ratio <= 0.95:
                score += 20
            elif height_ratio > 0.3:
                score += 10
            
            # 3. 宽高比得分（0.4-1.5为最佳，汉字较宽，字母数字较窄）
            aspect_ratio = char_width / char_height if char_height > 0 else 0
            if 0.4 <= aspect_ratio <= 1.5:
                score += 25
            elif 0.25 <= aspect_ratio <= 2.0:
                score += 15
            elif aspect_ratio > 0.15:
                score += 5
            
            # 4. 位置得分（靠近边缘扣分）
            distance_from_left = x1
            distance_from_right = width - x2
            min_distance = min(distance_from_left, distance_from_right)
            
            if min_distance > width * 0.08:
                score += 15
            elif min_distance > width * 0.05:
                score += 10
            elif min_distance > width * 0.03:
                score += 5
            # 非常靠近边缘（<3%）不加分，甚至扣分
            else:
                score -= 10
            
            scored_bboxes.append((score, x1, y1, x2, y2))
        
        # 按分数排序，取前7个
        scored_bboxes.sort(reverse=True, key=lambda x: x[0])
        char_bboxes = [(x1, y1, x2, y2) for score, x1, y1, x2, y2 in scored_bboxes[:7]]
        
        # 按x坐标从左到右排序
        char_bboxes.sort(key=lambda x: x[0])
        
        # 8. 字符识别
        recognized_chars = []
        for idx, (x1, y1, x2, y2) in enumerate(char_bboxes):
            # 提取字符图像
            char_img = img_resized[y1:y2, x1:x2]
            
            # 识别字符
            char = recognize_char(char_img, idx)
            recognized_chars.append(char)
        
        # 拼接车牌号
        plate_number = ''.join(recognized_chars)
        
        # 9. 绘制分割结果和识别结果
        vis_img = img_resized.copy()
        
        # 绘制边界框
        for idx, (x1, y1, x2, y2) in enumerate(char_bboxes):
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # 使用PIL绘制中文字符
        # 转换为PIL图像
        vis_img_pil = Image.fromarray(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(vis_img_pil)
        
        # 尝试加载中文字体，如果失败则使用默认字体
        try:
            # Windows系统中文字体路径
            font = ImageFont.truetype("C:/Windows/Fonts/simhei.ttf", 20)
        except:
            try:
                font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 20)
            except:
                font = ImageFont.load_default()
        
        # 在每个字符框上方绘制识别结果（蓝色）
        for idx, (x1, y1, x2, y2) in enumerate(char_bboxes):
            char_label = recognized_chars[idx] if idx < len(recognized_chars) else '?'
            # PIL使用RGB颜色，蓝色是(0, 0, 255)
            draw.text((x1, max(0, y1-25)), char_label, fill=(0, 0, 255), font=font)
        
        # 转换回OpenCV格式
        vis_img = cv2.cvtColor(np.array(vis_img_pil), cv2.COLOR_RGB2BGR)
        
        vis_path = output_path + "_segmentation.jpg"
        cv2.imwrite(vis_path, vis_img)
        
        # 打印识别结果
        print(f"    识别结果: {plate_number}")
        
        return recognized_chars
    
    except Exception as e:
        print(f"    警告: 字符分割失败 - {e}")
        import traceback
        traceback.print_exc()
        return []


def main():
    # 加载字符模板
    load_templates()
    
    # 加载YOLOv8模型
    model_path = r'C:\Users\leoli\Desktop\plate_yolov8\weights\yolov8s.pt'
    
    model = YOLO(model_path)
    print("模型加载成功！\n")
    
    # 设置输入输出路径
    imgs_folder = "imgs"
    output_folder = "output"
    
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)
    
    # 检查输入文件夹
    if not os.path.exists(imgs_folder):
        print(f"错误: 图片文件夹 '{imgs_folder}' 不存在")
        print(f"请在项目根目录创建 '{imgs_folder}' 文件夹并放入待识别的车牌图片")
        return
    
    # 获取所有图片文件
    image_files = [f for f in os.listdir(imgs_folder) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    if not image_files:
        print(f"警告: 在 '{imgs_folder}' 文件夹中没有找到图片文件")
        print("支持的格式: .jpg, .jpeg, .png, .bmp")
        return
    
    print(f"找到 {len(image_files)} 张图片待处理\n")
    
    # 处理每张图片
    total_plates = 0
    for idx, img_file in enumerate(image_files, 1):
        print(f"[{idx}/{len(image_files)}] 处理: {img_file}")
        
        # 读取图片
        img_path = os.path.join(imgs_folder, img_file)
        image = cv2.imread(img_path)
        
        if image is None:
            print(f"  ✗ 无法读取图片")
            continue
        
        print(f"  图片尺寸: {image.shape[1]}x{image.shape[0]}")
        
        # 使用YOLOv8进行车牌检测
        results = model(image, conf=0.3, iou=0.5)
        
        # 处理检测结果
        plate_count = 0
        result_image = image.copy()
        
        for result in results:
            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                plate_count = len(boxes)
                
                for i, box in enumerate(boxes):
                    # 获取边界框坐标
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # 获取置信度
                    conf = float(box.conf[0].cpu().numpy())
                    
                    # 绘制边界框
                    cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # 添加标签
                    label = f"Plate {i+1}: {conf:.2f}"
                    cv2.putText(result_image, label, (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    # 裁剪车牌区域
                    plate_img = image[y1:y2, x1:x2]
                    
                    # 倾斜校正和二次裁剪
                    corrected_plate = correct_plate_tilt(plate_img)
                    
                    # 保存校正后的车牌
                    base_name = os.path.splitext(img_file)[0]
                    plate_filename = f"plate_{base_name}_{i+1}"
                    plate_path = os.path.join(output_folder, plate_filename + ".jpg")
                    cv2.imwrite(plate_path, corrected_plate)
                    
                    print(f"    车牌 {i+1} (置信度: {conf:.2f}) → {plate_path}")
                    
                    # 字符分割
                    segment_output_path = os.path.join(output_folder, plate_filename)
                    segment_characters(corrected_plate, segment_output_path)
        
        if plate_count == 0:
            print(f"  ✗ 未检测到车牌")
        else:
            print(f"  ✓ 检测到 {plate_count} 个车牌区域")
            total_plates += plate_count
        
        # 不再保存标注后的完整图片，只保存裁剪的车牌
        # output_path = os.path.join(output_folder, f"detected_{img_file}")
        # cv2.imwrite(output_path, result_image)
        # print(f"  → 结果已保存: {output_path}\n")
        print()
    
    # 总结
    print("="*50)
    print(f"处理完成!")
    print(f"共处理 {len(image_files)} 张图片")
    print(f"检测到 {total_plates} 个车牌")
    print(f"结果保存在 '{output_folder}' 文件夹中")
    print("="*50)


if __name__ == "__main__":
    main()
