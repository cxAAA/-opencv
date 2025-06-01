import cv2 as cv
import matplotlib.pyplot as plt
import os
import numpy as np
from collections import defaultdict

# 配置参数
SHOW_IMAGES = False  # 设为True显示每张图片，False则不显示
SHOW_HISTOGRAM = True  # 显示检测结果的直方图

# 初始化性能统计字典
performance_stats = defaultdict(list)
folder_stats = defaultdict(lambda: {'images': 0, 'faces': 0, 'failed': 0})

# 加载分类器
print("加载分类器...")
face_cas = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")
if face_cas.empty():
    face_cas = cv.CascadeClassifier("haarcascade_frontalface_default.xml")
if face_cas.empty():
    raise IOError("无法加载人脸级联分类器")

eyes_cas = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_eye.xml")
if eyes_cas.empty():
    eyes_cas = cv.CascadeClassifier("haarcascade_eye.xml")
if eyes_cas.empty():
    print("警告: 无法加载眼睛级联分类器，将跳过眼睛检测")

# 配置数据集路径
dataset_path = "train_faces"
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"数据集目录不存在: {dataset_path}")

# 获取所有文件夹
folders = [f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))]
print(f"找到 {len(folders)} 个文件夹，开始处理...")

# 遍历每个文件夹
for folder_idx, folder_name in enumerate(folders):
    folder_path = os.path.join(dataset_path, folder_name)

    # 遍历文件夹中的图片
    for img_file in os.listdir(folder_path):
        # 支持更多灰度图像格式
        if not img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.pgm', '.bmp', '.tiff')):
            continue

        img_path = os.path.join(folder_path, img_file)

        # 专门处理灰度图像
        img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)  # 强制以灰度模式读取
        if img is None:
            print(f"  加载失败: {img_path}")
            continue

        # 更新文件夹统计
        folder_stats[folder_name]['images'] += 1

        # 为灰度图像创建彩色版本用于可视化
        color_img = cv.cvtColor(img, cv.COLOR_GRAY2BGR) if len(img.shape) == 2 else img.copy()

        # 人脸检测 - 使用灰度图像
        face_rects = face_cas.detectMultiScale(
            img,
            scaleFactor=1.05,
            minNeighbors=5,
            minSize=(30, 30)
        )

        # 记录检测结果
        num_faces = len(face_rects)
        performance_stats['total_images'].append(img_path)
        performance_stats['detected_faces'].append(num_faces)
        performance_stats['folder'].append(folder_name)

        folder_stats[folder_name]['faces'] += num_faces
        if num_faces == 0:
            folder_stats[folder_name]['failed'] += 1

        # 在图像上绘制检测结果
        output_img = color_img.copy()
        for (x, y, w, h) in face_rects:
            cv.rectangle(output_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # 眼睛检测
            if not eyes_cas.empty():
                roi_gray = img[y:y + h, x:x + w]
                eyes = eyes_cas.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))
                for (ex, ey, ew, eh) in eyes:
                    # 在彩色图像上绘制眼睛
                    cv.rectangle(output_img, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (0, 0, 255), 2)


    print(
        f"文件夹 {folder_idx + 1}/{len(folders)}: {folder_name} 处理完成 - {folder_stats[folder_name]['images']}张图片, {folder_stats[folder_name]['faces']}张人脸")

# 性能评估
if performance_stats['total_images']:
    total_images = len(performance_stats['total_images'])
    total_faces_detected = sum(performance_stats['detected_faces'])
    avg_faces_per_image = total_faces_detected / total_images

    print("\n" + "=" * 50)
    print("人脸检测性能评估报告")
    print("=" * 50)
    print(f"数据集路径: {dataset_path}")
    print(f"文件夹数量: {len(folders)}")
    print(f"分析图像数量: {total_images}")
    print(f"检测到总人脸数: {total_faces_detected}")
    print(f"平均每张图像检测人脸数: {avg_faces_per_image:.2f}")

    # 检测失败图像分析
    zero_detections = [img for img, count in zip(performance_stats['total_images'],
                                                 performance_stats['detected_faces']) if count == 0]
    failure_rate = len(zero_detections) / total_images * 100

    print("\n检测失败分析:")
    print(f"检测失败图像数: {len(zero_detections)}")
    print(f"检测失败率: {failure_rate:.2f}%")

    # 按文件夹统计失败情况
    print("\n按文件夹统计失败率:")
    for folder, stats in folder_stats.items():
        if stats['images'] > 0:
            folder_failure_rate = stats['failed'] / stats['images'] * 100
            print(f" - {folder}: {stats['failed']}/{stats['images']} ({folder_failure_rate:.1f}%)")

    # 多脸检测分析
    multi_face_images = [(img, count, folder) for img, count, folder in zip(
        performance_stats['total_images'],
        performance_stats['detected_faces'],
        performance_stats['folder']) if count > 1]

    print("\n多人脸图像:")
    for img_path, count, folder in multi_face_images:
        print(f" - {folder}/{os.path.basename(img_path)}: {count} 张人脸")
    print(f"总计 {len(multi_face_images)} 张图像包含多个人脸")


    # 生成性能总结
    print("\n" + "=" * 50)
    print("性能总结")
    print("=" * 50)
    print(f"总图像数: {total_images}")
    print(f"检测到人脸: {total_faces_detected}")
    print(f"平均每张图像人脸数: {avg_faces_per_image:.2f}")
    print(f"检测失败图像数: {len(zero_detections)} ({failure_rate:.1f}%)")
    print(f"多人脸图像数: {len(multi_face_images)}")

    # 找出失败率最高的文件夹
    if folder_stats:  # 确保folder_stats不为空
        max_failure_folder = max(folder_stats.items(),
                                 key=lambda x: x[1]['failed'] / x[1]['images'] if x[1]['images'] > 0 else 0)
        folder_name, stats = max_failure_folder
        if stats['images'] > 0:
            failure_rate = stats['failed'] / stats['images'] * 100
            print(f"\n失败率最高的文件夹: {folder_name} ({failure_rate:.1f}%)")

else:
    print("未找到有效图像进行分析")



