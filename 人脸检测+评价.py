import cv2
import matplotlib.pyplot as plt
import time
import psutil
import numpy as np
from collections import deque

# 静态图片人脸检测与评估

def evaluate_image_detection():
    # 加载图片
    img = cv2.imread("image/Musk.jpg")
    if img is None:
        print("错误：无法加载图片！")
        return

    # 创建评估结果字典
    evaluation = {
        'face_detected': 0,
        'expected_faces': 1,  # 假设图片中有1个人脸
        'eyes_detected': 0,
        'expected_eyes': 2,  # 假设有2只眼睛
        'false_positives': 0,
        'processing_time': 0,
        'face_sizes': [],
        'eye_sizes': []
    }

    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 加载分类器
    face_cas = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    eyes_cas = cv2.CascadeClassifier("haarcascade_eye.xml")

    # 记录开始时间
    start_time = time.time()

    # 人脸检测
    face_rects = face_cas.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=8, minSize=(50, 50))
    evaluation['face_detected'] = len(face_rects)

    # 绘制人脸，检测眼睛
    for facerect in face_rects:
        x, y, w, h = facerect
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
        roi_color = img[y:y + h, x:x + w]
        roi_gray = gray[y:y + h, x:x + w]

        # 记录人脸大小
        evaluation['face_sizes'].append((w, h))

        # 眼睛检测
        eyes = eyes_cas.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
        evaluation['eyes_detected'] += len(eyes)

        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 3)
            # 记录眼睛大小
            evaluation['eye_sizes'].append((ew, eh))

    # 计算处理时间
    evaluation['processing_time'] = time.time() - start_time

    # 计算误检率（如果有多个检测框）
    if evaluation['face_detected'] > evaluation['expected_faces']:
        evaluation['false_positives'] = evaluation['face_detected'] - evaluation['expected_faces']

    # 显示图片
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('人脸检测结果')
    plt.axis('off')

    # 在图片上添加评估结果
    result_text = f"人脸召回率: {min(evaluation['face_detected'] / evaluation['expected_faces'], 1.0) * 100:.1f}%\n"
    result_text += f"眼睛检出率: {min(evaluation['eyes_detected'] / evaluation['expected_eyes'], 1.0) * 100:.1f}%\n"
    result_text += f"处理时间: {evaluation['processing_time'] * 1000:.1f} ms"

    plt.figtext(0.5, 0.01, result_text, wrap=True, horizontalalignment='center', fontsize=12)
    plt.show()

    # 打印详细评估结果
    print("\n===== 静态图片检测评估 =====")
    print(f"检测到的人脸数: {evaluation['face_detected']}/{evaluation['expected_faces']}")
    print(f"检测到的眼睛数: {evaluation['eyes_detected']}/{evaluation['expected_eyes']}")
    print(f"人脸召回率: {min(evaluation['face_detected'] / evaluation['expected_faces'], 1.0) * 100:.1f}%")
    print(f"眼睛检出率: {min(evaluation['eyes_detected'] / evaluation['expected_eyes'], 1.0) * 100:.1f}%")
    print(
        f"人脸误检率: {evaluation['false_positives'] / evaluation['face_detected'] * 100 if evaluation['face_detected'] > 0 else 0:.1f}%")
    print(f"平均人脸大小: {np.mean(evaluation['face_sizes'], axis=0) if evaluation['face_sizes'] else (0, 0)}")
    print(f"平均眼睛大小: {np.mean(evaluation['eye_sizes'], axis=0) if evaluation['eye_sizes'] else (0, 0)}")
    print(f"处理时间: {evaluation['processing_time'] * 1000:.1f} ms")

    return evaluation


# ========================
# 实时视频人脸检测与评估
# ========================
def evaluate_realtime_detection():
    cap = cv2.VideoCapture(0)  # 摄像头输入
    if not cap.isOpened():
        print("错误：无法打开摄像头！")
        return

    # 加载分类器
    face_cas = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    eyes_cas = cv2.CascadeClassifier("haarcascade_eye.xml")

    # 初始化评估指标
    evaluation = {
        'frame_count': 0,
        'face_detections': 0,
        'eye_detections': 0,
        'false_positives': 0,
        'start_time': time.time(),
        'fps_history': deque(maxlen=30),  # 存储最近30帧的FPS
        'cpu_usage': [],
        'detection_times': [],
        'face_sizes': [],
        'eye_sizes': [],
        'last_fps': 0
    }

    # 创建性能监控窗口
    cv2.namedWindow("Real-time Detection", cv2.WINDOW_NORMAL)

    print("实时人脸检测中... 按'q'键退出")

    while True:
        # 记录帧开始时间
        frame_start = time.time()

        # 读取帧
        ret, frame = cap.read()
        if not ret:
            break

        # 转换为灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 人脸检测
        faces = face_cas.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))
        evaluation['face_detections'] += len(faces)

        # 检测并绘制结果
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]

            # 记录人脸大小
            evaluation['face_sizes'].append((w, h))

            # 眼睛检测
            eyes = eyes_cas.detectMultiScale(roi_gray)
            evaluation['eye_detections'] += len(eyes)

            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(frame, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (255, 0, 0), 2)
                # 记录眼睛大小
                evaluation['eye_sizes'].append((ew, eh))

        # 计算帧处理时间
        frame_time = time.time() - frame_start
        evaluation['detection_times'].append(frame_time)
        evaluation['frame_count'] += 1

        # 计算FPS
        current_fps = 1.0 / frame_time if frame_time > 0 else 0
        evaluation['fps_history'].append(current_fps)
        evaluation['last_fps'] = np.mean(evaluation['fps_history'])

        # 获取CPU使用率
        evaluation['cpu_usage'].append(psutil.cpu_percent())

        # 在帧上显示性能指标
        avg_face_size = np.mean(evaluation['face_sizes'], axis=0) if evaluation['face_sizes'] else (0, 0)
        avg_eye_size = np.mean(evaluation['eye_sizes'], axis=0) if evaluation['eye_sizes'] else (0, 0)

        # 性能指标显示
        cv2.putText(frame, f"FPS: {evaluation['last_fps']:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Faces: {len(faces)}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Eyes: {evaluation['eye_detections']}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"CPU: {evaluation['cpu_usage'][-1]:.1f}%", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Face Size: {avg_face_size[0]:.0f}x{avg_face_size[1]:.0f}", (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # 显示结果
        cv2.imshow("Real-time Detection", frame)

        # 退出条件
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()

    # 计算总体性能指标
    total_time = time.time() - evaluation['start_time']
    avg_fps = evaluation['frame_count'] / total_time
    avg_cpu = np.mean(evaluation['cpu_usage'])
    avg_detection_time = np.mean(evaluation['detection_times']) * 1000

    # 打印最终评估报告
    print("\n===== 实时视频检测评估报告 =====")
    print(f"总处理帧数: {evaluation['frame_count']}")
    print(f"总检测人脸数: {evaluation['face_detections']}")
    print(f"总检测眼睛数: {evaluation['eye_detections']}")
    print(f"平均FPS: {avg_fps:.1f}")
    print(f"平均CPU使用率: {avg_cpu:.1f}%")
    print(f"平均检测时间: {avg_detection_time:.1f} ms/帧")
    print(f"平均人脸大小: {np.mean(evaluation['face_sizes'], axis=0) if evaluation['face_sizes'] else (0, 0)}")
    print(f"平均眼睛大小: {np.mean(evaluation['eye_sizes'], axis=0) if evaluation['eye_sizes'] else (0, 0)}")

    # 绘制性能图表
    plt.figure(figsize=(12, 8))

    # FPS图表
    plt.subplot(2, 2, 1)
    fps_values = list(evaluation['fps_history'])
    plt.plot(fps_values, label='FPS')
    plt.axhline(y=avg_fps, color='r', linestyle='--', label=f'Avg FPS: {avg_fps:.1f}')
    plt.title('帧率变化 (FPS)')
    plt.xlabel('帧数')
    plt.ylabel('FPS')
    plt.legend()

    # CPU使用率图表
    plt.subplot(2, 2, 2)
    plt.plot(evaluation['cpu_usage'], label='CPU Usage')
    plt.axhline(y=avg_cpu, color='r', linestyle='--', label=f'Avg CPU: {avg_cpu:.1f}%')
    plt.title('CPU使用率变化')
    plt.xlabel('帧数')
    plt.ylabel('CPU (%)')
    plt.legend()

    # 人脸大小分布
    plt.subplot(2, 2, 3)
    if evaluation['face_sizes']:
        face_widths = [w for w, h in evaluation['face_sizes']]
        face_heights = [h for w, h in evaluation['face_sizes']]
        plt.hist(face_widths, bins=20, alpha=0.5, label='Width')
        plt.hist(face_heights, bins=20, alpha=0.5, label='Height')
        plt.title('人脸大小分布')
        plt.xlabel('像素')
        plt.ylabel('频率')
        plt.legend()

    # 眼睛大小分布
    plt.subplot(2, 2, 4)
    if evaluation['eye_sizes']:
        eye_widths = [w for w, h in evaluation['eye_sizes']]
        eye_heights = [h for w, h in evaluation['eye_sizes']]
        plt.hist(eye_widths, bins=20, alpha=0.5, label='Width')
        plt.hist(eye_heights, bins=20, alpha=0.5, label='Height')
        plt.title('眼睛大小分布')
        plt.xlabel('像素')
        plt.ylabel('频率')
        plt.legend()

    plt.tight_layout()
    plt.show()

    return evaluation


# ======================
# 主程序
# ======================
if __name__ == "__main__":
    # 执行静态图片检测与评估
    print("正在进行静态图片人脸检测...")
    image_results = evaluate_image_detection()

    # 执行实时视频检测与评估
    print("\n正在进行实时视频人脸检测...")
    video_results = evaluate_realtime_detection()

    # 综合性能报告
    print("\n===== 综合性能报告 =====")
    print(f"静态图片处理时间: {image_results['processing_time'] * 1000:.1f} ms")
    print(f"实时视频平均FPS: {np.mean(video_results['fps_history']):.1f}")
    print(f"实时视频平均CPU使用率: {np.mean(video_results['cpu_usage']):.1f}%")