import cv2 as cv
import matplotlib.pyplot as plt

# 1. 加载图像
img = cv.imread("image/kun.jpg")
if img is None:
    raise FileNotFoundError("图像未找到！")
plt.imshow(img[:, :, ::-1])
plt.title("Original Image")
plt.show()

# 2. 转换为灰度图
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# 3. 加载分类器（添加错误检查）
face_cas = cv.CascadeClassifier("haarcascade_frontalface_default.xml")
if face_cas.empty():
    raise ValueError("人脸分类器加载失败！")

eyes_cas = cv.CascadeClassifier("haarcascade_eye.xml")
if eyes_cas.empty():
    raise ValueError("眼睛分类器加载失败！")

# 4. 人脸检测（优化参数）
face_rects = face_cas.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(50, 50)
)

# 5. 检测并绘制结果
for (x, y, w, h) in face_rects:
    # 绘制人脸框
    cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # 提取人脸ROI
    roi_color = img[y:y + h, x:x + w]
    roi_gray = gray[y:y + h, x:x + w]

    # 眼睛检测（优化参数）
    eyes = eyes_cas.detectMultiScale(
        roi_gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(20, 20)
    )
    for (ex, ey, ew, eh) in eyes:
        cv.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)

# 6. 显示结果
plt.imshow(img[:, :, ::-1])
plt.title("Face and Eye Detection")
plt.show()