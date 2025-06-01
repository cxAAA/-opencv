import cv2 as cv
import matplotlib.pyplot as plt
#图片人脸检测
img=cv.imread("image/Musk.jpg")
# plt.imshow(img[:,:,::-1])
# plt.show()
gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
#实例化检测器
face_cas=cv.CascadeClassifier("haarcascade_frontalface_default.xml")
# face_cas.load('haarcascade_frontalface_default.xml')
eyes_cas=cv.CascadeClassifier("haarcascade_eye.xml")
# eyes_cas.load('haarcascade_eye.xml')
#人脸检测
face_rects=face_cas.detectMultiScale(gray,scaleFactor=1.05,minNeighbors=8,minSize=(50,50))
#绘制人脸，检测眼睛
for facerect in face_rects:
    x,y,w,h=facerect
    cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)
    roi_color=img[y:y+h,x:x+w]
    roi_gray=gray[y:y+h,x:x+w]
    eyes = eyes_cas.detectMultiScale(roi_gray,scaleFactor=1.1,minNeighbors=5,minSize=(50, 50))
    for (ex,ey,ew,eh) in eyes:
        cv.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),3)
plt.imshow(img[:,:,::-1])
plt.show()



#摄像头人脸检测
cap = cv.VideoCapture(0)  # 摄像头输入
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_cas.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))

    for (x, y, w, h) in faces:
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        eyes = eyes_cas.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv.rectangle(frame, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (255, 0, 0), 2)

    cv.imshow("Real-time Detection", frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()

# 评估代码
total_faces = 1  # 图片中实际人脸数
detected_faces = len(face_rects)  # 检测到的人脸数
detected_eyes = len(eyes)  # 检测到的眼睛数

face_recall = detected_faces / total_faces
eye_recall = detected_eyes / (2 * total_faces) if total_faces > 0 else 0

print(f"人脸召回率: {face_recall:.2%}")
print(f"眼睛检出率: {eye_recall:.2%}")



