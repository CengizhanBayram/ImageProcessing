import cv2
import matplotlib.pyplot as plt

# 1. Resmi Yükleyin
hakkı_bulut = cv2.imread("hakkı_bulut.jpg", 0)

# 2. Orijinal Resmi Gösterin
plt.figure(), plt.imshow(hakkı_bulut, cmap="gray"), plt.axis("off")
plt.show()  # Resmi göstermek için plt.show() ekleyin

# 3. Yüz Cascade'ini Tanımlayın
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# 4. Yüzleri Algılayın
face_rect = face_cascade.detectMultiScale(hakkı_bulut)

# 5. Kopyalanmış Resim Üzerine Dikdörtgenler Çizin
hakkı_bulut_with_rectangles = hakkı_bulut.copy()
for (x, y, w, h) in face_rect:
    cv2.rectangle(hakkı_bulut_with_rectangles, (x, y), (x + w, y + h), (255, 255, 255), 10)

cap = cv2.VideoCapture(0)
while True :
    ret, frame = cap.read()
    if ret:
        face_rect = face_cascade.detectMultiScale(frame, minNeighbors=7)
        for (x, y, w, h) in face_rect:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 10)
        cv2.imshow("face detect frame ", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()     
