# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 22:07:46 2024

@author: cengh
"""
import cv2 
import time

# Kameradan video akışını almak için VideoCapture nesnesi oluşturulur.
cap = cv2.VideoCapture(0)

# Kamera özellikleri alınır ve yazdırılır: genişlik ve yükseklik
print("Kamera genişliği: ", cap.get(3))
print("Kamera yüksekliği: ", cap.get(4))

# Kamera açılamazsa hata mesajı verilir.
if not cap.isOpened():
    print("Video akışı açılamadı.")

# Sonsuz bir döngü başlatılır. Bu döngüde video akışı alınır ve gösterilir.
while True:
    # Video akışından bir kare okunur (ret), ve o kareye ait veri (frame) alınır.
    ret, frame = cap.read()

    # Eğer kare başarıyla okunduysa devam edilir, aksi takdirde döngüden çıkılır.
    if ret:
        # Kameradan gelen her kare arasında kısa bir bekleme süresi eklenir (10 ms).
        time.sleep(0.01)
        
        # Kameradan gelen kare, bir pencerede gösterilir. Pencere adı: "Frame".
        cv2.imshow("Frame", frame)
    else:
        break
    
    # 'q' tuşuna basıldığında döngüden çıkılır.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Döngü bittiğinde, video akışı serbest bırakılır ve pencere kapatılır.
cap.release()
cv2.destroyAllWindows()
