import cv2
import numpy as np

# Renk uzaylarını tanımlama
HSV_HUE_RANGE = (30, 180)
HSV_SAT_RANGE = (20, 255)
HSV_VAL_RANGE = (30, 255)

# Takip edilen nesnenin ilk konumunu alma
def get_initial_position(frame):
    # ... (Kullanıcıdan ilk konumu alma kodunu buraya ekleyin)
    # Örnek kod:
    x, y = cv2.selectROI("Nesneyi Seçin", frame)
    return x, y

# Takip fonksiyonu
def track_object(frame, x, y, previous_frame):
    # Nesneyi HSV renk uzayında dönüştürme
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Renk aralığı maskesi oluşturma
    mask = cv2.inRange(hsv, HSV_HUE_RANGE, HSV_SAT_RANGE, HSV_VAL_RANGE)

    # Nesnenin arka planını çıkarma
    masked_frame = cv2.bitwiseAnd(frame, frame, mask=mask)

    # Nesnenin merkezini bulma
    mean, stddev = cv2.meanStdDev(masked_frame)
    center_x, center_y = int(mean[0]), int(mean[1])

    # Hareketi tahmin etme (Kalman filtresi gibi)
    # ... (Kalman filtresi kodunu buraya ekleyin)

    # Tahmin edilen ve gerçek konumları görselleştirme
    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
    cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

    # Görüntüleri gösterme
    cv2.imshow('Orijinal Görüntü', frame)
    cv2.imshow('Maskeli Görüntü', masked_frame)

    return center_x, center_y

def main():
    # Kameradan görüntü yakalama
    cap = cv2.VideoCapture(0)

    # İlk konumu alma
    ret, frame = cap.read()
    x, y = get_initial_position(frame)

    # Önceki kareyi kaydetme
    previous_frame = frame

    while True:
        # Kameradan görüntü yakalama
        ret, frame = cap.read()

        # Nesneyi takip etme
        x, y = track_object(frame, x, y, previous_frame)

        # Önceki kareyi güncelleme
        previous_frame = frame

        # 'q' tuşuna basılmasını bekleyin
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
