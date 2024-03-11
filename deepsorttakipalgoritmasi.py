import tensorflow as tf
import numpy as np
import cv2

# Modelin yüklenmesi
model = tf.keras.models.load_model("deep_sort_model.h5")

# Takip fonksiyonu
def track_object(frame, previous_frame):
    # Görüntüyü ön işleme
    # ... (Görüntü ön işleme kodunu buraya ekleyin)

    # Nesne algılama
    # ... (YOLOv5 veya SSDLite gibi bir model kullanarak nesne algılama kodunu buraya ekleyin)

    # Algılanan nesnelerin özelliklerini çıkarma
    features = model.predict(detections)

    # Deep SORT ile takip etme
    tracker = Sort()
    track_ids, track_bboxes = tracker.update(features)

    # Görselleştirme
    for track_id, track_bbox in zip(track_ids, track_bboxes):
        x, y, w, h = track_bbox
        cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
        cv2.putText(frame, str(track_id), (int(x), int(y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Görüntüleri gösterme
    cv2.imshow('Takip Edilen Nesneler', frame)

    return track_ids, track_bboxes

def main():
    # Kameradan görüntü yakalama
    cap = cv2.VideoCapture(0)

    # Önceki kareyi kaydetme
    ret, previous_frame = cap.read()

    while True:
        # Kameradan görüntü yakalama
        ret, frame = cap.read()

        # Nesneyi takip etme
        track_ids, track_bboxes = track_object(frame, previous_frame)

        # Önceki kareyi güncelleme
        previous_frame = frame

        # 'q' tuşuna basılmasını bekleyin
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
