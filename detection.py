import cv2
import random
from ultralytics import YOLO
import winsound

def colours(cls_num):
    random.seed(cls_num)
    return tuple(random.randint(0,255) for _ in range(3))

# Load pre-trained model
yolo = YOLO('yolov8s.pt')

cap = cv2.VideoCapture(0)
alarm = False

while True:
    ret, frm = cap.read()
    if not ret:
        break
    phone_det = False
    
    # Applied our tuned threshold from the sweep study
    results = yolo.track(frm, 
                         stream=True, 
                         classes=[67], 
                         conf=0.70)

    for result in results:
        class_names = result.names
        for box in result.boxes:
            phone_det = True
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cls = int(box.cls[0])
            class_name = class_names[cls]
            confidence = float(box.conf[0])
            colour = colours(cls)

            cv2.rectangle(frm, (x1, y1), (x2, y2), colour, 2)
            cv2.putText(frm, f"{class_name} {confidence:.2f}",
                        (x1, max(y1 - 10, 20)), 
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, colour, 2)

    if phone_det and not alarm:
        winsound.PlaySound(
            "alarm.wav",
            winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_LOOP
        )  
        alarm = True
    elif not phone_det and alarm:
        winsound.PlaySound(None, 0)      
        alarm = False

    cv2.imshow("Detection", frm)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

winsound.PlaySound(None, 0)
cap.release()
cv2.destroyAllWindows()