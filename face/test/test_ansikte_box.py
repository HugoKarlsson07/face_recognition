import threading
import cv2
import os
from deepface import DeepFace

# Ladda Haar Cascade för ansiktsdetektering
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Ladda alla referensbilder från en mapp
reference_folder = r"C:\Users\hugoo\face_reconistion\face\train_faces"
reference_images = []

for filename in os.listdir(reference_folder):
    if filename.endswith((".jpg", ".jpeg", ".png")):
        img_path = os.path.join(reference_folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            reference_images.append(img)

if not reference_images:
    print("Fel: Ingen giltig referensbild hittades!")
    exit()

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Fel: Kan inte öppna kameran!")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

counter = 0
face_match = None  # None = Ingen detektion, False = Ingen match, True = Match

def check_face(frame):
    global face_match
    try:
        for ref_img in reference_images:
            result = DeepFace.verify(frame, ref_img.copy())
            if result["verified"]:
                face_match = True  # Matchad
                return
        face_match = False  # Ingen match
    except Exception as e:
        print("Fel vid ansiktsigenkänning:", e)
        face_match = False

while True:
    ret, frame = cap.read()

    if ret:
        # Konvertera bilden till gråskala för ansiktsdetektering
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

        # Starta ansiktsigenkänning var 60:e frame i en ny tråd
        if counter % 60 == 0 and len(faces) > 0:
            try:
                threading.Thread(target=check_face, args=(frame.copy(),)).start()
            except Exception:
                pass
        counter += 1

        # Rita en ruta runt detekterade ansikten
        for (x, y, w, h) in faces:
            # Bestäm färg baserat på status
            if face_match is None:
                box_color = (0, 255, 255)  # 🔶 Gul: Väntar på igenkänning
            elif face_match:
                box_color = (0, 0, 255)  # 🔴 Röd: Matchad
            else:
                box_color = (0, 255, 0)  # 🟢 Grön: Ingen match

            # Rita rektangeln runt ansiktet
            cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 3)

        # Visa videoströmmen
        cv2.imshow("Video", frame)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
