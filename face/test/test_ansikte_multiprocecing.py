import cv2
import os
import multiprocessing
import threading
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

# Hämta antal CPU-kärnor och använd max-1
num_processes = multiprocessing.cpu_count() - 1

# Multiprocessing pool för snabbare ansiktsigenkänning
pool = multiprocessing.Pool(processes=num_processes)

# Funktion för parallell ansiktsigenkänning
def compare_face(ref_img, frame):
    try:
        result = DeepFace.verify(frame, ref_img.copy(), model_name="SFace")  # Snabbare modell
        return result["verified"]
    except Exception as e:
        print("Fel vid ansiktsigenkänning:", e)
        return False

# Funktion som startas i en egen tråd för att checka ansikten
def check_face(frame):
    global face_match
    results = pool.starmap(compare_face, [(img, frame) for img in reference_images])
    face_match = any(results)  # Om någon bild matchar → True, annars False

while True:
    ret, frame = cap.read()

    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

        # Starta ansiktsigenkänning i en separat tråd var 60:e frame
        if counter % 60 == 0 and len(faces) > 0:
            threading.Thread(target=check_face, args=(frame.copy(),), daemon=True).start()
        counter += 1

        # Rita en ruta runt ansiktet baserat på status
        for (x, y, w, h) in faces:
            if face_match is None:
                box_color = (0, 255, 255)  # 🔶 Gul: Väntar på igenkänning
            elif face_match:
                box_color = (0, 0, 255)  # 🔴 Röd: Matchad
            else:
                box_color = (0, 255, 0)  # 🟢 Grön: Ingen match

            cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 3)

        cv2.imshow("Video", frame)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
pool.close()
pool.join()
