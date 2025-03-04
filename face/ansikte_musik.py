import cv2
import os
import threading
from deepface import DeepFace
import pygame

# Initiera pygame.mixer
pygame.mixer.init()

# Ladda musikfilen
music_path = r"C:\Users\hugoo\face_reconistion\face\music\song.mp3"
if os.path.exists(music_path):
    pygame.mixer.music.load(music_path)
else:
    print("Fel: Musikfilen hittades inte!")
    exit()

# Ladda Haar Cascade för ansiktsdetektering
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Ladda referensbilder
reference_folder = r"C:\Users\hugoo\face_reconistion\face\train_faces"
reference_images = []

for filename in os.listdir(reference_folder):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        img_path = os.path.join(reference_folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            reference_images.append(img)

if not reference_images:
    print("Fel: Ingen giltig referensbild hittades!")
    exit()

# Starta kameran
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("Fel: Kan inte öppna kameran!")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

counter = 0
face_match = None  # None = Ingen detektion, False = Ingen match, True = Match
lock = threading.Lock()
music_playing = False  # Flagga för att undvika upprepade spelningar

def check_face(frame):
    global face_match, music_playing
    try:
        for ref_img in reference_images:
            result = DeepFace.verify(frame, ref_img, model_name="SFace", enforce_detection=False)
            if result.get("verified", False):
                with lock:
                    face_match = True
                    if not music_playing:  # Spela bara musik om den inte redan spelas
                        pygame.mixer.music.play()
                        music_playing = True
                return
        
        # Om ingen match hittas, stoppa musiken
        with lock:
            face_match = False
            if music_playing:
                pygame.mixer.music.stop()
                music_playing = False
                
    except Exception as e:
        print("Fel vid ansiktsigenkänning:", e)
        with lock:
            face_match = False
            if music_playing:
                pygame.mixer.music.stop()
                music_playing = False

while True:
    ret, frame = cap.read()
    if not ret:
        print("Fel: Misslyckades att läsa från kameran!")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

    # Starta ansiktsigenkänning var 60:e frame om ett ansikte hittas
    if counter % 60 == 0 and len(faces) > 0:
        threading.Thread(target=check_face, args=(frame.copy(),), daemon=True).start()
    counter += 1

    # Om inga ansikten detekteras, stoppa musiken
    if len(faces) == 0 and music_playing:
        with lock:
            pygame.mixer.music.stop()
            music_playing = False

    # Rita en ruta runt ansiktet
    with lock:
        current_face_match = face_match

    for (x, y, w, h) in faces:
        if current_face_match is None:
            box_color = (0, 255, 255)  # 🔶 Väntar på igenkänning
        elif current_face_match:
            box_color = (0, 0, 255)  # 🔴 Matchad
        else:
            box_color = (0, 255, 0)  # 🟢 Ingen match

        cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 3)

    cv2.imshow("Video", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
