import cv2

img = cv2.imread(r"C:\Users\hugoo\face_reconistion\face\jag.jpg")
if img is None:
    print("Fel: Kan inte läsa in bilden!")
else:
    print("Bilden laddades in korrekt.")
