Face Recognition Music Player
This Python program recognizes faces by comparing them to reference images and plays music upon finding a match.​

Features
Face Recognition: Compares detected faces to a set of reference images.​

Music Playback: Plays specific music tracks when a face match is found.​

Installation
Clone the Repository:

bash
Kopiera
Redigera
git clone https://github.com/HugoKarlsson07/face_recognition.git
Navigate to the Project Directory:

bash
Kopiera
Redigera
cd face_recognition
Install Required Dependencies:

bash
Kopiera
Redigera
pip install -r requirements.txt
Usage
Prepare Reference Images:

Store the reference images in the reference_images directory.​

Ensure each image file is named appropriately to identify the person.​

Run the Program:

bash
Kopiera
Redigera
python main.py
Operation:

The program will access your webcam to detect faces.​

Upon detecting a face, it compares it to the reference images.​

If a match is found, the corresponding music track will play.​

Dependencies
Python 3.x

OpenCV​

face_recognition​

pygame​

Ensure all dependencies are installed as specified in the requirements.txt file.​

Contributing
Contributions are welcome! Please fork the repository and submit a pull request with your enhancements.​

License
This project is licensed under the MIT License.​

​This README was generated based on the project description and typical usage patterns.
