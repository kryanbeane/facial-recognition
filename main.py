import os
import cv2
import numpy as np
import face_recognition as face_rec
from flask import Flask, render_template, Response
import socket
from flask_socketio import SocketIO

hostname = socket.gethostname()
myIpAddress = socket.gethostbyname(hostname)
app = Flask(__name__)
socketioApp = SocketIO(app)

# 1st Step - Import images and convert to RBG
images_path = 'images'
images = []
image_names = []
image_list = os.listdir(images_path)

print("http://" + myIpAddress + ":8080")

# Adding images from directory to the image array and adding the names to the name array
for image in image_list:
    currentImage = cv2.imread(f'{images_path}/{image}')
    images.append(currentImage)
    image_names.append(os.path.splitext(image)[0].upper())


def find_encodings(images_list):
    encoding_list = []

    # Convert each image to RGB, encode the images, and add the encoded images to the list
    for img in images_list:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encoding = face_rec.face_encodings(img)[0]
        encoding_list.append(encoding)
    return encoding_list


def recognise_faces():
    encode_list_known = find_encodings(images)
    capture = cv2.VideoCapture(0)

    while True:
        success, img = capture.read()

        # Reduce image size to increase speed
        image_small = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        image_small = cv2.cvtColor(image_small, cv2.COLOR_BGR2RGB)

        current_frame_faces = face_rec.face_locations(image_small)
        current_frame_encoding = face_rec.face_encodings(image_small, current_frame_faces)

        for encoded_face, face_location in zip(current_frame_encoding, current_frame_faces):
            matches = face_rec.compare_faces(encode_list_known, encoded_face)
            face_distance = face_rec.face_distance(encode_list_known, encoded_face)
            print(face_distance)

            match_index = np.argmin(face_distance)

            if matches[match_index]:
                name = image_names[match_index].upper()
                print(name)

                y1, x2, y2, x1 = face_location
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0))
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, name, (x1 + 6, y2 + 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

        # cv2.imshow('Webcam', img)
        # cv2.waitKey(1)
        ret, buffer = cv2.imencode('.jpg', img)  # convert capture to jpg for browser
        img = buffer.tobytes()
        # concatenate frames and output
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n')

        if cv2.waitKey(1) == ord('q'):  # break loop if q pressed
            break

    capture.release()  # turn off cam
    cv2.destroyAllWindows()  # close all windows


@app.route('/video_feed')
def video_feed():
    return Response(recognise_faces(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    # Video streaming Home Page
    return render_template('index.html')


def run():
    socketioApp.run(app)


if __name__ == '__main__':
    socketioApp.run(app)
