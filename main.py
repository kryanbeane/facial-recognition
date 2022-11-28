import os
import cv2
import numpy as np
import face_recognition as face_rec

# denisImg = face_recognition.load_image_file('images/denis.jpg')
# denisImg = cv2.cvtColor(denisImg, cv2.COLOR_BGR2RGB)
#
# testImg = face_recognition.load_image_file('images/denis.jpg')
# testImg = cv2.cvtColor(denisImg, cv2.COLOR_BGR2RGB)
#
# print(results, faceDistance)
# cv2.putText(testImg, f'{results} {round(faceDistance[0], 2)}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
#
# cv2.imshow('Denis Moskalenko', denisImg)
# cv2.imshow('Denis Test', testImg)
# cv2.waitKey(0)

# 1st Step - Import images and convert to RBG
images_path = 'images'
images = []
image_names = []
image_list = os.listdir(images_path)

# Adding images from directory to the image array and adding the names to the name array
for image in image_list:
    currentImage = cv2.imread(f'{images_path}/{image}')
    images.append(currentImage)
    image_names.append(os.path.splitext(image)[0].upper())


def find_encodings(images_list):
    encoding_list = []

    # Convert each image to RGB, encode the images, and add the encoded images to the list

    for img in images_list:
        print(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encoding = face_rec.face_encodings(img)[0]
        encoding_list.append(encoding)
    return encoding_list


encode_list_known = find_encodings(images)

capture = cv2.VideoCapture(0)

while True:
    success, image = capture.read()

    # Reduce image size to increase speed
    image_small = cv2.resize(image, (0, 0), None, 0.25, 0.25)
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
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0))
            cv2.rectangle(image, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(image, name, (x1 + 6, y2 + 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('Webcam', image)
    cv2.waitKey(1)


# faceLocations = face_recognition.face_locations(denisImg)[0]
# encodeDenis = face_recognition.face_encodings(denisImg)[0]
# cv2.rectangle(denisImg, (faceLocations[3], faceLocations[0]),
#               (faceLocations[1], faceLocations[2]), (255, 0, 255), 2)
#
# faceLocationTest = face_recognition.face_locations(testImg)[0]
# encodeDenisTest = face_recognition.face_encodings(testImg)[0]
# cv2.rectangle(testImg, (faceLocationTest[3], faceLocationTest[0]),
#               (faceLocationTest[1], faceLocationTest[2]), (255, 0, 255), 2)
#
# results = face_recognition.compare_faces([encodeDenis], encodeDenisTest)
# # Find most similar - best match by finding distance. The lower the distance, better the match
# faceDistance = face_recognition.face_distance([encodeDenis], encodeDenisTest)
#
# denisImg = face_recognition.load_image_file('images/denis.jpg')
# denisImg = cv2.cvtColor(denisImg, cv2.COLOR_BGR2RGB)
#
# testImg = face_recognition.load_image_file('images/denis.jpg')
# testImg = cv2.cvtColor(denisImg, cv2.COLOR_BGR2RGB)


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
