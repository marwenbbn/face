# Train multiple images per person
# Find and recognize faces in an image using a SVC with scikit-learn

"""
Structure:
        <test_image>.jpg
        <train_dir>/
            <person_1>/
                <person_1_face-1>.jpg
                <person_1_face-2>.jpg
                .
                .
                <person_1_face-n>.jpg
           <person_2>/
                <person_2_face-1>.jpg
                <person_2_face-2>.jpg
                .
                .
                <person_2_face-n>.jpg
            .
            .
            <person_n>/
                <person_n_face-1>.jpg
                <person_n_face-2>.jpg
                .
                .
                <person_n_face-n>.jpg
"""

import face_recognition
from sklearn import svm
import os
import cv2
# Training the SVC classifier

# The training data would be all the face encodings from all the known images and the labels are their names
encodings = []
names = []

# Training directory
train_dir = os.listdir('./faces-svm/')

# Loop through each person in the training directory
for person in train_dir:
    pix = os.listdir("./faces-svm/" + person)

    # Loop through each training image for the current person
    for person_img in pix:
        # Get the face encodings for the face in each image file
        face = face_recognition.load_image_file("./faces-svm/" + person + "/" + person_img)
        face_bounding_boxes = face_recognition.face_locations(face)

        #If training image contains exactly one face
        if len(face_bounding_boxes) == 1:
            face_enc = face_recognition.face_encodings(face)[0]
            # Add face encoding for current image with corresponding label (name) to the training data
            encodings.append(face_enc)
            names.append(person)
        else:
            print(person + "/" + person_img + " was skipped and can't be used for training")

# Create and train the SVC classifier
clf = svm.SVC(gamma='scale')
clf.fit(encodings,names)



video_capture = cv2.VideoCapture(0)

while True :
    ret, frame = video_capture.read()

    small_frame = cv2.resize(frame, (0, 0), fx=0.25 ,fy=0.25)

    rgb_small_frame = small_frame[:, :, ::-1]




    # Load the test image with unknown faces into a numpy array
    #test_image = face_recognition.load_image_file('./salazar.jpg')

    # Find all the faces in the test image using the default HOG-based model
    face_locations = face_recognition.face_locations(rgb_small_frame)
    no = len(face_locations)

    for i in range(no):
        test_image_enc = face_recognition.face_encodings(rgb_small_frame)[i]
        print(clf.predict([test_image_enc]))
        name = clf.predict([test_image_enc])
        top, right, bottom, left = face_locations[i]
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, *name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)


    cv2.imshow('saved', frame)

    k = cv2.waitKey(1)
    if k==27:    # Esc key to stop
        break