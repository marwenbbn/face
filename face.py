from unicodedata import name
import face_recognition
import cv2
import os
import numpy as np
video_capture = cv2.VideoCapture(0)
#video_capture.open("http://192.168.1.51:8080/video")

known_face_encodings = []
known_face_names = []
detect= True
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

def init ():
    dir_list = os.listdir("./faces")
    for img in dir_list:
        image = face_recognition.load_image_file("./faces/"+img)
        face_encoding = face_recognition.face_encodings(image)[0]   
        name = img.replace(".jpg","")
        known_face_encodings.append(face_encoding)
        known_face_names.append(name)
init()

def detection(): 
    global process_this_frame,face_locations,face_encodiace_locations,face_encodings,face_names,detectngs,face_names,detect
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame


    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    k = cv2.waitKey(33)
    if k==27:    # Esc key to stop
        return True
    elif k==97:  # normally -1 returned,so don't print it
        detect = False
    else:
        print (k) # else print its val
        return False 
saved_img  = ''
new_name =""
taping = False
def save():
    global saved_img,new_name,taping,known_face_encodings,known_face_names,detect
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    k = cv2.waitKey(33)
    if k==27:    # Esc key to stop
        return True
    elif k==97 and not taping:  # normally -1 returned,so don't print it
        detect = False
    elif k==115 and not taping:  # normally -1 returned,so don't print it
        saved_img = frame.copy()
        cv2.imshow('saved', saved_img)
    elif k==121 and not taping:  # normally -1 returned,so don't print it
        taping = True
    elif k==13 and taping: 
        cv2.imwrite("./faces/"+new_name+".jpg", saved_img)
        image = face_recognition.load_image_file("./faces/"+new_name+".jpg")
        face_encoding = face_recognition.face_encodings(image)[0]   
        known_face_encodings.append(face_encoding)
        known_face_names.append(new_name)
        cv2.destroyWindow('saved') 
        taping = False
        new_name= ""
        detect = True
    elif taping:
        if k >0:
            new_name += chr(k)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(saved_img, new_name, (30,30), font, 1.0, (255, 0, 0), 1)
            cv2.imshow('saved', saved_img)

        print (k) # else print its val
        return False
while True :
    ret, frame = video_capture.read()

    small_frame = cv2.resize(frame, (0, 0), fx=0.25 ,fy=0.25)

    rgb_small_frame = small_frame[:, :, ::-1]
    if detect :
        if detection(): break
    else :
        save()
 

