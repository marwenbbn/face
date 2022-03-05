
from PyQt5 import QtWidgets, uic
import sys
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import cv2
import face_recognition
from sklearn import svm
import os
import numpy as np
import uuid
# The training data would be all the face encodings from all the known images and the labels are their names




class Ui(QtWidgets.QMainWindow):
    def __init__(self):
        super(Ui, self).__init__()
        uic.loadUi('untitled.ui', self)

        self.pushButton_2.clicked.connect(self.add)
        self.pushButton.clicked.connect(self.shoot)

        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

        self.addthis.clicked.connect(self.addthism)

        self.Worker1 = Worker1()
        self.Worker1.start()
        self.Worker1.ImageUpdate.connect(self.ImageUpdateSlot)

        self.show()
    def addthism(self):
        self.newImg = cv2.cvtColor(self.newImg, cv2.COLOR_BGR2RGB)
        cv2.imshow("this",self.newImg)
        if(self.textEdit.toPlainText() !=""):
            if not os.path.isdir("./faces-svm/"+self.textEdit.toPlainText()):
                os.mkdir("./faces-svm/"+self.textEdit.toPlainText())
            filename = str(uuid.uuid4())
            cv2.imwrite("./faces-svm/"+self.textEdit.toPlainText()+"/"+filename+".jpg", self.newImg)
    def accept(self):
        self.Worker1.train()
        self.reject()
        print("save")
    def reject(self):
        self.groupBox.setEnabled(False)
        self.Worker1.detection = True
        self.label_4.clear()

    def shoot(self):
        incomingImage = self.frame.convertToFormat(4)
        width = incomingImage.width()
        height = incomingImage.height()
        ptr = incomingImage.bits()
        ptr.setsize(incomingImage.byteCount())
        arr = np.array(ptr).reshape(height, width, 4)
        
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        s =80
        top, right, bottom, left  = face_recognition.face_locations(arr)[0]
        face = arr[ top-50:bottom+50,left-50:right+50]
        cv2.imshow("ff",face)
        self.newImg= face
        a = QRect (left-s, top-s,(right-left+s*2), (bottom-top+s*2))
        cropped = self.frame.copy(a)
        Pic = cropped.scaled(224, 224)
        
        self.label_4.setPixmap(QPixmap.fromImage(Pic))

        


    def ImageUpdateSlot(self, Image):
        if not self.Worker1.detection:
            self.frame =Image
        self.label.setPixmap(QPixmap.fromImage(Image))

    def CancelFeed(self):
        self.Worker1.stop()
    def add(self):
        self.groupBox.setEnabled(True)
        self.Worker1.detection = False



class Worker1(QThread):
    ImageUpdate = pyqtSignal(QImage)
    def run(self):
        self.train()
        self.ThreadActive = True
        self.detection = True
        Capture = cv2.VideoCapture(0)
        while self.ThreadActive:
            ret, frame = Capture.read()
            if ret:
                
                small_frame = cv2.resize(frame, (0, 0), fx=0.25 ,fy=0.25)

                rgb_small_frame = small_frame[:, :, ::-1]

                if self.detection:

                    face_locations = face_recognition.face_locations(rgb_small_frame)
                    no = len(face_locations)

                    for i in range(no):
                        test_image_enc = face_recognition.face_encodings(rgb_small_frame)[i]
                        name = self.clf.predict([test_image_enc])
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
                Image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                #FlippedImage = cv2.flip(Image, 2)
                ConvertToQtFormat = QImage(Image.data, Image.shape[1], Image.shape[0], QImage.Format_RGB888)
                Pic = ConvertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
                self.ImageUpdate.emit(Pic)
    def train(self):
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
        self.clf = clf

    def stop(self):
        self.ThreadActive = False
        self.quit()
app = QtWidgets.QApplication(sys.argv)
window = Ui()
app.exec_()
