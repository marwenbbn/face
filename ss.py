import sys
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5 import QtCore,QtGui
import cv2
import face_recognition
from sklearn import svm
import os

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


class MainWindow(QWidget):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.VBL = QVBoxLayout()

        self.FeedLabel = QLabel()
        self.FeedLabel.setGeometry(QtCore.QRect(10, 10, 491, 351))
        self.FeedLabel.setText("hi")
        self.FeedLabel.setObjectName("label")
        self.VBL.addWidget(self.FeedLabel)

        self.vbox = QVBoxLayout()

        self.groupBox = QGroupBox("dd")
        self.groupBox.setEnabled(False)
        self.groupBox.setGeometry(QtCore.QRect(520, 30, 231, 221))
        self.groupBox.setTitle("")
        self.groupBox.setObjectName("groupBox")
        self.groupBox.setLayout(self.vbox)

        self.label_2 = QLabel()
        self.label_2.setGeometry(QtCore.QRect(10, 30, 101, 16))
        self.label_2.setObjectName("label_2")
        self.vbox.addWidget(self.label_2)
        self.horizontalLayoutWidget = QWidget(self.groupBox)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(10, 60, 221, 31))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")
        self.horizontalLayout = QHBoxLayout(self.horizontalLayoutWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label_3 = QLabel(self.horizontalLayoutWidget)
        self.label_3.setObjectName("label_3")
        self.horizontalLayout.addWidget(self.label_3)
        self.textEdit = QTextEdit(self.horizontalLayoutWidget)
        self.textEdit.setObjectName("textEdit")
        self.horizontalLayout.addWidget(self.textEdit)
        self.buttonBox = QDialogButtonBox(self.groupBox)
        self.buttonBox.setGeometry(QtCore.QRect(40, 190, 166, 25))
        self.buttonBox.setStandardButtons(QDialogButtonBox.Cancel|QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.pushButton = QPushButton(self.groupBox)
        self.pushButton.setGeometry(QtCore.QRect(10, 100, 89, 25))
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QPushButton()
        self.VBL.addWidget(self.pushButton_2)
        self.pushButton_2.setGeometry(QtCore.QRect(660, 20, 89, 25))
        self.pushButton_2.setObjectName("pushButton_2")
        self.VBL.addWidget(self.groupBox)
        
        


        self.Worker1 = Worker1()

        self.Worker1.start()
        self.Worker1.ImageUpdate.connect(self.ImageUpdateSlot)
        self.setLayout(self.VBL)



    def ImageUpdateSlot(self, Image):
        self.FeedLabel.setPixmap(QPixmap.fromImage(Image))

    def CancelFeed(self):
        self.Worker1.stop()
    def p(self):
        print("hello")

class Worker1(QThread):
    ImageUpdate = pyqtSignal(QImage)
    def run(self):
        self.ThreadActive = True
        Capture = cv2.VideoCapture(0)
        while self.ThreadActive:
            ret, frame = Capture.read()
            if ret:
                
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
                Image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                #FlippedImage = cv2.flip(Image, 2)
                ConvertToQtFormat = QImage(Image.data, Image.shape[1], Image.shape[0], QImage.Format_RGB888)
                Pic = ConvertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
                self.ImageUpdate.emit(Pic)
    def stop(self):
        self.ThreadActive = False
        self.quit()

if __name__ == "__main__":
    App = QApplication(sys.argv)
    Root = MainWindow()
    Root.show()
    sys.exit(App.exec())