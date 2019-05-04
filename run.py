# -*- coding: UTF-8 -*-
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog
from pose.estimator import TfPoseEstimator
from pose.networks import get_graph_path
from utils.sort import Sort
from utils.actions import actionPredictor
from utils.joint_preprocess import *
import sys
import cv2
import numpy as np
import time
import settings

poseEstimator = None
indicators = {
    'camera': False,
}

def load_model():
    global poseEstimator
    poseEstimator = TfPoseEstimator(
        get_graph_path('mobilenet_thin'), target_size=(432, 368))


class Ui_MainWindow(QtWidgets.QWidget):

    def __init__(self, parent=None):
        super(Ui_MainWindow, self).__init__(parent)
        self.tracker = Sort(settings.sort_max_age, settings.sort_min_hit)
        self.timer_camera = QtCore.QTimer()
        # self.cap = cv2.VideoCapture()
        # self.cap = cv2.VideoCapture()
        self.cap = None
        self.CAM_NUM = 0
        self.set_ui()
        self.slot_init()
        self.__flag_mode = 0
        self.fps = 0.00
        self.data = {}
        self.memory = {}
        self.joints = []
        self.current = []
        self.previous = []
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
        self.out = cv2.VideoWriter('x.mp4', self.fourcc, 1, (settings.winWidth, settings.winHeight))

    def set_ui(self):

        self.__layout_main = QtWidgets.QHBoxLayout()
        self.__layout_fun_button = QtWidgets.QVBoxLayout()
        self.__layout_data_show = QtWidgets.QVBoxLayout()

        self.button_open_camera = QtWidgets.QPushButton(u'Camera OFF')

        self.button_mode_1 = QtWidgets.QPushButton(u'Attitude Estimation OFF')
        self.button_mode_2 = QtWidgets.QPushButton(u'Multiplayer tracking OFF')
        self.button_mode_3 = QtWidgets.QPushButton(u'Behavior recognition OFF')
        self.button_video = QtWidgets.QPushButton("Load Video")

        self.button_start_playing = QtWidgets.QPushButton(u'Start')
        self.button_stop_playing = QtWidgets.QPushButton(u'Stop')

        self.button_close = QtWidgets.QPushButton(u'Exit')

        self.button_open_camera.setMinimumHeight(50)
        self.button_mode_1.setMinimumHeight(50)
        self.button_mode_2.setMinimumHeight(50)
        self.button_mode_3.setMinimumHeight(50)
        self.button_video.setMinimumHeight(50)
        self.button_start_playing.setMinimumHeight(50)
        self.button_stop_playing.setMinimumHeight(50)


        self.button_close.setMinimumHeight(50)

        self.button_close.move(10, 100)

        self.infoBox = QtWidgets.QTextBrowser(self)
        self.infoBox.setGeometry(QtCore.QRect(25, 435, 200, 180))

        # 信息显示
        self.label_show_camera = QtWidgets.QLabel()
        self.label_move = QtWidgets.QLabel()
        self.label_move.setFixedSize(200, 200)

        self.label_show_camera.setFixedSize(settings.winWidth + 1, settings.winHeight + 1)
        self.label_show_camera.setAutoFillBackground(True)

        self.__layout_fun_button.addWidget(self.button_open_camera)
        self.__layout_fun_button.addWidget(self.button_mode_1)
        self.__layout_fun_button.addWidget(self.button_mode_2)
        self.__layout_fun_button.addWidget(self.button_mode_3)
        self.__layout_fun_button.addWidget(self.button_video)
        self.__layout_fun_button.addWidget(self.button_start_playing)
        self.__layout_fun_button.addWidget(self.button_stop_playing)
        self.__layout_fun_button.addWidget(self.button_close)
        self.__layout_fun_button.addWidget(self.label_move)

        self.__layout_main.addLayout(self.__layout_fun_button)
        self.__layout_main.addWidget(self.label_show_camera)

        self.setLayout(self.__layout_main)
        self.label_move.raise_()
        self.setWindowTitle(u'Real-time multi-person attitude estimation and behavior recognition system')

    def slot_init(self):
        self.button_open_camera.clicked.connect(self.switch_to_camera)
        self.timer_camera.timeout.connect(self.show_camera)

        self.button_mode_1.clicked.connect(self.button_event2)
        self.button_mode_2.clicked.connect(self.button_event2)
        self.button_mode_3.clicked.connect(self.button_event2)
        self.button_start_playing.clicked.connect(self.start_playing)
        self.button_stop_playing.clicked.connect(self.stop_playing)
        self.button_video.clicked.connect(self.getfile)
        self.button_close.clicked.connect(self.close)

    
    
    def button_event2(self):
        sender = self.sender()
        if sender == self.button_mode_1 and self.timer_camera.isActive():
            if self.__flag_mode != 1:
                self.__flag_mode = 1
                self.button_mode_1.setText(u'Attitude Estimation ON')
                self.button_mode_2.setText(u'Multiplayer Tracking OFF')
                self.button_mode_3.setText(u'Behavior Recognition OFF')
            else:
                self.__flag_mode = 0
                self.button_mode_1.setText(u'Attitude Estimation OFF')
                self.infoBox.setText(u'Camera is on')
        elif sender == self.button_mode_2 and self.timer_camera.isActive():
            if self.__flag_mode != 2:
                self.__flag_mode = 2
                self.button_mode_1.setText(u'Attitude Estimation OFF')
                self.button_mode_2.setText(u'Multiplayer tracking ON')
                self.button_mode_3.setText(u'Behavior recognition OFF')
            else:
                self.__flag_mode = 0
                self.button_mode_2.setText(u'Multiplayer tracking OFF')
                self.infoBox.setText(u'camera已打开')
        elif sender == self.button_mode_3 and self.timer_camera.isActive():
            if self.__flag_mode != 3:
                self.__flag_mode = 3
                self.button_mode_1.setText(u'Attitude Estimation OFF')
                self.button_mode_2.setText(u'Multiplayer tracking OFF')
                self.button_mode_3.setText(u'Behavior recognition ON')
            else:
                self.__flag_mode = 0
                self.button_mode_3.setText(u'Behavior recognition OFF')
                self.infoBox.setText(u'camera已打开')
        else:
            self.__flag_mode = 0
            self.button_mode_1.setText(u'Attitude Estimation OFF')
            self.button_mode_2.setText(u'Multiplayer tracking OFF')
            self.button_mode_3.setText(u'Behavior recognition OFF')
            

    def show_camera(self):
        start = time.time()
        ret, frame = self.cap.read()
        # if not ret:
        #     print('this should save the video')
        #     self.timer_camera.stop()
        #     self.out.release()
            
        if ret:
            show_s = cv2.resize(frame, (settings.winWidth, settings.winHeight))
            show = cv2.cvtColor(show_s, cv2.COLOR_BGR2RGB)
            if self.__flag_mode == 1:
                self.infoBox.setText(u'当前为人体Attitude Estimation模式')
                humans = poseEstimator.inference(show)
                show = TfPoseEstimator.draw_humans(show, humans, imgcopy=False)

            elif self.__flag_mode == 2:
                self.infoBox.setText(u'当前为Multiplayer tracking模式')
                humans = poseEstimator.inference(show)
                show, joints, bboxes, xcenter, sk = TfPoseEstimator.get_skeleton(show, humans, imgcopy=False)
                show = TfPoseEstimator.draw_humans(show, humans, imgcopy=False)
                height = show.shape[0]
                width = show.shape[1]
                if bboxes:
                    result = np.array(bboxes)
                    det = result[:, 0:5]
                    det[:, 0] = det[:, 0] * width
                    det[:, 1] = det[:, 1] * height
                    det[:, 2] = det[:, 2] * width
                    det[:, 3] = det[:, 3] * height
                    trackers = self.tracker.update(det)

                    for d in trackers:
                        xmin = int(d[0])
                        ymin = int(d[1])
                        xmax = int(d[2])
                        ymax = int(d[3])
                        label = int(d[4])
                        cv2.rectangle(show, (xmin, ymin), (xmax, ymax),
                                      (int(settings.c[label % 32, 0]),
                                       int(settings.c[label % 32, 1]),
                                       int(settings.c[label % 32, 2])), 4)


            # This is the part to get the kt points
            elif self.__flag_mode == 3:
                self.infoBox.setText(u'当前为人体Behavior recognition模式')
                humans = poseEstimator.inference(show)
                ori = np.copy(show)
                show, joints, bboxes, xcenter, sk= TfPoseEstimator.get_skeleton(show, humans, imgcopy=False)
                show = TfPoseEstimator.draw_humans(show, humans, imgcopy=False)
                height = show.shape[0]
                width = show.shape[1]
                if bboxes:
                    result = np.array(bboxes)
                    det = result[:, 0:5]
                    det[:, 0] = det[:, 0] * width
                    det[:, 1] = det[:, 1] * height
                    det[:, 2] = det[:, 2] * width
                    det[:, 3] = det[:, 3] * height
                    trackers = self.tracker.update(det)
                    self.current = [i[-1] for i in trackers]

                    if len(self.previous) > 0:
                        for item in self.previous:
                            if item not in self.current and item in self.data:
                                del self.data[item]
                            if item not in self.current and item in self.memory:
                                del self.memory[item]

                    self.previous = self.current
                    for d in trackers:
                        xmin = int(d[0])
                        ymin = int(d[1])
                        xmax = int(d[2])
                        ymax = int(d[3])
                        label = int(d[4])
                        try:
                            j = np.argmin(np.array([abs(i - (xmax + xmin) / 2.) for i in xcenter]))
                        except:
                            j = 0
                        if joint_filter(joints[j]):
                            joints[j] = joint_completion(joint_completion(joints[j]))
                            if label not in self.data:
                                self.data[label] = [joints[j]]
                                self.memory[label] = 0
                            else:
                                self.data[label].append(joints[j])

                            if len(self.data[label]) == settings.L:
                                pred = actionPredictor().move_status(self.data[label])
                                if pred == 0:
                                    pred = self.memory[label]
                                else:
                                    self.memory[label] = pred
                                self.data[label].pop(0)

                                location = self.data[label][-1][1]
                                if location[0] <= 30:
                                    location = (51, location[1])
                                if location[1] <= 10:
                                    location = (location[0], 31)

                                cv2.putText(show, settings.move_status[pred], (location[0] - 30, location[1] - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                            (0, 255, 0), 2)

                        cv2.rectangle(show, (xmin, ymin), (xmax, ymax),
                                      (int(settings.c[label % 32, 0]),
                                       int(settings.c[label % 32, 1]),
                                       int(settings.c[label % 32, 2])), 4)

            
            self.out.write(show_s) # Write out frame to video

            end = time.time()
            self.fps = 1. / (end - start)
            cv2.putText(show, 'FPS: %.2f' % self.fps, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0], QtGui.QImage.Format_RGB888)
            self.label_show_camera.setPixmap(QtGui.QPixmap.fromImage(showImage))
        else:
            print('this should save the video')
            self.timer_camera.stop()
            self.out.release()


    def closeEvent(self, event):
        # self.out.release()
        ok = QtWidgets.QPushButton()
        cancel = QtWidgets.QPushButton()

        msg = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Warning, u"shut down", u"Are you sure you want to quit?")

        msg.addButton(ok, QtWidgets.QMessageBox.ActionRole)
        msg.addButton(cancel, QtWidgets.QMessageBox.RejectRole)
        ok.setText(u'Yes')
        cancel.setText(u'Cancel')
        if msg.exec_() == QtWidgets.QMessageBox.RejectRole:
            event.ignore()
        else:
            if self.cap.isOpened():
                self.cap.release()
            if self.timer_camera.isActive():
                self.timer_camera.stop()
            event.accept()
            print("System exited.")


    def getfile(self):
      dlg = QFileDialog()
      dlg.setFileMode(QFileDialog.AnyFile)
      fname = dlg.getOpenFileName(self, 'Open file', 
         'c:\\')
      # self.le.setPixmap(QPixmap(fname))
      if fname:
        self.load_video(fname[0])
        
   # def getfiles(self):
   #    dlg = QFileDialog()
   #    dlg.setFileMode(QFileDialog.AnyFile)
   #    dlg.setFilter("Text files (*.txt)")
   #    filenames = QStringList()
        
   #    if dlg.exec_():
   #       filenames = dlg.selectedFiles()
   #       f = open(filenames[0], 'r')
            
   #       with f:
   #          data = f.read()
   #          self.contents.setText(data)

    def load_video(self, name):
        self.cap = cv2.VideoCapture(name)
        # self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.winWidth)
        # self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.winHeight)
        self.timer_camera.stop()
        # self.timer_camera.start(1)

    def switch_to_camera(self):
        indicators['camera'] = not indicators['camera'] 
        [self.stop_camera, self.start_camera][int(indicators['camera'])]()
        self.button_open_camera.setText(
            ['Camera OFF', 'Camera ON'][int(indicators['camera'])]
        )
        

    def stop_camera(self):
        # self.out.release()
        self.cap.release()
        self.timer_camera.stop()

    def start_camera(self):
        self.cap = cv2.VideoCapture(0)
        # self.timer_camera.stop()
        # self.timer_camera.start(1)

    def start_playing(self):
        self.timer_camera.start(1)

    def stop_playing(self):
        self.timer_camera.stop()
        self.out.release()

    def save_video(self):
        name = QtGui.QFileDialog.getSaveFileName(self, 'Save File')
        # file = open(name,'w')
        # text = self.textEdit.toPlainText()
        # file.write(text)
        # file.close()


if __name__ == '__main__':
    load_model()
    print("Load all models done!")
    print("The system starts ro run.")
    app = QtWidgets.QApplication(sys.argv)
    ui = Ui_MainWindow()
    ui.show()
    sys.exit(app.exec_())
