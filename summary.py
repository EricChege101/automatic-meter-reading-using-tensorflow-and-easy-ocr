from io import open_code
import os
from PyQt5 import QtCore, QtGui, QtWidgets
import sys
from PyQt5 import QtWidgets
from PyQt5.uic import loadUi
from PyQt5.QtWidgets import QDialog, QApplication, QFileDialog, QMainWindow
from PyQt5.uic import loadUi




class MainWindow(QMainWindow):
     def __init__(self):
        super(MainWindow,self).__init__()
        loadUi("summary.ui",self)
        self.detectionbutton.clicked.connect(self.detection)
        self.csvbutton.clicked.connect(self.csvfile)
        self.ephoto.setPixmap(QtGui.QPixmap("Tensorflow\workspace\images\Working folder\10.jpg"))
        with open('Tensorflow\workspace\images\Working folder\myfile.txt','r+') as myfile:
            data = myfile.read()
        self.reading.setText(data)
        

        
        

     def detection(self):
        os.startfile("Detected_Images")
        #dialog = QFileDialog()
        #folder_path = dialog.getExistingDirectory(self, "open", "Detected_Images")
        #return folder_path
    
    
     def csvfile(self):
        os.startfile('detection_results.csv')
        #cssv=open("detection_results.csv")
        #dialog = QFileDialog()
        #file = dialog.getOpenFileName(self,"open","detection_results.csv") 
        #return file
        #return cssv

        
        

     

app=QApplication(sys.argv)
mainwindow=MainWindow()
Widget=QtWidgets.QStackedWidget()
Widget.addWidget(mainwindow)
Widget.setFixedWidth(951)
Widget.setFixedHeight(781)
Widget.show()
sys.exit(app.exec_())
