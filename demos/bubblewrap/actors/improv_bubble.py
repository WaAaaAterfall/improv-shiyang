# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'improv_bubble.ui'
#
# Created by: PyQt5 UI code generator 5.15.7
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Demo(object):
    def setupUi(self, Demo):
        Demo.setObjectName("Demo")
        Demo.resize(600, 485)
        self.groupBox = QtWidgets.QGroupBox(Demo)
        self.groupBox.setEnabled(True)
        self.groupBox.setGeometry(QtCore.QRect(30, 22, 121, 111))
        self.groupBox.setObjectName("groupBox")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.groupBox)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.pushButton_4 = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_4.setObjectName("pushButton_4")
        self.verticalLayout_2.addWidget(self.pushButton_4)
        self.pushButton_3 = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_3.setObjectName("pushButton_3")
        self.verticalLayout_2.addWidget(self.pushButton_3)
        self.label = QtWidgets.QLabel(Demo)
        self.label.setGeometry(QtCore.QRect(180, 20, 111, 20))
        self.label.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.label.setObjectName("label")
        self.groupBox_3 = QtWidgets.QGroupBox(Demo)
        self.groupBox_3.setEnabled(True)
        self.groupBox_3.setGeometry(QtCore.QRect(30, 150, 121, 101))
        self.groupBox_3.setObjectName("groupBox_3")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.groupBox_3)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.radioButton = QtWidgets.QRadioButton(self.groupBox_3)
        self.radioButton.setObjectName("radioButton")
        self.verticalLayout_3.addWidget(self.radioButton)
        self.radioButton_2 = QtWidgets.QRadioButton(self.groupBox_3)
        self.radioButton_2.setObjectName("radioButton_2")
        self.verticalLayout_3.addWidget(self.radioButton_2)
        self.frame = QtWidgets.QFrame(Demo)
        self.frame.setGeometry(QtCore.QRect(180, 40, 381, 221))
        self.frame.setStyleSheet("")
        self.frame.setObjectName("frame")

        self.retranslateUi(Demo)
        QtCore.QMetaObject.connectSlotsByName(Demo)

    def retranslateUi(self, Demo):
        _translate = QtCore.QCoreApplication.translate
        Demo.setWindowTitle(_translate("Demo", "Demo"))
        self.groupBox.setTitle(_translate("Demo", "Operations"))
        self.pushButton_4.setText(_translate("Demo", "Setup"))
        self.pushButton_3.setText(_translate("Demo", "Run"))
        self.label.setText(_translate("Demo", "Bubblewrap"))
        self.groupBox_3.setTitle(_translate("Demo", "Plot style"))
        self.radioButton.setText(_translate("Demo", "Lineplot"))
        self.radioButton_2.setText(_translate("Demo", "Scatterplot"))