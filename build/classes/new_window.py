import sys
from PyQt5 import QtGui
from PyQt5.QtGui import QIcon,QPixmap
from PyQt5.QtWidgets import QMainWindow, \
    QApplication, QWidget,\
    QMessageBox,QAction,QPushButton,\
    QLineEdit,QLabel
import datetime
from NewPro import new_graph,predict_stock
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
now = datetime.datetime.now()

class Window(QMainWindow):
    def __init__(self):
        super().__init__()

        self.title = "Stock Information Entering :"
        self.top = 200
        self.left = 200
        self.width = 600
        self.height = 500

        self.setWindowIcon(QtGui.QIcon("icon.png"))

        self.InitWindow()

    def InitWindow(self):
        # Image Addition
        self.imglabel = QLabel(self)
        self.imglabel.setPixmap(QPixmap("main3.jpg"))
        self.imglabel.setGeometry(-148, 0, 1500, 700)
        #setGeometry(,top,right,bottom)
        #Labels
        self.labelLogin = QLabel("<h2>Enter info :</h2>", self)
        self.labelLogin.move(40, 63)

        self.labelCompanyNames = QLabel("<h3>SYMBOL-----COMPANY</h3>",self)
        self.labelCompanyNames.move(1000, 63)
        self.labelCompanyNames.resize(250,30)

        self.labelComp = QLabel("<h3>Enter the company (IBM ,GOOG,AAPL):</h3>",self)
        self.labelComp.move(40,93)
        self.labelComp.resize(250,30)

        self.labelYear = QLabel("<h3>Enter the year (between 2016-2019):</h3>", self)
        self.labelYear.move(40,123)
        self.labelYear.resize(250,30)

        self.labelMonth = QLabel("<h3>Enter the month (between 1-12):</h3>", self)
        self.labelMonth.move(40, 153)
        self.labelMonth.resize(250, 30)

        self.labelDate = QLabel("<h3>Enter the date (between 1-30):</h3>", self)
        self.labelDate.move(40, 183)
        self.labelDate.resize(250, 30)

        self.labelGraph = QLabel("<h3>Enter the graph type (From Area graph-1/Candle stick-2):</h3>", self)
        self.labelGraph.move(40, 213)
        self.labelGraph.resize(380, 30)

        #Company
        self.companybox = QLineEdit(self)
        self.companybox.move(450,100)
        self.companybox.resize(300,20)

        #Year
        self.yearbox = QLineEdit(self)
        self.yearbox.move(450,130)
        self.yearbox.resize(300,20)

        #Month
        self.monthbox = QLineEdit(self)
        self.monthbox.move(450,160)
        self.monthbox.resize(300,20)

        #Date
        self.datebox = QLineEdit(self)
        self.datebox.move(450,190)
        self.datebox.resize(300,20)

        #Graph
        self.graphbox = QLineEdit(self)
        self.graphbox.move(450, 220)
        self.graphbox.resize(300, 20)

        #Enter button1 Download
        self.start = QPushButton("Download",self)
        self.start.move(600,500)
        self.start.resize(200,50)
        self.start.clicked.connect(self.onClick2)

        #Enter button2 Predict
        self.predict = QPushButton("Predict",self)
        self.predict.move(600,550)
        self.predict.resize(200,50)
        self.predict.clicked.connect(self.onClickPre)

        #Predbox
        self.predbox = QLineEdit(self)
        self.predbox.move(600,615)
        self.predbox.resize(200,20)

        #Status Bar
        message = now.strftime("%Y-%m-%d %H:%M")
        self.statusBar().showMessage(message)

        #Window
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.showMaximized()
        self.show()

        mainMenu = self.menuBar()
        fileMenu = mainMenu.addMenu("File")
        viewMenu = mainMenu.addMenu("View")
        editMenu = mainMenu.addMenu("Edit")
        searchMenu = mainMenu.addMenu("Search")
        toolMenu = mainMenu.addMenu("Tool")
        helpMenu = mainMenu.addMenu("Help")

        exitButton = QAction(QIcon("icon2.ico"),"Exit",self) #QAction is an abstraction for actions performed on menubar, tool bars,etc
        exitButton.setShortcut("Ctrl+E")
        exitButton.setStatusTip("Exit Application")
        exitButton.triggered.connect(self.close)
        fileMenu.addAction(exitButton)

        #Toolbar
        exitAct = QAction(QIcon('icon2.ico'),"Exit",self)
        exitAct.setShortcut("Ctrl+Q")
        exitAct.triggered.connect(self.CloseApp)

        copyAct = QAction(QIcon("icon3.ico"),"Copy",self)

        copyAct.setShortcut("Ctrl+C")

        pasteAct = QAction(QIcon("icon.png"),"Paste",self)
        pasteAct.setShortcut("Ctrl+V")

        deleteAct = QAction(QIcon("icon5.ico"), "Delete", self)
        deleteAct.setShortcut("Ctrl+D")

        saveAct = QAction(QIcon("icon6.ico"), "Save", self)
        saveAct.setShortcut("Ctrl+S")

        self.toolbar = self.addToolBar("Toolbar")
        self.toolbar.addAction(exitAct)
        self.toolbar.addAction(copyAct)
        self.toolbar.addAction(pasteAct)
        self.toolbar.addAction(deleteAct)
        self.toolbar.addAction(saveAct)


    def CloseApp(self):
        self.close()

    def onClick2(self):
        new_graph.onClick3(self.companybox.text(),self.yearbox.text(),self.monthbox.text(),self.datebox.text(),self.graphbox.text())
    def onClickPre(self):
        predict_stock.pred_value(self.predbox.text(),self.companybox.text())


if __name__=="__main__":
    App = QApplication(sys.argv)
    window = Window()
    sys.exit(App.exec())
