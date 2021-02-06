
from PyQt5.QtWidgets import QApplication, QFileDialog
from PyQt5.QtWidgets import *
from PyQt5.uic import *
from PyQt5.Qt import QApplication, QUrl, QDesktopServices


from PyQt5.QtCore import pyqtSlot
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from PyQt5 import uic
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtWidgets import QMainWindow, QLabel, QGridLayout,QDesktopWidget, QWidget,QTableWidget,QTableView,QTableWidgetItem,QHeaderView,QGraphicsScene,QGraphicsPixmapItem,QFileDialog
import cv2
from PyQt5 import QtCore, QtGui, QtWidgets
import xlwt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import sys

from sklearn.metrics import accuracy_score
# from arayuz import Ui_Dialog
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import sys 
from PyQt5.QtWidgets import QDialog, QApplication, QPushButton, QVBoxLayout 
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas 
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar 
import matplotlib.pyplot as plt 
import random

class Windoww(QDialog): 
       
    # constructor 
    def __init__(self, parent=None): 
        super(Windoww, self).__init__(parent) 
   
        self.figure = plt.figure() 
        self.canvas = FigureCanvas(self.figure)  
        self.toolbar = NavigationToolbar(self.canvas, self)   
        layout = QVBoxLayout()            
        layout.addWidget(self.toolbar) 
        layout.addWidget(self.canvas)          
        self.setLayout(layout) 
   
    # action called by thte push button 
    def plot(self): 
              
        self.figure.clear()    
        self.canvas.draw() 
class Windoww2(QDialog): 
       
    # constructor 
     def __init__(self, parent=None): 
        super(Windoww2, self).__init__(parent) 
        loadUi("arayuz2.ui", self)
 
     def csvYukle(self,x_train,y_train,x_test,y_test):
            from pandas import DataFrame
            x_train = DataFrame(x_train)
            y_train = DataFrame(y_train)
            y_test = DataFrame(y_test)
            x_test = DataFrame(x_test)
            
            c=len(x_train.columns)
            r=len(x_train.values)
            self.tableWidget.setColumnCount(c)
            self.tableWidget.setRowCount(r)
            colmnames=["Cinsiyet","Yaşlı","Evli","Ekonomik Bagımlı","Abonelik Süresi","Telefon Hizmeti","Birden Fazla Hat",
                        "İnternet Servisi","Çevrimiçi Güvenlik","Çevrimiçi Yedekleme","Cihaz Koruma","Teknik Destek","Televizyon",
                        "Film","Sözleşme Süresi","Çevrimiçi Fatura","Ödeme Yöntemi","Aylık Ödeme","Toplam Ödeme"]       
            self.tableWidget.setHorizontalHeaderLabels(colmnames)
            
            for i,row in enumerate(x_train):            
                 for j,cell in enumerate(x_train.values):         
                      self.tableWidget.setItem(j,i, QtWidgets.QTableWidgetItem(str(cell[i])))  
        
    
            c=len(y_train.columns)
            r=len(y_train.values)
            self.tableWidget_2.setColumnCount(c)
            self.tableWidget_2.setRowCount(r)
            colmnames=["Müşteri Kaybı"]       
            self.tableWidget_2.setHorizontalHeaderLabels(colmnames)
            
            for i,row in enumerate(y_train):            
                 for j,cell in enumerate(y_train.values):         
                      self.tableWidget_2.setItem(j,i, QtWidgets.QTableWidgetItem(str(cell[i])))  


            c=len(y_test.columns)
            r=len(y_test.values)
            self.tableWidget_4.setColumnCount(c)
            self.tableWidget_4.setRowCount(r)
            colmnames=["Müşteri Kaybı"]       
            self.tableWidget_4.setHorizontalHeaderLabels(colmnames)
            
            for i,row in enumerate(y_test):            
                 for j,cell in enumerate(y_test.values):         
                      self.tableWidget_4.setItem(j,i, QtWidgets.QTableWidgetItem(str(cell[i])))  
                      
            c=len(x_test.columns)
            r=len(x_test.values)
            self.tableWidget_3.setColumnCount(c)
            self.tableWidget_3.setRowCount(r)
            colmnames=["Cinsiyet","Yaşlı","Evli","Ekonomik Bagımlı","Abonelik Süresi","Telefon Hizmeti","Birden Fazla Hat",
                        "İnternet Servisi","Çevrimiçi Güvenlik","Çevrimiçi Yedekleme","Cihaz Koruma","Teknik Destek","Televizyon",
                        "Film","Sözleşme Süresi","Çevrimiçi Fatura","Ödeme Yöntemi","Aylık Ödeme","Toplam Ödeme","Müşteri Kaybı"]       
            self.tableWidget_3.setHorizontalHeaderLabels(colmnames)
            
            for i,row in enumerate(x_test):            
                 for j,cell in enumerate(x_test.values):         
                      self.tableWidget_3.setItem(j,i, QtWidgets.QTableWidgetItem(str(cell[i])))  


class window(QDialog):

    def __init__(self):
        super(window, self).__init__()
        loadUi("arayuz.ui", self)
       
       
        self.adet=0
        self.sayi=0
        self.s=""
       
        self.pushButton.clicked.connect(self.ekleme)
        self.pushButton_2.clicked.connect(self.gorsel)
        self.pushButton_3.clicked.connect(self.show_new_window)
        # self.pushButton_3.clicked.connect(self.showPredfgr)
        # self.pushButton_3.clicked.connect(self.showPredfgr2)
        # self.pushButton_3.clicked.connect(self.showPredfgr3)
        # self.pushButton_3.clicked.connect(self.showPredfgr4)
        # self.pushButton_3.clicked.connect(self.showPredfgr5)
        # self.pushButton_3.clicked.connect(self.showPredfgr6)
        self.pushButton_3.clicked.connect(self.regAlgm)
     
        self.pushButton_4.clicked.connect(self.derinOgrenme)
        self.pushButton_5.clicked.connect(self.ensembleLearning)
        self.allmodelss=[]
        self.w=None
        self.cm2 = [[0, 0], 
                    [0, 0]]

           
    def csvYukle(self,veridoldur):
        
        c=len(veridoldur.columns)
        r=len(veridoldur.values)
        self.tableWidget.setColumnCount(c)
        self.tableWidget.setRowCount(r)
        colmnames=["Cinsiyet","Yaşlı","Evli","Ekonomik Bagımlı","Abonelik Süresi","Telefon Hizmeti","Birden Fazla Hat",
                   "İnternet Servisi","Çevrimiçi Güvenlik","Çevrimiçi Yedekleme","Cihaz Koruma","Teknik Destek","Televizyon",
                   "Film","Sözleşme Süresi","Çevrimiçi Fatura","Ödeme Yöntemi","Aylık Ödeme","Toplam Ödeme","Müşteri Kaybı"]       
        self.tableWidget.setHorizontalHeaderLabels(colmnames)
        
        for i,row in enumerate(veridoldur):            
             for j,cell in enumerate(veridoldur.values):         
                  self.tableWidget.setItem(j,i, QtWidgets.QTableWidgetItem(str(cell[i])))  
        
    
    def ekleme(self):
      
        file_name, _ = QFileDialog.getOpenFileName(self, 'Open Image File', r".\Desktop", ".xls(*.xls) .csv(*.csv)")
        self.veriler = pd.read_csv(file_name)
        self.veriler.rename(columns={'customerID':'Müşteri ID','gender':'Cinsiyet','SeniorCitizen':'Yaşlı','Partner':'Evli',
                   'Dependents':'Ekonomik Bağımlı','tenure':'Abonelik Süresi','PhoneService':'Telefon Hizmeti',
                   'MultipleLines':'Birden Fazla Hat','InternetService':'İnternet Servisi',
                   'OnlineSecurity':'Çevrimiçi Güvenlik','OnlineBackup':'Çevrimiçi Yedekleme',
                   'DeviceProtection':'Cihaz Koruma','TechSupport':'Teknik Destek','StreamingTV':'Televizyon',
                   'StreamingMovies':'Film','Contract':'Sözleşme Süresi','PaperlessBilling':'Çevrimiçi Fatura',
                   'PaymentMethod':'Ödeme Yöntemi','MonthlyCharges':'Aylık Ödeme','TotalCharges':'Toplam Ödeme',
                   'Churn':'Müşteri Kaybı'},inplace=True)
        
        self.veriler.drop('Müşteri ID', axis=1, inplace=True)
        self.veriler["Müşteri Kaybı"]=  self.veriler["Müşteri Kaybı"].replace("No","Müşteri Kaybı Yok") 
        self.veriler["Müşteri Kaybı"]=  self.veriler["Müşteri Kaybı"].replace("Yes","Müşteri Kaybı Var")
        self.veriler["Yaşlı"]= self.veriler["Yaşlı"].replace(0, "No") 
        self.veriler["Yaşlı"]= self.veriler["Yaşlı"].replace(1, "Yes") 
        # print(self.veriler)
        self.veriler['Toplam Ödeme'] = pd.to_numeric( self.veriler['Toplam Ödeme'], errors='coerce')
        self.veriler['Toplam Ödeme'] =  self.veriler['Toplam Ödeme'].fillna(value=0)
        self.veriler['Yaşlı'] =  self.veriler['Yaşlı'].astype('object')
             
        self.csvYukle(self.veriler)
       
        
                     
    def labeEnc(self):
        # print("veri",len(self.veriler))
        from sklearn.preprocessing import LabelEncoder
        self.encoded = self.veriler.apply(lambda x: LabelEncoder().fit_transform(x) if x.dtype == 'object' else x)
    
    def aykiriVeri(self):

            Müşteri_Kaybı_Yaşandı=self.encoded.loc[self.encoded['Müşteri Kaybı'].abs()>0]
            # print(Müşteri_Kaybı_Yaşandı)
            Q1 = Müşteri_Kaybı_Yaşandı['Toplam Ödeme'].quantile(0.25)
            Q3 = Müşteri_Kaybı_Yaşandı['Toplam Ödeme'].quantile(0.75)
            IQR = Q3 - Q1
            Q=Q3+(1.5*IQR)
           
            encoded_out = self.encoded[~((self.encoded['Toplam Ödeme'] < (Q3 + 1.5 * IQR)))&(self.encoded['Müşteri Kaybı']>0)]
            # print(encoded_out.head(8000))
            # Aykırı veriler çağırıldı.Grafikte gözlemlediğimiz yoğunluktaki aykırı veri sayısı 109'muş.
            self.encoded.drop(self.encoded[~((self.encoded['Toplam Ödeme'] < (Q3 + 1.5 * IQR)))&(self.encoded['Müşteri Kaybı']>0)].index, inplace=True)
            # print(self.encoded.head(8000))
            
            Q1_A = Müşteri_Kaybı_Yaşandı['Abonelik Süresi'].quantile(0.25)
            Q3_A = Müşteri_Kaybı_Yaşandı['Abonelik Süresi'].quantile(0.75)
            IQR_A = Q3_A - Q1_A
            # print( IQR_A)
            Q_A=Q3_A+(1.5*IQR_A)
            # print(Q_A)
            encoded_A_out = self.encoded[~((self.encoded['Abonelik Süresi'] < (Q3_A + 1.5 * IQR_A)))&(self.encoded['Müşteri Kaybı']>0)]
            # print(encoded_A_out.head(8000))
            self.encoded.drop(self.encoded[~((self.encoded['Abonelik Süresi'] < (Q3_A + 1.5 * IQR_A)))&(self.encoded['Müşteri Kaybı']>0)].index, inplace=True)
            # print("enc",len(self.encoded))
         

    def pcaIslem(self,x_train,x_test,sayi):
        from sklearn.decomposition import PCA
        pca = PCA(n_components = sayi)
        
        X_train2 = pca.fit_transform(x_train)
        X_test2 = pca.transform(x_test)
        return X_train2,X_test2  
    
    
     # self.veriler.info()
      # print( self.veriler.head(8000))
      # print(self.veriler.describe())
      
      # self.veriler.loc[:, 'Müşteri Kaybı'].value_counts()
      # print(self.veriler.isnull().sum())  
    def gorsel(self):
     
         
            
      
            sns.countplot(x = "Müşteri Kaybı", data = self.veriler)
            Kategorik = self.veriler.select_dtypes(include='object').drop('Müşteri Kaybı', axis=1).columns.tolist()
            # Sayısal = self.veriler.select_dtypes(exclude='object').columns.tolist()
         
            plt.figure(figsize=(18,18))
            for i,c in enumerate(Kategorik):
                plt.subplot(5,4,i+1)
                sns.countplot(self.veriler[c], hue=self.veriler['Müşteri Kaybı'])
                plt.title(c)
                plt.xlabel('')
       
            plt.figure(figsize=(20,5))
            for i,c in enumerate(['Abonelik Süresi', 'Aylık Ödeme', 'Toplam Ödeme']):
                plt.subplot(1,3,i+1)
                sns.distplot(self.veriler[self.veriler['Müşteri Kaybı'] == 'Müşteri Kaybı Yok'][c], kde=True, color='blue', hist=False, kde_kws=dict(linewidth=2), label='Müşteri Kaybı Oluşmadı')
                sns.distplot(self.veriler[self.veriler['Müşteri Kaybı'] == 'Müşteri Kaybı Var'][c], kde=True, color='Orange', hist=False, kde_kws=dict(linewidth=2), label='Müşteri Kaybı Oluştu')
                plt.title(c)
                
            plt.show()
          
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import roc_curve
            from sklearn.metrics import roc_auc_score
            from matplotlib import pyplot
            from sklearn.datasets import make_classification
            from sklearn.naive_bayes import GaussianNB
            x,y=self.onHazirlik()
            test_size1=float(self.lineEdit.text())
            trainx, testx, trainy, testy = train_test_split(x, y, test_size=test_size1, random_state=42)
            NBG = GaussianNB()
              
            from sklearn.naive_bayes import BernoulliNB
            BNN = BernoulliNB()
            
            from sklearn.neighbors import KNeighborsClassifier
            KNN= KNeighborsClassifier(n_neighbors=5)

            from sklearn.linear_model import LogisticRegression  
            Logistic_Regression = LogisticRegression(C=0.5,tol=0.1,multi_class='multinomial',solver='newton-cg',penalty='l2',max_iter=100)      
            
            from sklearn.svm import SVC   
            svc1 = SVC(probability=True)
                  
            x, y = make_classification(n_samples=1000, n_classes=2, random_state=1)          
            KNN_tahmin = [0 for _ in range(len(testy))]
            NBG_tahmin = [0 for _ in range(len(testy))]
            svc1_tahmin = [0 for _ in range(len(testy))]
            Logistic_Regression_tahmin = [0 for _ in range(len(testy))]
            BNN_tahmin = [0 for _ in range(len(testy))]
            
            from sklearn.preprocessing import StandardScaler
            sc_X=StandardScaler()
            trainx=sc_X.fit_transform(trainx)
            testx=sc_X.transform(testx)
            
            model = NBG
            model.fit(trainx, trainy)
            self.allmodelss.append(model)
            model2 = KNN
            model2.fit(trainx, trainy)
            self.allmodelss.append(model2)
            model3=svc1
            model3.fit(trainx, trainy)
            self.allmodelss.append(model3)
            model4=Logistic_Regression
            model4.fit(trainx, trainy)
            self.allmodelss.append(model4)
            model5=BNN
            model5.fit(trainx, trainy)
            self.allmodelss.append(model5)
            NBG_tahmin = model.predict_proba(testx)
            KNN_tahmin = model2.predict_proba(testx)
            svc1_tahmin = model3.predict_proba(testx)
            Logistic_Regression_tahmin= model4.predict_proba(testx)
            BNN_tahmin= model5.predict_proba(testx)
            
            NBG_tahmin = NBG_tahmin[:, 1]
            KNN_tahmin = KNN_tahmin[:, 1]
            svc1_tahmin = svc1_tahmin[:, 1]
            Logistic_Regression_tahmin = Logistic_Regression_tahmin[:, 1]
            BNN_tahmin= BNN_tahmin[:, 1]
            
            KNN_hassasiyet = roc_auc_score(testy, KNN_tahmin)
            NBG_hassasiyet = roc_auc_score(testy, NBG_tahmin)
            svc1_hassasiyet = roc_auc_score(testy, svc1_tahmin)
            Logistic_Regression_hassasiyet = roc_auc_score(testy, Logistic_Regression_tahmin)
            BNN_hassasiyet= roc_auc_score(testy, BNN_tahmin)
            
            print('KNN: ROC AUC=%.3f' % (KNN_hassasiyet))
            print('Navy Bayes Gaussian: ROC AUC=%.3f' % (NBG_hassasiyet))
            print('SVC: ROC AUC=%.3f' % (svc1_hassasiyet))
            print('Logistic Regresyon: ROC AUC=%.3f' % (Logistic_Regression_hassasiyet))
            print('BNN: ROC AUC=%.3f' % (BNN_hassasiyet))
            
            
            KNN_fpr, KNN_tpr, _ = roc_curve(testy, KNN_tahmin)
            NBG_fpr, NBG_tpr, _ = roc_curve(testy, NBG_tahmin)
            svc1_fpr, svc1_tpr, _ = roc_curve(testy, svc1_tahmin)
            Logistic_Regression_fpr, Logistic_Regression_tpr, _ = roc_curve(testy, Logistic_Regression_tahmin)
            BNN_fpr, BNN_tpr, _ = roc_curve(testy, BNN_tahmin)
            
            pyplot.plot(KNN_fpr, KNN_tpr, linestyle='--', label='KNN')
            pyplot.plot(NBG_fpr, NBG_tpr, marker='.', label='Navy Bayes Gaussian')
            pyplot.plot(svc1_fpr, svc1_tpr, marker='.', label='SVC')
            pyplot.plot(Logistic_Regression_fpr, Logistic_Regression_tpr, marker='.', label='Logistic Regresyon')
            pyplot.plot(BNN_fpr, BNN_tpr, marker='.', label='BNN')
            
            pyplot.xlabel('Gerçek Müşteri Kaybı Oluştu')
            pyplot.ylabel('Gerçek Müşteri Kaybı Oluşmadı')
            
            pyplot.legend()
            
            pyplot.show()
         
            print("Başarı Değerleri")         
            model_accuracy = pd.Series(data=[KNN_hassasiyet,NBG_hassasiyet,svc1_hassasiyet,Logistic_Regression_hassasiyet,BNN_hassasiyet], 
            index=['Knn','GaussianNB','Svm','Logistic','Bnn'])
            fig= plt.figure(figsize=(8,8))
            model_accuracy.sort_values().plot.barh()
            plt.title('Model Accuracy')
            plt.show()
            print("--------------------")
        
            
    def onHazirlik(self):
            self.labeEnc()
            self.aykiriVeri()           

            x = self.encoded.drop('Müşteri Kaybı', axis = 1)              
            y = self.encoded['Müşteri Kaybı'] 
            return x,y
    

        
    def get_stacking(self):
            from sklearn.linear_model import LogisticRegression
            from sklearn.neighbors import KNeighborsClassifier
            from sklearn.svm import SVC
            from sklearn.naive_bayes import GaussianNB
            from sklearn.ensemble import StackingClassifier
            from sklearn.naive_bayes import BernoulliNB
            
            level0 = list()
            level0.append(('lr', LogisticRegression(C=0.5,tol=0.1,multi_class='multinomial',solver='newton-cg',penalty='l2',max_iter=100)))
            level0.append(('knn', KNeighborsClassifier(n_neighbors=5)))
            level0.append(('benoull', BernoulliNB()))
            level0.append(('svm', SVC(kernel='rbf')))
            level0.append(('bayes', GaussianNB()))
        	# define meta learner model
            level1 =LogisticRegression()
        	# define the stacking ensemble
            model = StackingClassifier(estimators=level0, final_estimator=level1, cv=5)
            return model
            
           
    def get_models(self):
            from sklearn.model_selection import RepeatedStratifiedKFold
            from sklearn.linear_model import LogisticRegression
            from sklearn.neighbors import KNeighborsClassifier
            from sklearn.svm import SVC
            from sklearn.naive_bayes import GaussianNB
            from sklearn.naive_bayes import BernoulliNB
        
           
            models = dict()
            models['lr'] = LogisticRegression(C=0.5,tol=0.1,multi_class='multinomial',solver='newton-cg',penalty='l2',max_iter=100)
            models['knn'] = KNeighborsClassifier(n_neighbors=5)
            models['benoull'] = BernoulliNB()
            models['svm'] = SVC(kernel='rbf')
            models['bayes'] = GaussianNB()
            models['stacking'] = self.get_stacking()
       	    return models
           

    def evaluate_model(self,model, X, y):
        # from sklearn.model_selection import RepeatedKFold
        from sklearn.model_selection import cross_val_score
        from sklearn.model_selection import RepeatedStratifiedKFold
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
        scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
       
        return scores  
    
    def ensembleLearning(self):
        from sklearn.linear_model import LogisticRegression
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.svm import SVC
        from sklearn.naive_bayes import GaussianNB
        from sklearn.naive_bayes import BernoulliNB
        from numpy import mean
        from numpy import std      
        from matplotlib import pyplot 
        test_size1=float(self.lineEdit.text())
        X, y = self.onHazirlik()   
        # print("StackingClassifier")
      
        # models = self.get_models()
        # # evaluate the models and store results
        # results, names = list(), list()
        # for name, model in models.items():
   
        #     scores = self.evaluate_model(model, X, y)
        #     results.append(scores)
        #     names.append(name)
        #     self.label_30.setText(str(round(mean(scores),3)))
        #     print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
        # # plot model performance for comparison
        # pyplot.boxplot(results, labels=names, showmeans=True)
        # pyplot.show()
      
        
        
        print("**********")
        print("BaggingClassifier")
        from sklearn.ensemble import BaggingClassifier
        from sklearn import tree
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size =test_size1 , random_state = 42)
        model=BaggingClassifier(tree.DecisionTreeClassifier(random_state=1))
        model.fit(x_train, y_train)
        y_test=np.array(y_test)
        y_pred=model.predict(x_test)
        print(round(accuracy_score(y_test,y_pred),3))
        self.label_33.setText(str(round(accuracy_score(y_test,y_pred),3)))
        self.Cmatrix(y_test,y_pred,"Bagging")
        self.pltRoc2(y_test,y_pred,"Bagging")
        print("**********")
        print("VotingClassifier")
      
        from sklearn.linear_model import LogisticRegression
        from sklearn.naive_bayes import GaussianNB
        from sklearn.ensemble import RandomForestClassifier, VotingClassifier
        clf1 = LogisticRegression(C=0.5,tol=0.1,multi_class='multinomial',solver='newton-cg',penalty='l2',max_iter=100)
        clf2 = KNeighborsClassifier(n_neighbors=5)
        clf3 = GaussianNB()
        clf4=  SVC(probability=True)
        clf5=  BernoulliNB()
      
   
        X = X
        y = y
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size =test_size1 , random_state = 42)
        eclf1 = VotingClassifier(estimators=[
                ('lr', clf1), ('knn', clf2), ('gnb', clf3),('svc', clf4),('bnn', clf5)], voting='soft')
        eclf1 = eclf1.fit(x_train, y_train)
        
        np.array_equal(eclf1.named_estimators_.lr.predict(X),
                       eclf1.named_estimators_['lr'].predict(X))
        
        eclf2 = VotingClassifier(estimators=[
                ('lr', clf1), ('knn', clf2), ('gnb', clf3),('svc', clf4),('bnn', clf5)],
                voting='soft')
        eclf2 = eclf2.fit(x_train,y_train)
        
        eclf3 = VotingClassifier(estimators=[
               ('lr', clf1), ('knn', clf2), ('gnb', clf3),('svc', clf4),('bnn', clf5)],
               voting='soft', weights=[1,1,1,1,1],
               flatten_transform=True)
        eclf3 = eclf3.fit(x_train,y_train)
        # print(eclf3.predict(X))
        y_pred=eclf3.predict(x_test)
        score = accuracy_score(y_test, y_pred) 
        print(round(score,3))
        self.label_29.setText(str(round(score,3)))
        self.Cmatrix(y_test,y_pred,"Voting")
        self.pltRoc2(y_test,y_pred,"Voting")
        print("**********")
        print("XGBClassifier") 
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size =test_size1 , random_state = 42)
        import xgboost as xgb
        model=xgb.XGBClassifier(random_state=1,learning_rate=0.01)
        model.fit(x_train, y_train)
        y_test=np.array(y_test)
        y_pred=model.predict(x_test)
        score = accuracy_score(y_test, y_pred) 
        print(round(score,3))
        self.label_34.setText(str(round(score,3)))
        self.Cmatrix(y_test,y_pred,"XGB")
        self.pltRoc2(y_test,y_pred,"XGB")
        print("--------------------")
        
    def show_new_window(self):
       
        self.w = Windoww2()
        self.w.show() 
        
        
    def derinOgrenme(self):
            a=Windoww()
            x,y=self.onHazirlik()
            test_size1=float(self.lineEdit.text())
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size =test_size1 , random_state = 42)
            from keras.utils import to_categorical 
            y_train = to_categorical(y_train, 2)
            y_test= to_categorical(y_test, 2)
            
          
            from keras.models import Sequential
            from keras.layers import Dense,Dropout,BatchNormalization,Activation
            #modeli oluşturalım
            model = Sequential()
            #eğitim verisinde kaç tane stun yani model için girdi sayısı var onu alalım
            n_cols = x_train.shape[1]
            #model katmanlarını ekleyelim
            model.add(Dense(16, input_shape=(n_cols,)))
            model.add(Activation("relu"))
            model.add(BatchNormalization())
            model.add(Dropout(0.5))
            model.add(Dense(9))
            model.add(Activation("relu"))
            model.add(BatchNormalization())
            model.add(Dropout(0.5))
            model.add(Dense(6))
            model.add(Activation("relu"))
            model.add(BatchNormalization())
            model.add(Dropout(0.5))
            model.add(Dense(2, activation='softmax'))
            model.summary()
            
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            
            
            
            history =model.fit(x_train, 
            y_train,
            validation_data=(x_test, y_test),
            batch_size=16, 
            shuffle=True,
            verbose=1,
            epochs=10)
            
            
            score = model.evaluate(x_test, y_test, verbose=0)
            print('Test loss:', score[0])
            print('Test accuracy:', score[1])
            from matplotlib import pyplot as plt
            # Plot training & validation accuracy values
            # plt.figure(figsize=(14,3))
            plt.subplot(1, 2, 1)
               
            plt.plot(history.history['accuracy'])
            plt.plot(history.history['val_accuracy'])
            plt.title('Model accuracy')
            plt.ylabel('Accuracy')
            plt.xlabel(str(round(score[1]*100,3)))
            plt.legend(['Train', 'Test'], loc='upper left')
            # Plot training & validation loss values
            
            plt.subplot(1, 2, 2)      
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('Model loss')
            plt.ylabel('Loss')
            plt.xlabel(str(round(score[0],3)))
            plt.legend(['Train', 'Test'], loc='upper left')
            plt.show()
            a.show()
            # y_true = np.array(y_test)
            # y_pred2 = np.squeeze(np.array(model.predict(x_test) ,dtype=np.int))
            # y_pred = model.predict(x_test)
            # y_test = y_test.reshape(-1, 1)
            # y_pred=y_pred.reshape(-1, 1)
            # print(round(accuracy_score(y_true, y_pred2), 2))
            # print("acc",accuracy_score(y_test, y_pred.round(), normalize=False))
            print('----Sonuç-----')
            
            
          
            score = model.evaluate(x_test, y_test, verbose=0)
            print('Test loss:', score[0])
            print('Test accuracy:', score[1])
            
            # x_test = x_test.reshape(-1, 1)
           
           
            # X.reshape(-1, 1)
            # from sklearn import preprocessing
            # le = preprocessing.LabelEncoder()
            # le.fit(y_test)
            # y_test=le.transform(y_test)
            # from sklearn import preprocessing
            # le = preprocessing.LabelEncoder()
            # le.fit(y_pred)
            # y_pred=le.transform(y_pred)
            # y_test= to_categorical(y_test, 1)
            # print(y_test.shape)
            # print("2313")
            # print(y_pred.shape)
     
            # print(y_pred)
            # print(y_test)
            # print(y_test.shape)
            # print(y_pred.shape)
            # print(len(y_test))
            # print(len(y_pred))
            y_pred = model.predict(x_test)
            y_test = y_test.reshape(-1, 1)
            y_pred=y_pred.reshape(-1, 1)
            
            # print(y_pred)
            # print(y_test)
            # print(y_test.shape)
            # print(y_pred.shape)
            # print(len(y_test))
            # print(len(y_pred))
            # print(y_pred.round())
           
            print(confusion_matrix(y_test, y_pred.round()))
            y_pred2=y_pred.round()
            self.Cmatrix(y_test,y_pred2,"Derin Öğrenme")
            self.pltRoc2(y_test,y_pred,"Derin Öğrenme")
            
            # from sklearn.metrics import roc_curve
            # from sklearn.metrics import roc_auc_score
            # from matplotlib import pyplot
            # lr_auc = roc_auc_score(y_test, y_pred)
            # # summarize scores
      
            # print('CNN: ROC AUC=%.3f' % (lr_auc))
            # # calculate roc curves
            
            # lr_fpr, lr_tpr, _ = roc_curve(y_test, y_pred)
            # # plot the roc curve for the model
           
            # pyplot.plot(lr_fpr, lr_tpr, marker='.', label='CNN')
            # # axis labels
            # pyplot.xlabel('False Positive Rate')
            # pyplot.ylabel('True Positive Rate')
            # # show the legend
            # pyplot.legend()
            # pyplot.show()
            print("--------------------")
            # show the plot
    def showPredfgr4(self,y_test,y_pred,isim):  
        self.predfigr4 = Windoww()
        cm = confusion_matrix(y_test, y_pred)
        classNames = ['VAR','YOK']
        cm_data = pd.DataFrame(cm,index = classNames,
                      columns = classNames)
        # plt.figure(figsize = (5,4))
        sns.heatmap(cm_data, annot=True,fmt="d")
        plt.title(isim)
        plt.ylabel('Actual label')
        plt.xlabel('Predicted label')
        plt.show()
        self.predfigr4.show() 
    def Cmatrix(self,y_test,y_pred,isim):
           
            # a6=Windoww()
            self.showPredfgr4(y_test,y_pred,isim)
        
            # a6.show()
            
            
            
    def showPredfgr6(self,y_test,y_pred,isim):  
        self.predfigr6= Windoww()
        cm = confusion_matrix(y_test, y_pred)
        classNames = ['VAR','YOK']
        cm_data = pd.DataFrame(cm,index = classNames,
                      columns = classNames)
        # plt.figure(figsize = (5,4))
        sns.heatmap(cm_data, annot=True,fmt="d")
        plt.title(isim)
        plt.ylabel('Actual label')
        plt.xlabel('Predicted label')
        plt.show()
        self.predfigr6.show() 
    def Cmatrix2(self,y_test,y_pred,isim):
           
            # a6=Windoww()
            self.showPredfgr6(y_test,y_pred,isim)
        
            # a6.show()
           
    def CmatrixFold(self,y_test,y_pred,isim):
           
       
           
            cm = confusion_matrix(y_test, y_pred)
            classNames = ['VAR','YOK']
            cm_data = pd.DataFrame(cm,index = classNames,
                          columns = classNames)
            # plt.figure(figsize = (5,4))
            sns.heatmap(cm_data, annot=True,fmt="d")
            plt.title(isim)
            plt.ylabel('Actual label')
            plt.xlabel('Predicted label')
            plt.show()
            
    def showPredfgr5(self,baslik):       
        self.predfigr5 = Windoww()
    
        classNames = ['0', '1']
        cm_data = pd.DataFrame(self.cm2, index=classNames,columns=classNames)
        # plt.figure(figsize=(5, 4))
        sns.heatmap(cm_data, annot=True, fmt="d")
        plt.title(baslik)
        plt.ylabel('Actual label')
        plt.xlabel('Predicted label')
        plt.show()
        self.predfigr5.show()        
            
    def kfoldCmatrix(self, y_test, y_pred,baslik):
        if self.foldsizee==int(self.lineEdit_2.text()):
            self.cm2=[[0, 0], 
                  [0, 0]]
        
        cm = confusion_matrix(y_test, y_pred)
        if self.foldsizee !=0:
            self.cm2 += cm
            self.foldsizee -=1
        if(self.foldsizee == 0):
            # a1=Windoww()
            self.showPredfgr5(baslik)
            # a1.show()
        
        
        
        
    def showPredfgr(self,y_test,y_pred):       
        self.predfigr1 = Windoww()
        print(">>TAHMİN<<")
        residuals = y_test-y_pred 
        sns.countplot(x =y_pred, data = residuals, color = 'red')
        plt.show()
        self.predfigr1.show()   
        
    def pltPredict(self,y_test,y_pred):
            # a2=Windoww()
            self.showPredfgr(y_test,y_pred)
          
          
            # a2.show()
     
        
     
          
    def showPredfgr2(self,y_test):       
        self.predfigr2 = Windoww()
    
        print(">>GERÇEK<<")
        sns.countplot(x =y_test, data = None, color = 'blue')
        plt.show()
        self.predfigr2.show()   
        
    def pltTrue(self,y_test):
            # a3=Windoww()
            self.showPredfgr2(y_test)
          
            # a3.show()
           
            
    def showEnsm(self,y_test,y_pred,baslik):       
        self.predfigr3 = Windoww()
        from sklearn.metrics import roc_curve
        from sklearn.metrics import roc_auc_score
        from matplotlib import pyplot
        lr_auc = roc_auc_score(y_test, y_pred)
        # summarize scores
  
        print('ALGRTM: ROC AUC=%.3f' % (lr_auc))
        # calculate roc curves
        
        lr_fpr, lr_tpr, _ = roc_curve(y_test, y_pred)
        # plot the roc curve for the model
       
        pyplot.plot(lr_fpr, lr_tpr, marker='.', label=baslik)
        # axis labels
        pyplot.xlabel('False Positive Rate')
        pyplot.ylabel('True Positive Rate')
        # show the legend
        pyplot.legend()
        pyplot.show()
        self.predfigr3.show()   
        
    def pltRoc2(self,y_test,y_pred,baslik):
            # a4=Windoww()
            self.showEnsm(y_test,y_pred,baslik)   
            
            
            
    def showPredfgr3(self,y_test,y_pred):       
        self.predfigr3 = Windoww()
        from sklearn.metrics import roc_curve
        from sklearn.metrics import roc_auc_score
        from matplotlib import pyplot
        lr_auc = roc_auc_score(y_test, y_pred)
        # summarize scores
  
        print('ALGRTM: ROC AUC=%.3f' % (lr_auc))
        # calculate roc curves
        
        lr_fpr, lr_tpr, _ = roc_curve(y_test, y_pred)
        # plot the roc curve for the model
       
        pyplot.plot(lr_fpr, lr_tpr, marker='.', label='ALGRTMA')
        # axis labels
        pyplot.xlabel('False Positive Rate')
        pyplot.ylabel('True Positive Rate')
        # show the legend
        pyplot.legend()
        pyplot.show()
        self.predfigr3.show()   
        
    def pltRoc(self,y_test,y_pred):
            # a4=Windoww()
            self.showPredfgr3(y_test,y_pred)
            
            # a4.show()
            
            
    def regAlgm(self):
        from sklearn.model_selection import cross_val_score
        sec=self.comboBox.currentText()
        if sec=='GaussianNB':
            x,y=self.onHazirlik()
            from sklearn.naive_bayes import GaussianNB
            test_size1=float(self.lineEdit.text())
            fold_size=int(self.lineEdit_2.text())
            self.foldsizee=fold_size
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = test_size1, random_state = 42)
            # yenipncre=Windoww2()
            self.w.csvYukle(x_train,y_train,x_test,y_test)
            # yenipncre.show()
          
            
            self.label_22.setText(str(len(x_train)))
            self.label_23.setText(str(len(y_train)))
            self.label_24.setText(str(len(x_test)))
            self.label_25.setText(str(len(x_test)))
            
            from sklearn.preprocessing import StandardScaler
            sc_X=StandardScaler()
            x_train=sc_X.fit_transform(x_train)
            x_test=sc_X.transform(x_test)
            
            NBG = GaussianNB()
            NBG.fit(x_train, y_train)
            y_pred = NBG.predict(x_test)
            self.pltPredict(y_test, y_pred)
            self.pltTrue(y_test)
            
                
            acc=accuracy_score(y_test, y_pred)*100
                 
            acc=accuracy_score(y_test, y_pred)*100
            self.label_5.setText(str(round(acc,2)))
            self.label_4.setText(str(round(r2_score(y_test,y_pred),2)))
            
            X_train2,X_test2 =self.pcaIslem(x_train,x_test,5)            
            classifier2 =  GaussianNB()
            classifier2.fit(X_train2,y_train)
            y_pred2=classifier2.predict(X_test2)
            self.Cmatrix2(y_test,y_pred2,"PCA")
            acc=accuracy_score(y_test, y_pred2)*100
            self.label_9.setText(str(round(acc,2)))
            self.label_8.setText(str(round(r2_score(y_test,y_pred2),2)))
           
            self.Cmatrix(y_test,y_pred,"Gaussian")
            print(confusion_matrix(y_test, y_pred))
            self.pltRoc2(y_test,y_pred,"Gaussian")
      
            from sklearn.model_selection import KFold
            from numpy import mean
            from sklearn.model_selection import cross_val_score     
            X = x.values
            y = y.values
            kf = KFold(n_splits=fold_size)
            kf.get_n_splits(X)
            
            sayma=0
            for train_index, test_index in kf.split(X):
                sayma+=1
                # print("TRAIN:", train_index, "TEST:", test_index)
                x_train, x_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                NBG = GaussianNB()
                NBG.fit(x_train, y_train)
                y_pred = NBG.predict(x_test)              
                acc=accuracy_score(y_test, y_pred)*100
                self.CmatrixFold(y_test,y_pred,"fold-"+str(sayma))
                self.kfoldCmatrix(y_test, y_pred,"Fold Sonuç")
                
                print(acc)
             

                                  
            model = GaussianNB()       
            scores = cross_val_score(model, X, y, scoring='accuracy', cv=kf, n_jobs=-1)
            self.label_12.setText(str(round(mean(scores*100),2)))
            self.label_13.setText(str(round(scores.max()*100,2)))          
            print('Accuracy: %.3f (%.3f)' % (mean(scores), scores.max()))
            print("--------------------")
            
        
            # dogru
            
            # from sklearn.model_selection import GridSearchCV, RandomizedSearchCV     
            # from sklearn.pipeline import Pipeline
     
            # pipeline = Pipeline([
            #   ('clf', GaussianNB())
            #                 ])
    
            # parameters = {
            #   'clf__priors': [None],
            #   'clf__var_smoothing': [0.00000001]
            #     }
    
            # cv = GridSearchCV(pipeline, param_grid=parameters)           
            # cv.fit(x_train, y_train)
            # print(cv.best_params_)
            # self.label_38.setText(str(cv.best_params_))
            
            
        ###############################         
        
         
            

        if sec=="BernoulliNB":
            x,y=self.onHazirlik()
            
            test_size1=float(self.lineEdit.text())
            fold_size=int(self.lineEdit_2.text())
            self.foldsizee=fold_size
           
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = test_size1, random_state = 42)
            self.w.csvYukle(x_train,y_train,x_test,y_test)
            self.label_22.setText(str(len(x_train)))
            self.label_23.setText(str(len(y_train)))
            self.label_24.setText(str(len(x_test)))
            self.label_25.setText(str(len(x_test)))
            
            from sklearn.preprocessing import StandardScaler
            sc_X=StandardScaler()
            x_train=sc_X.fit_transform(x_train)              
            x_test=sc_X.fit_transform(x_test)
            
            
            from sklearn.naive_bayes import BernoulliNB
            NBB = BernoulliNB()
            NBB.fit(x_train, y_train)
            y_pred = NBB.predict(x_test)
            self.pltPredict(y_test, y_pred)
            self.pltTrue(y_test)
            
            acc=accuracy_score(y_test, y_pred)*100
            self.label_5.setText(str(round(acc,2)))
            self.label_4.setText(str(round(r2_score(y_test,y_pred),2)))
            
            X_train2,X_test2 =self.pcaIslem(x_train,x_test,5)            
            classifier2 = BernoulliNB()
            classifier2.fit(X_train2,y_train)
            y_pred2=classifier2.predict(X_test2)
            self.Cmatrix2(y_test,y_pred2,"PCA")
            acc=accuracy_score(y_test, y_pred2)*100
            self.label_9.setText(str(round(acc,2)))
            self.label_8.setText(str(round(r2_score(y_test,y_pred2),2)))
        
            self.Cmatrix(y_test,y_pred,"BernoulliNB")
            print(confusion_matrix(y_test, y_pred))
            self.pltRoc2(y_test,y_pred,"BernoulliNB")
            
            from sklearn.model_selection import KFold
            from numpy import mean
            from sklearn.model_selection import cross_val_score     
            X = x.values
            y = y.values
            kf = KFold(n_splits=fold_size)
            kf.get_n_splits(X)
            
            sayma=0
            for train_index, test_index in kf.split(X):
                sayma+=1
                # print("TRAIN:", train_index, "TEST:", test_index)
                x_train, x_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                NBG = BernoulliNB()
                NBG.fit(x_train, y_train)
                y_pred = NBG.predict(x_test)  
                self.CmatrixFold(y_test,y_pred,"fold-"+str(sayma))
                self.kfoldCmatrix(y_test, y_pred,"Fold Sonuç")
                acc=accuracy_score(y_test, y_pred)*100
                print(acc)

                              
            model = BernoulliNB()
            scores = cross_val_score(model, X, y, scoring='accuracy', cv=kf, n_jobs=-1)
            self.label_12.setText(str(round(mean(scores*100),2)))
            self.label_13.setText(str(round(scores.max()*100,2)))          
            print('Accuracy: %.3f (%.3f)' % (mean(scores), scores.max()))
            
            print("--------------------")
            
            
            
            # dogru
            # from sklearn.model_selection import GridSearchCV, RandomizedSearchCV     
            # from sklearn.pipeline import Pipeline
            # from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
           
            # pipeline = Pipeline(steps=[
              
            #   ('bnb', BernoulliNB())
            #   ])

    
            # parameters = {
              
            #     'bnb__fit_prior': [False,True],
            #     'bnb__alpha': [0.0,0.5,1.0],

            #   }
            # x,y=self.onHazirlik()
     
            # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.33, random_state = 42)
            # cv = GridSearchCV(pipeline, param_grid=parameters)           
            # cv.fit(x, y)
            # print(cv.best_params_)
            # self.label_38.setText(str(cv.best_params_))
        
            
            # hata########################
            # import pandas as pd
            # import numpy as np
            # from sklearn import preprocessing
            # from sklearn.model_selection import GridSearchCV, RandomizedSearchCV 
            # from sklearn.svm import SVC as svc 
            # from sklearn.metrics import make_scorer, roc_auc_score
            # from scipy import stats
             
            # # DATA PREPARATION
       
            # # DEFINE MODEL AND PERFORMANCE MEASURE
            # mdl = BernoulliNB()
            # auc = make_scorer(roc_auc_score)
             
            # # GRID SEARCH FOR 20 COMBINATIONS OF PARAMETERS
            # grid_list = {"C": np.arange(2, 10, 2),
            #               "gamma": np.arange(0.1, 1, 0.2)}
             
            # grid_search = GridSearchCV(mdl, param_grid = grid_list, n_jobs = 4, cv = 3, scoring = auc) 
            # grid_search.fit(x, y) 
            # print(grid_search.cv_results_)
             
            # # RANDOM SEARCH FOR 20 COMBINATIONS OF PARAMETERS
            # rand_list = {"C": stats.uniform(2, 10),
            #               "gamma": stats.uniform(0.1, 1)}
                          
            # rand_search = RandomizedSearchCV(mdl, param_distributions = rand_list, n_iter = 20, n_jobs = 4, cv = 3, random_state = 2017, scoring = auc) 
            # rand_search.fit(x, y) 
            # print(rand_search.cv_results_)
            
          
            

            
        if sec=="Knn":
            x,y=self.onHazirlik()
     
            test_size1=float(self.lineEdit.text())
            fold_size=int(self.lineEdit_2.text())
            self.foldsizee=fold_size
            
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = test_size1, random_state = 42)
            self.w.csvYukle(x_train,y_train,x_test,y_test)
            self.label_22.setText(str(len(x_train)))
            self.label_23.setText(str(len(y_train)))
            self.label_24.setText(str(len(x_test)))
            self.label_25.setText(str(len(x_test)))
            
            from sklearn.preprocessing import StandardScaler
            sc_X=StandardScaler()
            x_train=sc_X.fit_transform(x_train)              
            x_test=sc_X.fit_transform(x_test)
          
            from sklearn.neighbors import KNeighborsClassifier
            knn = KNeighborsClassifier(n_neighbors=5)
            knn.fit(x_train, y_train)
            y_pred= knn.predict(x_test)
            self.pltPredict(y_test, y_pred)
            self.pltTrue(y_test)
            
            acc=accuracy_score(y_test, y_pred)*100
            self.label_5.setText(str(round(acc,2)))
            self.label_4.setText(str(round(r2_score(y_test,y_pred),2)))
            
            X_train2,X_test2 =self.pcaIslem(x_train,x_test,5)            
            classifier2 = KNeighborsClassifier(n_neighbors=5)
            classifier2.fit(X_train2,y_train)
            y_pred2=classifier2.predict(X_test2)
            self.Cmatrix2(y_test,y_pred2,"PCA")
            acc=accuracy_score(y_test, y_pred2)*100
            self.label_9.setText(str(round(acc,2)))
            self.label_8.setText(str(round(r2_score(y_test,y_pred2),2)))
            self.Cmatrix(y_test,y_pred,"Knn")
            print(confusion_matrix(y_test, y_pred))
            self.pltRoc2(y_test,y_pred,"Knn")
            
            from sklearn.model_selection import KFold
            from numpy import mean
            from sklearn.model_selection import cross_val_score     
            X = x.values
            y = y.values
            kf = KFold(n_splits=fold_size)
            kf.get_n_splits(X)
            
            sayma=0
            for train_index, test_index in kf.split(X):
                sayma+=1
                # print("TRAIN:", train_index, "TEST:", test_index)
                x_train, x_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                NBG = KNeighborsClassifier(n_neighbors=5)
                NBG.fit(x_train, y_train)
                y_pred = NBG.predict(x_test)  
                self.CmatrixFold(y_test,y_pred,"fold-"+str(sayma))
                self.kfoldCmatrix(y_test, y_pred,"Fold Sonuç")
                acc=accuracy_score(y_test, y_pred)*100
                print(acc)

                                  
            model = KNeighborsClassifier(n_neighbors=5)      
            scores = cross_val_score(model, X, y, scoring='accuracy', cv=kf, n_jobs=-1)
            self.label_12.setText(str(round(mean(scores*100),2)))
            self.label_13.setText(str(round(scores.max()*100,2)))
            print('Accuracy: %.3f (%.3f)' % (mean(scores), scores.max()))
                        
            
            print("--------------------")
            
            # dogru grid
           
            # from sklearn.datasets import make_blobs
            # from sklearn.model_selection import RepeatedStratifiedKFold
            # from sklearn.model_selection import GridSearchCV
            # from sklearn.neighbors import KNeighborsClassifier
            #  # define dataset
            # X, y = x,y
            # # define models and parameters
            # model =KNeighborsClassifier(n_neighbors=5)      
            # n_neighbors = range(1, 21, 2)
            # weights = ['uniform', 'distance']
            # metric = ['euclidean', 'manhattan', 'minkowski']
            # # define grid search
            # grid = dict(n_neighbors=n_neighbors,weights=weights,metric=metric)
            # cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
            # grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
            # grid_result = grid_search.fit(X, y)
            #  # summarize results
            # print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
            # self.label_38.setText(str(grid_result.best_params_))
            # means = grid_result.cv_results_['mean_test_score']
            # stds = grid_result.cv_results_['std_test_score']
            # params = grid_result.cv_results_['params']
            # for mean1, stdev, param in zip(means, stds, params):
            #      print("%f (%f) with: %r" % (mean1, stdev, param))
           
            
           
            
        if sec=='Lojistik':
            x,y=self.onHazirlik()
     
            test_size1=float(self.lineEdit.text())
            fold_size=int(self.lineEdit_2.text())
            self.foldsizee=fold_size
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = test_size1, random_state = 42)
            self.w.csvYukle(x_train,y_train,x_test,y_test)
            self.label_22.setText(str(len(x_train)))
            self.label_23.setText(str(len(y_train)))
            self.label_24.setText(str(len(x_test)))
            self.label_25.setText(str(len(x_test)))
            
            from sklearn.preprocessing import StandardScaler
            sc=StandardScaler()
            x_train=sc.fit_transform(x_train)
            x_test=sc.fit_transform(x_test)
            
            from sklearn.linear_model import LogisticRegression
       
            Logistic_Regression = LogisticRegression(C=0.5,tol=0.1,multi_class='multinomial',solver='newton-cg',penalty='l2',max_iter=100)
            Logistic_Regression.fit(x_train, y_train)
            
            y_pred=Logistic_Regression.predict(x_test)
            
            self.pltPredict(y_test, y_pred)
            self.pltTrue(y_test)
          
            acc=accuracy_score(y_test, y_pred)*100
            print(acc)
        
            acc=accuracy_score(y_test, y_pred)*100
            self.label_5.setText(str(round(acc,2)))
            self.label_4.setText(str(round(r2_score(y_test,y_pred),2)))
            
            X_train2,X_test2 =self.pcaIslem(x_train,x_test,5)            
            classifier2 = LogisticRegression(C=0.5,tol=0.1,multi_class='multinomial',solver='newton-cg',penalty='l2',max_iter=100)
            classifier2.fit(X_train2,y_train)
            y_pred2=classifier2.predict(X_test2)
            
            self.Cmatrix2(y_test,y_pred2,"PCA")
            acc=accuracy_score(y_test, y_pred2)*100
            self.label_9.setText(str(round(acc,2)))
            self.label_8.setText(str(round(r2_score(y_test,y_pred2),2)))
            self.Cmatrix(y_test,y_pred,"Lojistik")
            print(confusion_matrix(y_test, y_pred))
            self.pltRoc2(y_test,y_pred,"Lojistik")
            
            from sklearn.model_selection import KFold
            from numpy import mean
            from sklearn.model_selection import cross_val_score     
            X = x.values
            y = y.values
            kf = KFold(n_splits=fold_size)
            kf.get_n_splits(X)
            
            sayma=0
            for train_index, test_index in kf.split(X):
                sayma+=1
                # print("TRAIN:", train_index, "TEST:", test_index)
                x_train, x_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                NBG =LogisticRegression(C=0.5,tol=0.1,multi_class='multinomial',solver='newton-cg',penalty='l2',max_iter=100)
                NBG.fit(x_train, y_train)
                y_pred = NBG.predict(x_test)
                self.CmatrixFold(y_test,y_pred,"fold-"+str(sayma))
                self.kfoldCmatrix(y_test, y_pred,"Fold Sonuç")
                acc=accuracy_score(y_test, y_pred)*100
                print(acc)

                                  
            model =LogisticRegression(C=0.5,tol=0.1,multi_class='multinomial',solver='newton-cg',penalty='l2',max_iter=100)   
            scores = cross_val_score(model, X, y, scoring='accuracy', cv=kf, n_jobs=-1)
       
            self.label_12.setText(str(round(mean(scores*100),2)))
            self.label_13.setText(str(round(scores.max()*100,2)))          
            print('Accuracy: %.3f (%.3f)' % (mean(scores), scores.max()))
            
            
            # https://machinelearningmastery.com/hyperparameter-optimization-with-random-search-and-grid-search/
            # dogru çalışıyor 
            
            # from scipy.stats import loguniform
            # from sklearn.linear_model import LogisticRegression
            # from sklearn.model_selection import RepeatedStratifiedKFold
            # from sklearn.model_selection import RandomizedSearchCV
     
            # X, y = x,y
            # # define model
            # model = LogisticRegression()
            # # define evaluation
            # cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
            # # define search space
            # space = dict()
            # space['solver'] = ['newton-cg', 'lbfgs', 'liblinear']
            # space['penalty'] = ['none', 'l1', 'l2', 'elasticnet']
            # space['C'] = loguniform(1e-5, 100)
            # # define search
            # search = RandomizedSearchCV(model, space, n_iter=50, scoring='accuracy', n_jobs=-1, cv=cv, random_state=1)
            # # execute search
            # result = search.fit(X, y)
            # # summarize result
            # print('Best Score: %s' % result.best_score_)
            # print('Best Hyperparameters: %s' % result.best_params_)
            # self.label_36.setText(str(result.best_params_))
              
            # print("--------------------")
              
            # ########grid
        # # dogru
            # from sklearn.linear_model import LogisticRegression
            # from sklearn.model_selection import RepeatedStratifiedKFold
            # from sklearn.model_selection import GridSearchCV
       
            # X, y = x,y
            # # define model
            # model = LogisticRegression()
            # # define evaluation
            # cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
            # # define search space
            # space = dict()
            # space['solver'] = ['newton-cg', 'lbfgs', 'liblinear']
            # space['penalty'] = ['none', 'l1', 'l2', 'elasticnet']
            # space['C'] = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]
            # # define search
            # search = GridSearchCV(model, space, scoring='accuracy', n_jobs=-1, cv=cv)
            # # execute search
            # result = search.fit(X, y)
            # # summarize result
            # print('Best Score: %s' % result.best_score_)
            # print('Best Hyperparameters: %s' % result.best_params_)
            # self.label_38.setText(str(result.best_params_))
            
        if sec=='Svc': 
                from sklearn.svm import SVC
                x,y=self.onHazirlik()
     
                test_size1=float(self.lineEdit.text())
                fold_size=int(self.lineEdit_2.text())
                self.foldsizee=fold_size
                x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = test_size1, random_state = 42)
                self.w.csvYukle(x_train,y_train,x_test,y_test)
                self.label_22.setText(str(len(x_train)))
                self.label_23.setText(str(len(y_train)))
                self.label_24.setText(str(len(x_test)))
                self.label_25.setText(str(len(x_test)))
                
                from sklearn.preprocessing import StandardScaler
                sc_X=StandardScaler()
                x_train=sc_X.fit_transform(x_train)              
                x_test=sc_X.fit_transform(x_test)
                
                NBB = SVC(kernel='rbf')
                NBB.fit(x_train, y_train)
                y_pred = NBB.predict(x_test)
                self.pltPredict(y_test, y_pred)
                self.pltTrue(y_test)
                
                acc=accuracy_score(y_test, y_pred)*100
                self.label_5.setText(str(round(acc,2)))
                self.label_4.setText(str(round(r2_score(y_test,y_pred),2)))
           
                
                X_train2,X_test2 =self.pcaIslem(x_train,x_test,5)            
                classifier2 =  SVC(kernel='rbf')
                classifier2.fit(X_train2,y_train)
                y_pred2=classifier2.predict(X_test2)
                self.Cmatrix2(y_test,y_pred2,"PCA")
                acc=accuracy_score(y_test, y_pred2)*100
                
                self.label_9.setText(str(round(acc,2)))
                self.label_8.setText(str(round(r2_score(y_test,y_pred2),2)))
                self.Cmatrix(y_test,y_pred,"SVC")
                print(confusion_matrix(y_test, y_pred))
                self.pltRoc2(y_test,y_pred,"SVC")
                
                from sklearn.model_selection import KFold
                from numpy import mean
                from sklearn.model_selection import cross_val_score  
                from sklearn.preprocessing import StandardScaler
                
                X = x.values
                sc_X=StandardScaler()
                X=sc_X.fit_transform(X) 
                y = y.values
                kf = KFold(n_splits=fold_size)
                kf.get_n_splits(X)
                
                sayma=0
                for train_index, test_index in kf.split(X):
                    sayma+=1
                    # print("TRAIN:", train_index, "TEST:", test_index)
                    x_train, x_test = X[train_index], X[test_index]
                    y_train, y_test = y[train_index], y[test_index]
                    NBG = SVC(kernel='rbf')
                    NBG.fit(x_train, y_train)
                    y_pred = NBG.predict(x_test)   
                    self.CmatrixFold(y_test,y_pred,"fold-"+str(sayma))
                    self.kfoldCmatrix(y_test, y_pred,"Fold Sonuç")
                    acc=accuracy_score(y_test, y_pred)*100
                    print(acc)

                                  
                model = SVC(kernel='rbf') 
                scores = cross_val_score(model, X, y, scoring='accuracy', cv=kf, n_jobs=-1)
           
                self.label_12.setText(str(round(mean(scores*100),2)))
                self.label_13.setText(str(round(scores.max()*100,2)))          
                print('Accuracy: %.3f (%.3f)' % (mean(scores), scores.max()))           
                print("--------------------")
                
                # hem grit hem random dogru
                # import pandas as pd
                # import numpy as np
                # from sklearn import preprocessing
                # from sklearn.model_selection import GridSearchCV, RandomizedSearchCV 
                # from sklearn.svm import SVC as svc 
                # from sklearn.metrics import make_scorer, roc_auc_score
                # from scipy import stats
                 
                # # DATA PREPARATION
           
                # # DEFINE MODEL AND PERFORMANCE MEASURE
                # mdl = svc(probability = True, random_state = 1)
                # auc = make_scorer(roc_auc_score)
                 
                # # GRID SEARCH FOR 20 COMBINATIONS OF PARAMETERS
                # grid_list = {"C": np.arange(2, 10, 2),
                #               "gamma": np.arange(0.1, 1, 0.2)}
                 
                # grid_search = GridSearchCV(mdl, param_grid = grid_list, n_jobs = 4, cv = 3, scoring = auc) 
                # grid_search.fit(x, y) 
                # print(grid_search.cv_results_)
                 
                # # RANDOM SEARCH FOR 20 COMBINATIONS OF PARAMETERS
                # rand_list = {"C": stats.uniform(2, 10),
                #               "gamma": stats.uniform(0.1, 1)}
                              
                # rand_search = RandomizedSearchCV(mdl, param_distributions = rand_list, n_iter = 20, n_jobs = 4, cv = 3, random_state = 2017, scoring = auc) 
                # rand_search.fit(x, y) 
                # print(rand_search.cv_results_)
           
               ###################################################
                
                # dogru sadece grid
                # from sklearn.datasets import make_blobs
                # from sklearn.model_selection import RepeatedStratifiedKFold
                # from sklearn.model_selection import GridSearchCV
                # from sklearn.svm import SVC
                # # define dataset
                # X, y = x,y
                # # define model and parameters
                # model = SVC() 
                # kernel = ['poly', 'rbf', 'sigmoid']
                # C = [50, 10, 1.0, 0.1, 0.01]
                # gamma = ['scale']
                # # define grid search
                # grid = dict(kernel=kernel,C=C,gamma=gamma)
                # cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
                # grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
                # grid_result = grid_search.fit(X, y)
                # # summarize results
                # print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
                # means = grid_result.cv_results_['mean_test_score']
                # stds = grid_result.cv_results_['std_test_score']
                # params = grid_result.cv_results_['params']
                # for mean1, stdev, param in zip(means, stds, params):
                #     print("%f (%f) with: %r" % (mean1, stdev, param))
                # print("bitti")
            
          
            
            
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = window()
    window.show()
    sys.exit(app.exec())
