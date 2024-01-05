# Code source: Gaël Varoquaux
#              Andreas Müller
# Modified for documentation by Jaques Grobler
# License: BSD 3 clause
 
#from matplotlib.widgets import Cursor
import random
import numpy as np
import pandas as pd
from matplotlib import pyplot
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn import metrics
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report

# Your existing code here


h = 25  # step size in the mesh
# add metrics : time to predict
names = [
"Nearest Neighbors", #KNN
"Linear SVM",  #
"RBF SVM", ### SVM
"Gaussian Process",###
"Decision Tree",  # DT
"Random Forest", # RF
"Neural Net", ###
"AdaBoost",
"Naive Bayes", ###
"QDA", ###
"SGDClassifier"
]
# names = [

# "QDA" ###

# ]

# classifiers = [
# KNeighborsClassifier(n_neighbors=5),
# SVC(kernel="linear", C=0.025),
# SVC(gamma=2, C=1),
# GaussianProcessClassifier(),
# DecisionTreeClassifier(),
# RandomForestClassifier(),
# MLPClassifier(random_state=2, max_iter=50),
# AdaBoostClassifier(),
# GaussianNB(),
# QuadraticDiscriminantAnalysis(),
# SGDClassifier(loss="hinge", penalty="l2", max_iter=100),
# ]


# classifiers = [
#     KNeighborsClassifier(n_neighbors=5),
#     SVC(kernel="linear", C=0.025),
#     SVC(kernel='rbf', gamma=2, C=0.025),
#     GaussianProcessClassifier(),
#     DecisionTreeClassifier(),
#     RandomForestClassifier(),
#     MLPClassifier(random_state=2, max_iter=50),
#     AdaBoostClassifier(),
#     GaussianNB(),
#     QuadraticDiscriminantAnalysis(),
#     SGDClassifier(loss="hinge", penalty="l2", max_iter=50)
# ]


classifiers = [
    KNeighborsClassifier(n_neighbors=5),
    SVC(kernel="linear"),
    SVC(kernel='rbf'),
    GaussianProcessClassifier(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    MLPClassifier(),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),
    SGDClassifier()
]

# classifiers = [

# QuadraticDiscriminantAnalysis()

# ]
pathtest = './test_file.csv'
path='./output_file.csv'
#path='/home/med/Desktop/workflow/handsonML/archive_code/Packet_Dataset.csv'
# path1='/home/med/Desktop/workflow/handsonML/test.csv'
#path='/home/med/Desktop/workflow/handsonML/archive_code/dataset.csv'
#JMP_STALL : Number of jump register hazards
#IMISS Cycles waiting for instruction fethces, excluding jumps and branches
#LD : Number of load instructions
#ST : Number of store instructions
#JUMP : Number of jumps(unconditional)
#BRANCH : Number of branches(conditional)
#Branch_TAKEN : Number of branches taken (conditional)
#COMP_INSTR : Number of compressed instructions retired
#PIP_STALL : Cycles from stalled pipeline 
# Cycles,Minstret,LD_STALL,JMP_STALL,IMISS,LD,ST,JUMP,BRANCH,BRANCH_TAKEN,COMP_INSTR,PIP_STALL,Label
name = ['Packet_Nbre', 'Cycles', 'Minstret', 'JMP_STALL', 'IMISS', 'LD', 'ST', 'JUMP', 'BRANCH', 'BRANCH_TAKEN', 'COMP_INSTR', 'PIP_STALL']
features = ['Cycles', 'Minstret', 'JMP_STALL', 'IMISS', 'LD', 'ST', 'JUMP', 'BRANCH', 'BRANCH_TAKEN', 'COMP_INSTR', 'PIP_STALL']

# name=['Label','Cycles','Minstrert','JMP_STALL','IMISS','LD','ST','JUMP','BRANCH','BRANCH_TAKEN','COMP_INSTR','PIP_STALL']
# features=['Cycles','Minstret','JMP_STALL','IMISS','LD','ST','JUMP','BRANCH','BRANCH_TAKEN','COMP_INSTR','PIP_STALL']
#df=pd.read_csv(path)
df=pd.read_csv(path)
test=pd.read_csv(pathtest)
#df=df.drop(['Cycle'],1)
#X=np.array(df.drop(['Packet_Nbre'],1).astype(np.int32))
#X=df.drop(['Packet_Nbre'],axis=1)
X=df.drop(['Packet_Nbre','JMP_STALL','PIP_STALL'],axis=1)
y=df.Packet_Nbre

X_test=test.drop(['Packet_Nbre','JMP_STALL','PIP_STALL'],axis=1)
y_test=test.Packet_Nbre


print(df.info())
print(df.head())
print(df.Packet_Nbre.value_counts())

#y=np.array(df['Packet_Nbre'])

X_train, X_test1, y_train, y_test2 = train_test_split(    X, y,test_size=0.05)
#X_train, y_train = train_test_split(X, y,test_size=0.35)

classhead="Classifier,Accuracy,Precision,Recall,F1-Score\n"
csv_file1 = open("classifier_comparision.csv", "w")
csv_file1.write(classhead)
accuracy=[]
accuracytr=[]
for name, clf in zip(names, classifiers):
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    pred=clf.predict(X_test)

    #print(classification_report(y_test, pred)) 

    confusion=metrics.confusion_matrix(y_test,pred)

    TP=confusion[1,1]
    TN=confusion[0,0]
    FP=confusion[0,1]
    FN=confusion[1,0]
    precision=(TP / (TP + FP)) # Precision=TP/(TP+FP).
    recall=(TP/(TP+FN))  # Recall=TP/(TP+FN) 
    F1_Score= (2*(precision*recall)/(precision+recall)) # F1−Score=2∗(Precision∗recall)/(Precision+recall)
    accuracy.append(metrics.accuracy_score(y_test, pred))
    accura=metrics.accuracy_score(y_test, pred)
    classifier=""+name+",%.3f"%accura+",%.3f"%precision+",%.3f"%recall+",%.3f"%F1_Score+"\n"
    print("classifier=",name,"accuracy =%.3f"%metrics.accuracy_score(y_test, pred),"precision=%.3f "%precision,"recall =%.3f "%recall,"F1_score=%.3f "%F1_Score)
    open("classifier_comparision.csv","a")
    csv_file1.write(classifier)

    #classifier.to_csv('classifier_comparision.csv',sep=',',header=False, index=False,encoding='utf-8',mode="a")
    #print("TP=",TP,"TN",TN,"FP",FP,"FN",FN)
    #print("\n")
print(accuracy)
print(accuracytr)

# ax=df.drop(['Packet_Nbre','Cycles'],axis=1).plot()
# #cursor =Cursor(ax, horizOn = True, vertOn=True, color='red', linewidth=1,   useblit=True)
# plt.xticks(fontsize=17.0,fontweight='bold')
# plt.yticks(fontsize=17.0,fontweight='bold')
# plt.title('HPC value per Packet Network', fontsize=24.0 ,fontweight='bold')
# plt.xlabel('Number of Packet Network', fontsize=24.0 ,fontweight='bold')
# plt.ylabel('HPC-Hardware Performance Counter value', fontsize=24.0 ,fontweight='bold')
# plt.legend(fontsize=17.0)
# plt.show()
