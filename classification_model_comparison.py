# Code source: Gaël Varoquaux
#              Andreas Müller
# Modified for documentation by Jaques Grobler
# License: BSD 3 clause

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
h = 25  # step size in the mesh

names = [
    "Nearest Neighbors",
    "Linear SVM",
    "RBF SVM",
    "Gaussian Process",
    "Decision Tree",
    "Random Forest",
    "Neural Net",
    "AdaBoost",
    "Naive Bayes",
    "QDA",
]

classifiers = [
    KNeighborsClassifier(5),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    MLPClassifier(),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),
]
path='/home/med/Desktop/workflow/handsonML/Packet_Dataset.csv'

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

name=['Cycles','Minstrert','JMP_STALL','IMISS','LD','ST','JUMP','BRANCH','BRANCH_TAKEN','COMP_INSTR','PIP_STALL']
df=pd.read_csv(path)

df=df.drop(['Cycle'],1)
X=np.array(df.drop(['Packet_Nbre'],1).astype(np.int32))

y=np.array(df['Packet_Nbre'])


X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42)

for name, clf in zip(names, classifiers):
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        pred=clf.predict(X_test)
        confusion=metrics.confusion_matrix(y_test,pred)
        TP=confusion[1,1]
        TN=confusion[0,0]
        FP=confusion[0,1]
        FN=confusion[1,0]
        precision=TP / (TP + FP)
        print("classifier=",name,"accuracy",score*100,"%","equivalent=",metrics.accuracy_score(y_test, pred),"precision=",precision)
        print("TP=",TP,"TN",TN,"FP",FP,"FN",FN)
        print("\n")

df.plot()
plt.show()
