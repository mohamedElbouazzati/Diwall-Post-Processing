# Code source: Gaël Varoquaux
#              Andreas Müller
# Modified for documentation by Jaques Grobler
# License: BSD 3 clause
 
#from matplotlib.widgets import Cursor
import random
from nbformat import write
import numpy as np
import pandas as pd
from matplotlib import pyplot
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn import tree

from sklearn.metrics import classification_report
h = 25  # step size in the mesh
from sklearn import preprocessing

def encode_feature(array):
    """ Encode a categorical array into a number array
    
    :param array: array to be encoded
    :return: numerical array
    """
  
    encoder = preprocessing.LabelEncoder()
    encoder.fit(array)
    return encoder.transform(array)

def print_feature_importance(names_array, importances_array):
    """ Prints out a feature importance array as a dataframe. """
    importances = pd.DataFrame(data=names_array)
    importances[1] = importances_array
    importances = importances.T
    importances.drop(0, axis=0, inplace=True)
    importances.columns = feature_names
    
    print(str(importances.reset_index(drop=True)))
    
def build_tree(features, targets, feature_names, class_names):

  
    train_features, test_features, train_targets, test_targets = train_test_split(features, targets, test_size=0.2, random_state=123)

    decision_tree = tree.DecisionTreeClassifier(random_state=456, criterion="entropy",max_depth=3)
    decision_tree = decision_tree.fit(train_features, train_targets)
    # Visualizing the decision tree
    
    # 1. Saving the image of the decision as a png   
    plt.subplots(figsize=(17, 12))
    tree.plot_tree(decision_tree, feature_names=feature_names, filled=True, rounded=True, class_names=class_names)
    plt.savefig("decision_tree.png")
    # 2. Output the tree as text in the console
    tree_as_text = tree.export_text(decision_tree, feature_names=feature_names)
    print(tree_as_text)
    # Feature Importance
    # Turns the feature importance array into a dataframe, so it has a table-like output format
    print_feature_importance(feature_names, decision_tree.feature_importances_)
    # Training and test mean accuracy
    train_error = np.round(decision_tree.score(train_features, train_targets), 2)
    test_error = np.round(decision_tree.score(test_features, test_targets), 2)
    
    print("Training Set Mean Accuracy = " + str(train_error))
    print("Test Set Mean Accuracy = " + str(test_error))
# add metrics : time to predict
names = [
   
    "Decision Tree",  # DT

]

classifiers = [
  
    DecisionTreeClassifier()

]
path='/home/med/Desktop/workflow/handsonML/label/1000_same_size_35_42/Packet_Dataset.csv'

                    # JMP_STALL : Number of jump register hazards
                    # IMISS Cycles waiting for instruction fethces, excluding jumps and branches
                    # LD : Number of load instructions
                    # ST : Number of store instructions
                    # JUMP : Number of jumps(unconditional)
                    # BRANCH : Number of branches(conditional)
                    # Branch_TAKEN : Number of branches taken (conditional)
                    # COMP_INSTR : Number of compressed instructions retired
                    # PIP_STALL : Cycles from stalled pipeline 
class_names = ['legitime', 'stack_overflow','heap_overflow']
feature_names = ['Cycles','Minstrert','IMISS','LD','ST','JUMP','BRANCH','BRANCH_TAKEN','COMP_INSTR','PIP_STALL']
name=['Packet_Nbre','Cycles','Minstrert','JMP_STALL','IMISS','LD','ST','JUMP','BRANCH','BRANCH_TAKEN','COMP_INSTR','PIP_STALL']

df=pd.read_csv(path)
X=df.drop(['Packet_Nbre','Cycles','JMP_STALL'],axis=1)
features=np.array(X)
features[:, 1] = encode_feature(features[:, 1])
print(df.info())
print(df.head())
print(df.Packet_Nbre.value_counts())
y=df.Packet_Nbre
targets=np.array(df.Packet_Nbre)
targets = encode_feature(targets)
X_train, X_test, y_train, y_test = train_test_split(
        X, y,test_size=0.4)

for name, clf in zip(names, classifiers):
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        pred=clf.predict(X_test)
        confusion=metrics.confusion_matrix(y_test,pred)
        TP=confusion[1,1]
        TN=confusion[0,0]
        FP=confusion[0,1]
        FN=confusion[1,0]
        precision=(TP / (TP + FP))*99.99 # Precision=TP/(TP+FP).
        recall=(TP/(TP+FN))*99.99  # Recall=TP/(TP+FN) 
        F1_Score= (2*(precision*recall)/(precision+recall)) # F1−Score=2∗(Precision∗recall)/(Precision+recall
        accura=metrics.accuracy_score(y_test, pred)*99.99
        classifier=""+name+",%.3f"%accura+",%.3f"%precision+",%.3f"%recall+",%.3f"%F1_Score+"\n"
        print("classifier=",name,"accuracy =%.3f"%metrics.accuracy_score(y_test, pred),"precision=%.3f "%precision,"recall =%.3f "%recall,"F1_score=%.3f "%F1_Score)        
        tree.plot_tree(clf)
        plt.show()
build_tree(features, targets, feature_names, class_names)

