
#!/usr/bin/python3

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics

from sklearn.metrics import classification_report
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Load the data from a CSV file
data =pd.read_csv('dataset_lorawan.csv')
#CYCLES,INSTR,LD_STALL,JMP_STALL,IMISS,LD,ST,JUMP,BRANCH,BRANCH_TAKEN,COMP_INSTR,PIP_STALL

X = data.drop(['Label'], axis=1) # drop the target column
y = data['Label']

random_seed = 100
# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed)

# Train a decision tree classifier
clf = DecisionTreeClassifier(max_depth=2, random_state=random_seed, criterion="entropy")
clf.fit(X_train, y_train)

# Use SelectFromModel to select the features with the highest importance scores
sfm = SelectFromModel(clf, threshold=0.01)
sfm.fit(X_train, y_train)
X_train_selected = sfm.transform(X_train)
X_test_selected =sfm.transform(X_test)
# Print the number of features before and after feature selection
print("Number of features before feature selection:", X_train.shape[1])
print("Number of features after feature selection:", X_train_selected.shape[1])

# Train the decision tree model on the selected features
# Encode class labels
targets = np.array(data.Label)
le = LabelEncoder()
targets = le.fit_transform(targets)
class_names = le.classes_.tolist()  # Convert numpy array to a list

clf = DecisionTreeClassifier(max_depth=2, random_state=random_seed,max_features=2, criterion="entropy")
clf.fit(X_train_selected, y_train)
print(X_train_selected[0])
score = clf.score(X_test_selected, y_test)
pred=clf.predict(X_test_selected)
confusion=metrics.confusion_matrix(y_test,pred)
TP=confusion[1,1]
TN=confusion[0,0]
FP=confusion[0,1]
FN=confusion[1,0]
precision=(TP / (TP + FP)) # Precision=TP/(TP+FP).
recall=(TP/(TP+FN))  # Recall=TP/(TP+FN) 
F1_Score= (2*(precision*recall)/(precision+recall)) # F1−Score=2∗(Precision∗recall)/(Precision+recall
accura=metrics.accuracy_score(y_test, pred)
name='decision_tree'
classifier=""+name+",accuracy=%.3f"%accura+",precision=%.3f"%precision+",recall=%.3f"%recall+",f1score=%.3f"%F1_Score+"\n"
print(classifier)
# Convert the 'Index' object to a list
feature_names = X_train.columns[sfm.get_support()].tolist()

plt.figure(figsize=(10, 10))
plot_tree(clf, feature_names=feature_names, filled=True, 
          class_names=class_names, impurity=False, label='all')
plt.savefig("decision_tree_LORAWAN.jpeg")
