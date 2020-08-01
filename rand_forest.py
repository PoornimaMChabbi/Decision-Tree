import pandas as pd
import numpy as np
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
training_dataset=sys.argv[1]
testing_dataset=sys.argv[3]

train = pd.read_csv(training_dataset,header=None)
test = pd.read_csv(testing_dataset,header=None)

myList1=[]
for i in range(len(train.columns)):
    myList1.append('p'+str(i))
train=train.rename(columns=dict(zip(train.columns,myList1)))
    
myList1=[]
for i in range(len(test.columns)):
    myList1.append('p'+str(i))
test=test.rename(columns=dict(zip(test.columns,myList1)))

train_target = train.iloc[:, -1]
train_data= train.iloc[:,:-1]


test_data = test.iloc[:,:-1]
test_target = test.iloc[:, -1]

print(test_data)
print(test_target)

classifier = RandomForestClassifier()
classifier.fit(train_data,train_target)

prediction=classifier.predict(test_data)

print(np.asarray(list(prediction)))
#making a confusion matrix
cm=confusion_matrix(np.asarray(test_target),prediction)
#
print('Accuracy Score :',accuracy_score(np.asarray(test_target), prediction))
print(classification_report(np.asarray(test_target), prediction))
print(cm)