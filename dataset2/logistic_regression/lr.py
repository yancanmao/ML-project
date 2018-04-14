import numpy
import csv
import math
import operator
import random 
from sklearn.linear_model import LogisticRegression


def loadDataset(split,X,Y, X_train=[] , Y_train=[],  X_test=[],  Y_test=[]):
    c=0
    for i in range(0,X.shape[0]):
        if random.random() < split:
            X_train.append(X[i])
            Y_train.append(Y[i])
            c=c+1
        else:
            X_test.append(X[i])
            Y_test.append(Y[i])
    return c

#create reduced feature matrix
reader=csv.reader(open("reduced_features.csv","r"),delimiter=",")
X=list(reader)
X=numpy.array(X)
X=X.astype(numpy.float)


#create result vector
reader=csv.reader(open("target_output.csv","r"),delimiter=",")
Y=list(reader)
Y=numpy.array(Y)
Y=Y.astype(numpy.int)
Y=Y.ravel()	

X_train=[]
Y_train=[]
X_test=[]
Y_test=[]


c=loadDataset(0.7,X,Y, X_train , Y_train,  X_test,  Y_test)
 
clf = LogisticRegression(max_iter=100,C=1)
clf.fit(X_train,Y_train)

count = 100;
score_test = 0;
score_train = 0;
while count > 0:
    clf.fit(X_train, Y_train)
    clf.predict(X_train)
    score_train += clf.score(X_train, Y_train)
    clf.predict(X_test)
    score_test += clf.score(X_test, Y_test)
    print ("round ",count)
    count -= 1;
    pass 
print("..................................Traing set................................")
print ("Training Set accuracy = ", score_train)
print ("Training Set error = ", 100-score_train)

print("..................................Test set...................................")
print ("Test Set accuracy = ", score_test)
print ("Test Set error = ",(100-score_test))