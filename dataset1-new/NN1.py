import pandas
import numpy
import matplotlib.pyplot as plt
df=pandas.read_csv('Preprocessed/data_combined.csv')
# print (df[:5])
df['THRESTBPS'] = df['THRESTBPS'].replace(['?'],'120')
df['THRESTBPS'] = df['THRESTBPS'].astype('int64')
v=df.FBS.values=='?'
#randomly filling values with 80% with 0 and 20% with 1s
df.loc[v, 'FBS'] = numpy.random.choice(('0','1'), v.sum(), p=(0.8,0.2))
# print (df['FBS'].value_counts())
df['FBS']=df['FBS'].astype('int64')
df['CHOL'].value_counts().head()
df['CHOL']=df['CHOL'].replace('?','-69')#temporarily replacing ? with -69
df['CHOL']=df['CHOL'].astype('int64')
k=int(df[df['CHOL']!=-69]['CHOL'].mean())
df['CHOL']=df['CHOL'].replace(-69,k)
df['THALACH'].value_counts().head()
df['THALACH']=df['THALACH'].replace('?','-69')#temporarily replacing ? with -69
df['THALACH']=df['THALACH'].astype('int64')
k=int(df[df['THALACH']!=-69]['THALACH'].mean())
df['THALACH']=df['THALACH'].replace(-69,k)
v=df.EXANG.values=='?'
df.loc[v,'EXANG'] = numpy.random.choice(('0','1'), v.sum(), p=(0.61,0.39))
df['EXANG']=df["EXANG"].astype('int64')
df['OLDPEAK']=df['OLDPEAK'].replace('?','-69')#temporarily replacing ? with -69
df['OLDPEAK']=df['OLDPEAK'].astype('float64')
k=df[df['OLDPEAK']!=-69]['OLDPEAK'].mean()
df['OLDPEAK']=df['OLDPEAK'].replace(-69,numpy.round(k,1))
v=df.SLOPE.values=='?'
df.loc[v,'SLOPE'] = numpy.random.choice(('2','1','3'), v.sum(), p=(0.6,0.30,0.10))
# print df['SLOPE'].value_counts()
df['SLOPE']=df['SLOPE'].astype('int64')
v=df.CA.values=='?'
df.loc[v,'CA'] = numpy.random.choice(('0','1','2','3'), v.sum(), p=(0.60,0.20,0.13,0.07))
df['CA']=df['CA'].astype('int64')
# print df['CA'].value_counts()
df['THAL']=df['THAL'].replace('?',-1)
'''
df['THAL']=df['THAL'].replace('?',-1)
for row in df.iterrows():
    if row['THAL']==-1 and row['CATEGORY']>=1:
        df.loc[row.Index, 'ifor'] = 7
        
    elif row['THAL']==-1 and row['CATEGORY']==0:
        df.loc[row.Index, 'ifor'] = 3
'''
df.loc[(df['THAL']==-1)&(df['CATEGORY']!=0),'THAL']='7'
#print df['THAL'].value_counts()
df.loc[(df['THAL']==-1)&(df['CATEGORY']==0),'THAL']='3'
# print df['THAL'].value_counts()
df['THAL']=df['THAL'].astype('int64')
dummies = pandas.get_dummies(df["CP"],prefix="CP")
df = df.join(dummies)

dummies = pandas.get_dummies(df["RESTECG"],prefix="RESTECG")
df      = df.join(dummies)

dummies = pandas.get_dummies(df["SLOPE"],prefix="SLOPE")
df      = df.join(dummies)

dummies = pandas.get_dummies(df["THAL"],prefix="THAL")
df      = df.join(dummies)

#dummies = pandas.get_dummies(df["EXANG"],prefix="EXANG")
#df = df.join(dummies)

#del df['SEX']
del df['CP']
del df['RESTECG']
del df['SLOPE']
del df['THAL']
#del df['EXANG']
print (df.dtypes)

for g in df.columns:
    if df[g].dtype=='uint8':
        df[g]=df[g].astype('int64')

df.dtypes
df.loc[df['CATEGORY']>0,'CATEGORY']=1
stdcols = ["AGE","THRESTBPS","CHOL","THALACH","OLDPEAK"]
nrmcols = ["CA"]
stddf   = df.copy()
stddf[stdcols] = stddf[stdcols].apply(lambda x: (x-x.mean())/x.std())
stddf[nrmcols] = stddf[nrmcols].apply(lambda x: (x-x.min())/(x.max()-x.min()))
#stddf[stdcols] = stddf[stdcols].apply(lambda x: (x-x.mean())/(x.max()-x.min()))


for g in stdcols:
    print (g,max(stddf[g]),min(stddf[g]))
    
for g in nrmcols:
    print (g,max(stddf[g]),min(stddf[g]))

df_copy=stddf.copy()
df_copy=df_copy.drop(['CATEGORY'],axis=1)

dat=df_copy.values
labels=df['CATEGORY'].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(dat,labels, test_size=0.25, random_state=42)

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel

clf = ExtraTreesClassifier()
clf = clf.fit(dat,labels)
g=clf.feature_importances_
c=stddf.drop(['CATEGORY'],axis=1).columns

g=g*100

import keras 
import tensorflow
from sklearn import metrics
import matplotlib.pyplot as plt # side-stepping mpl backend
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers.core import Dropout, Flatten, Activation, Dense
import math

def make_model(activ,opti,ip,layers,trainx,trainy,testx,testy):
    model = Sequential()
    model.add(Dense(layers[0], input_dim=ip, init='uniform', activation=activ))
    model.add(Dense(layers[1], init='uniform', activation=activ))
    model.add(Dense(1, init='uniform', activation=activ))
    model.compile(loss='mean_squared_logarithmic_error', optimizer=opti, metrics=['accuracy'])
    model.fit(trainx,trainy,epochs=600,batch_size=512,verbose=2,validation_data=(testx,testy))
    
    trainScore = model.evaluate(trainx,trainy, verbose=0)
    print ("Train Score: ",100-trainScore[0]*100)
    testScore = model.evaluate(testx,testy, verbose=0)
    print ("Test Score: ",100-testScore[0]*100)
    
    return model

#without k best features,sigmoid and rmsprop
m1=make_model('sigmoid','rmsprop',x_train.shape[1],[x_train.shape[1],16],x_train,y_train,x_test,y_test)