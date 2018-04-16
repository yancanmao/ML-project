import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn import metrics
import pandas as pd
import matplotlib.pyplot as plt



# Create Pandas dataframe.
columns = ["age", "sex", "cp", "restbp", "chol", "fbs", "restecg", 
           "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"]
df0     = pd.read_table("data/heart_disease_all14.csv", sep=',', header=None, names=columns)
df      = df0.copy()
dummies = pd.get_dummies(df["cp"],prefix="cp")
df      = df.join(dummies)
del df["cp"]
del df["cp_4.0"]
df      = df.rename(columns = {"cp_1.0":"cp_1","cp_2.0":"cp_2","cp_3.0":"cp_3"})

dummies = pd.get_dummies(df["restecg"],prefix="recg")
df      = df.join(dummies)
del df["restecg"]
del df["recg_0.0"]
df      = df.rename(columns = {"recg_1.0":"recg_1","recg_2.0":"recg_2"})

dummies = pd.get_dummies(df["slope"],prefix="slope")
df      = df.join(dummies)
del df["slope"]
del df["slope_2.0"]
df      = df.rename(columns = {"slope_1.0":"slope_1","slope_3.0":"slope_3"})

dummies = pd.get_dummies(df["thal"],prefix="thal")
df      = df.join(dummies)
del df["thal"]
del df["thal_3.0"]
df      = df.rename(columns = {"thal_6.0":"thal_6","thal_7.0":"thal_7"})

# Replace response variable values and rename
df["num"].replace(to_replace=[1,2,3,4],value=1,inplace=True)
df      = df.rename(columns = {"num":"hd"})

# New list of column labels after the above operations
new_columns_1 = ["age", "sex", "restbp", "chol", "fbs", "thalach", 
                 "exang", "oldpeak", "ca", "hd", "cp_1", "cp_2",
                 "cp_3", "recg_1", "recg_2", "slope_1", "slope_3",
                 "thal_6", "thal_7"]

# Standardize the dataframe
stdcols = ["age","restbp","chol","thalach","oldpeak"]
nrmcols = ["ca"]
stddf   = df.copy()
stddf[stdcols] = stddf[stdcols].apply(lambda x: (x-x.mean())/x.std())
stddf[nrmcols] = stddf[nrmcols].apply(lambda x: (x-x.mean())/(x.max()-x.min()))

new_columns_2 = new_columns_1[:9] + new_columns_1[10:]
new_columns_2.insert(0,new_columns_1[9])
stddf = stddf.reindex(columns=new_columns_2)

# Convert dataframe into lists for use by classifiers
yall = stddf["hd"]
Xall = stddf[new_columns_2[1:]].values


'''
Print out some features of the optimal logistic regression model
for the full data set (accuracy, precision, recall,...)
'''

model = LogisticRegression(fit_intercept=False,penalty="l1",dual=False,C=1000.0)
comb = [1,2,5,8,9,10,11,14,17]
print (comb)
Xsel  = []
for patient in Xall:
    Xsel.append([patient[ind] for ind in comb])
print(model)
lrfit = model.fit(Xsel,yall)
print('\nLogisticRegression score on full data set: %f\n' % lrfit.score(Xsel,yall))
ypred = model.predict(Xsel)
print ('\nClassification report on full data set:')
print(metrics.classification_report(yall,ypred))
print ('\nConfusion matrix:')
print(metrics.confusion_matrix(yall,ypred))
log_reg_coef = model.coef_[0]
log_reg_intc = model.intercept_
print('\nLogisticRegression coefficients: %s' %log_reg_coef)
print('LogisticRegression intercept: %f' %log_reg_intc)

# Separate disease from no-disease cases for a histogram of logistic regression probabilities:
Xsel0 = [patient for (patient,status) in zip(Xsel,yall) if status==0] # No-disease cases
Xsel1 = [patient for (patient,status) in zip(Xsel,yall) if status==1] # Disease cases
print('\nNumber of disease cases: %i, no-disease cases: %i' %(len(Xsel1),len(Xsel0)))
Xsel0_Prob = [p1 for (p0,p1) in model.predict_proba(Xsel0)] # Predicted prob. of heart disease for no-disease cases
Xsel1_Prob = [p1 for (p0,p1) in model.predict_proba(Xsel1)] # Predicted prob. of heart disease for disease cases

# Here we do a little test to make sure we understand how the logistic regression coefficients work:
qsum = 0.0
for patient,prob in zip(Xsel0,Xsel0_Prob):
    lrprob = 1.0/(1.0+np.exp(-sum(patient[i]*log_reg_coef[i] for i in range(9))-log_reg_intc))
    qsum  += (lrprob-prob)**2
for patient,prob in zip(Xsel1,Xsel1_Prob):
    lrprob = 1.0/(1.0+np.exp(-sum(patient[i]*log_reg_coef[i] for i in range(9))-log_reg_intc))
    qsum  += (lrprob-prob)**2
print('Sum of quadratic differences between probability calculations: %f' %qsum) # Should be zero!

# Here is the plot:
fig, axes = plt.subplots( nrows=1, ncols=1, figsize=(6,6) )
plt.subplots_adjust( top=0.92 )
plt.suptitle("Cleveland Data Set", fontsize=20)
axes.hist(Xsel0_Prob,color=["chartreuse"],histtype="step",label="no-disease cases (160)")
axes.hist(Xsel1_Prob,color=["crimson"],histtype="step",label="disease cases (139)")
axes.set_xlabel("Predicted Probability of Disease",fontsize=15)
axes.set_ylabel("Number of Patients",fontsize=15)
axes.set_ylim( 0.0, 90.0 )
axes.legend(prop={'size': 15},loc="upper right")
plt.show()