import pandas as pd
from sklearn.externals import joblib


def logistic_regression(input):
    data = data_cleaning(input)
    # print(stddf)
    # model = joblib.load("../dataset1/log_reg.pkl")
    model = joblib.load("./log_reg.pkl")
    result = model.predict(data)
    print(result)
    # new_columns_2 = new_columns_1[:9] + new_columns_1[10:]
    # new_columns_2.insert(0, new_columns_1[9])
    # stddf = stddf.reindex(columns=new_columns_2)
    # Xall = stddf[new_columns_2].values
    if result[0] == 1:
        return True
    else:
        return False


def data_cleaning(input, comb=None):
    columns = ["age", "sex", "cp", "restbp", "chol", "fbs", "restecg",
               "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
    y = pd.DataFrame(columns=columns)
    y.loc[1] = input
    y["cp_1"] = 0.0
    y["cp_2"] = 0.0
    y["cp_3"] = 0.0
    y["recg_1"] = 0.0
    y["recg_2"] = 0.0
    y["slope_1"] = 0.0
    y["slope_3"] = 0.0
    y["thal_6"] = 0.0
    y["thal_7"] = 0.0
    cp_value = y.loc[1]["cp"]
    recg_value = y.loc[1]["restecg"]
    slope_value = y.loc[1]["slope"]
    thal_value = y.loc[1]["thal"]
    # dummy process
    del y["cp"]
    del y["restecg"]
    del y["slope"]
    del y["thal"]
    # dummy cp
    if cp_value == 1.0:
        y["cp_1"] = 1.0
    elif cp_value == 2.0:
        y["cp_2"] = 1.0
    elif cp_value == 3.0:
        y["cp_3"] = 1.0
    # dummy recg
    if recg_value == 1.0:
        y["recg_1"] = 1.0
    elif recg_value == 2.0:
        y["recg_2"] = 1.0
    # dummy slope
    if slope_value == 1.0:
        y["slope_1"] = 1.0
    elif slope_value == 3.0:
        y["slope_3"] = 1.0
    # dummy thal
    if thal_value == 6.0:
        y["thal_6"] = 1.0
    elif thal_value == 7.0:
        y["thal_7"] = 1.0
    print(y)
    # need to dummy the data
    new_columns_1 = ["age", "sex", "restbp", "chol", "fbs", "thalach",
                     "exang", "oldpeak", "ca", "hd", "cp_1", "cp_2",
                     "cp_3", "recg_1", "recg_2", "slope_1", "slope_3",
                     "thal_6", "thal_7"]
    # need normalize
    stdcols = ["age", "restbp", "chol", "thalach", "oldpeak"]
    nrmcols = ["ca"]
    stddf = y.copy()
    stddf["age"] = stddf["age"].apply(lambda x: (x - 54.521739) / 9.030264)
    stddf["restbp"] = stddf["restbp"].apply(lambda x: (x - 131.715719) / 17.747751)
    stddf["chol"] = stddf["chol"].apply(lambda x: (x - 246.785953) / 52.532582)
    stddf["thalach"] = stddf["thalach"].apply(lambda x: (x - 149.327759) / 23.121062)
    stddf["oldpeak"] = stddf["oldpeak"].apply(lambda x: (x - 1.058528) / 1.162769)
    stddf[nrmcols] = stddf[nrmcols].apply(lambda x: (x - 0.672241) / 3.0)
    if comb is None:
        return stddf.values
    # comb = [0, 1, 3, 5, 6, 7, 8, 9, 11, 14, 17]
    X = []
    for i in comb:
        X.append(stddf.values[0][i])
    # print(type(stddf.values))
    # print(stddf.values)
    print(X)
    print("-----------------------------------------")
    return [X]


def naive_bayes(input):
    comb = [0, 1, 3, 5, 6, 7, 8, 9, 11, 14, 17]
    data = data_cleaning(input, comb)
    # print(stddf)
    # model = joblib.load("../dataset1/log_reg.pkl")
    model = joblib.load("./gnb.pkl")
    result = model.predict(data)
    print(result)
    # new_columns_2 = new_columns_1[:9] + new_columns_1[10:]
    # new_columns_2.insert(0, new_columns_1[9])
    # stddf = stddf.reindex(columns=new_columns_2)
    # Xall = stddf[new_columns_2].values
    if result[0] == 1:
        return True
    else:
        return False


def SVM(input):
    comb = [0, 1, 4, 5, 6, 7, 8, 10, 14, 16, 17]
    data = data_cleaning(input, comb)
    # print(stddf)
    # model = joblib.load("../dataset1/log_reg.pkl")
    model = joblib.load("./svm.pkl")
    result = model.predict(data)
    print(result)
    # new_columns_2 = new_columns_1[:9] + new_columns_1[10:]
    # new_columns_2.insert(0, new_columns_1[9])
    # stddf = stddf.reindex(columns=new_columns_2)
    # Xall = stddf[new_columns_2].values
    if result[0] == 1:
        return True
    else:
        return False


def boost_gradient(input):

    return True
