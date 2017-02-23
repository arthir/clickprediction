import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pylab as pl

import sys
import math
import numpy as np
import sklearn as sklearn
from sklearn import *
from sklearn.multiclass import OneVsRestClassifier

data_file = "urls_newsmedia_dataset.domain4.txt"
user_file = "/home/mgabielk/twitter_bitly_crawler/users_data.txt"
user_url_file = "/home/mgabielk/twitter_bitly_crawler/users_urls.col"

dataset = np.loadtxt(data_file, delimiter="\t", skiprows=1)

file_prefix = sys.argv[1]
predict_ctr_or_clicks = sys.argv[2]
log_of_features = sys.argv[3]

features = dict()
features['1hr of shares'] = [13]
features['4hr of shares'] = [13,14,15,16]
features['24hr of shares'] = [13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36]
features['1hr of impressions'] = [45]
features['4hr of impressions'] = [45,46,47,48]
features['24hr of impressions'] = [45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68]
features['1hr of shares+impressions'] = [13,45]
features['4hr of shares+impressions'] = [13,14,15,16,45,46,47,48]
features['24hr of shares+impressions'] = [13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68]
features['1hr of clicks'] = [38] # TODO: this one's a little weird since we're calculating based on this
features['1hr of clicks+impressions'] = [38,45]
features['4hr of clicks'] = [38, 39, 40, 41]
features['4hr of clicks+impressions'] = [38, 39, 40, 41,45,46,47,48]

def linear_regression(cols, feature_name, fout2):
    X = dataset[:,cols]
    if log_of_features == 'True':
        X_scaled = np.log(preprocessing.scale(X)+1)
    else:
        X_scaled = preprocessing.scale(X)
    if predict_ctr_or_clicks == 'clicks':
        if "LogClicks" in file_prefix:
            y = np.log(dataset[:,44]+1)
        else:
            y = dataset[:,44]
    else:
        y = dataset[:,44]/dataset[:,9]
    y[np.where(np.isnan(y))] = 0
    x_train, x_test, y_train, y_test = sklearn.cross_validation.train_test_split(X_scaled, y, test_size=0.33, random_state=42)
    regr = linear_model.LinearRegression()
    #regr = linear_model.SGDRegressor()
    #regr = svm.SVR()
    regr.fit(x_train, y_train)
    #print('Coefficients: ', regr.coef_)
    #print("Residual sum of squares: %f" % np.mean((regr.predict(x_test) - y_test) ** 2))
    # Explained variance score: 1 is perfect prediction 
    #print('Variance score: %f' % regr.score(x_test, y_test))
    y_pred = regr.predict(x_test)
    for i, pred in enumerate(y_pred):
        fout2.write(feature_name + "\t" + str(float(y_test[i])) + "\t" + str(y_pred[i]) + "\n")
    fout.write(feature_name +"\t"+ str(np.mean((y_pred - y_test) ** 2)) +"\t"+ str(regr.score(x_test, y_test)) + "\n")

fout = open("linearRegression_"+file_prefix+"_metrics_domain4.txt", "w+")
fout.write("features\tRSS\tVarianceScore\n")
fout2 = open("linearRegression_"+file_prefix+"_results_domain4.txt", "w+")
fout2.write("features\treal_value\tpredicted_value\n")
for feature_name, feature_cols in features.items():
    #print "Features: ", feature_name
    linear_regression(features[feature_name], feature_name, fout2)
