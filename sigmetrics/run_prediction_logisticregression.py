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

def getClass(i):
    if predict_ctr_or_clicks == "clicks":
        if i==0:
            return "=0"
        elif i < 5:
            return ">0 and <5"
        elif i < 10:
            return ">=5 and <10"
        elif i < 100:
            return ">=10 and <100"
        elif i < 1000:
            return ">=100 and <1000"
        else:
            return ">=1000"

def logistic_regression(cols, feature_name):
    X = dataset[:,cols]
    if log_of_features == 'True':
        X_scaled = np.log(preprocessing.scale(X)+1)
    else:
        X_scaled = preprocessing.scale(X)
    if predict_ctr_or_clicks == 'clicks':
        y = dataset[:,44]
    else:
        y = dataset[:,44]/dataset[:,9]
    y[np.where(np.isnan(y))] = 0
    temp = []
    for i in y:
        temp.append(getClass(float(i)))
    y = temp
    x_train, x_test, y_train, y_test = sklearn.cross_validation.train_test_split(X_scaled, y, test_size=0.33, random_state=42)
    regr = linear_model.LogisticRegression(multi_class='ovr')
    regr.fit(x_train, y_train)
    y_pred = regr.predict(x_test)
    [precision, recall, f1, support] = sklearn.metrics.precision_recall_fscore_support(y_test, y_pred, average=None, labels=class_ranges)
    acc = sklearn.metrics.accuracy_score(y_test, y_pred)
    cm = sklearn.metrics.confusion_matrix(y_test, y_pred, labels=class_ranges)
    plot_cm(cm, feature_name)
    for i, cr in enumerate(class_ranges):
        fout.write(feature_name+"\t"+cr+"\t"+str(precision[i])+"\t"+str(recall[i])+"\t"+str(f1[i])+"\t"+str(support[i])+"\t"+str(acc)+"\n")
    # fout.write(feature_name +"\t"+ str(np.mean((regr.predict(x_test) - y_test) ** 2)) +"\t"+ str(regr.score(x_test, y_test)) + "\n")


def plot_cm(cm, feature_name):
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm = cm_normalized
    plt.matshow(cm)
    plt.colorbar()
    tick_marks = np.arange(len(class_ranges))
    plt.xticks(tick_marks, class_ranges, rotation=45)
    plt.yticks(tick_marks, class_ranges)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig("./confusion_matrix/multiclass_svmSVCauto_"+file_prefix+"_cm_"+feature_name+".pdf", bbox_inches='tight')
    plt.close()

class_ranges = ["=0", ">0 and <5", ">=5 and <10", ">=10 and <100", ">=100 and <1000", ">=1000"]

fout = open("logisticRegression_"+file_prefix+"_metrics_domain4.txt", "w+")
fout.write("features\tctr_range\tprecision\trecall\tf1_score\tsupport\taccuracy\n")
#fout.write("features\tRSS\tVarianceScore\n")
for feature_name, feature_cols in features.items():
    #print "Features: ", feature_name
    logistic_regression(features[feature_name], feature_name)


def random_classification(strategy, fout):
    X = dataset
    X_scaled = preprocessing.scale(X)
    if predict_ctr_or_clicks == "clicks":
        y = dataset[:,44]
    else:
        y = dataset[:,44]/dataset[:,9] # all clicks / total impressions                                                                          
    y[np.where(np.isnan(y))] = 0
    y_ = []
    for i in y:
        y_.append(getClass(i))
    x_train, x_test, y_train, y_test = sklearn.cross_validation.train_test_split(X_scaled, y_, test_size=0.5, random_state=42)
    clf = sklearn.dummy.DummyClassifier(strategy=strategy)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    [precision, recall, f1, support] = sklearn.metrics.precision_recall_fscore_support(y_test, y_pred, average=None, labels=class_ranges)
    acc = sklearn.metrics.accuracy_score(y_test, y_pred)
    cm = sklearn.metrics.confusion_matrix(y_test, y_pred, labels=class_ranges)
    plot_cm(cm, strategy)
    for i, cr in enumerate(class_ranges):
        fout.write(strategy+"\t"+cr+"\t"+str(precision[i])+"\t"+str(recall[i])+"\t"+str(f1[i])+"\t"+str(support[i])+"\t"+str(acc)+"\n")

random_classification("stratified", fout)
random_classification("most_frequent", fout)

fout.close()
