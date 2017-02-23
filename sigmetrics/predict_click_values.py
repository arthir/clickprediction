import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pylab as pl

import math
import numpy as np
import sklearn as sklearn
from sklearn import *
from sklearn.multiclass import OneVsRestClassifier
import sklearn.dummy
import sklearn.tree
import sklearn.neighbors

data_file = "urls_newsmedia_dataset.domain4.txt"
user_file = "/home/mgabielk/twitter_bitly_crawler/users_data.txt"
user_url_file = "/home/mgabielk/twitter_bitly_crawler/users_urls.col"

dataset = np.loadtxt(data_file, delimiter="\t", skiprows=1)

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
features['4hr of clicks+impressions'] = [38, 39, 40, 41,45,46,47,48]
#features['avg usr ctr'] = [1]

users_posting_urls = dict() # [user] -> [urls]
urls_posted_by_users = dict() # [url] -> [users]
with open(user_url_file, "rb") as f:
    for line in f:
        fields = line.strip().split("\t")
        user = fields[0]
        url = fields[1]
        if user not in users_posting_urls:
            users_posting_urls[user] = []
        users_posting_urls[user].append(url)
        if url not in urls_posted_by_users:
            urls_posted_by_users[url] = []
        urls_posted_by_users[url].append(user)

def getClass(i):
    if i==0:
        return "=0"
    elif i < 5:
        return ">0 and <5"
    elif i < 10:
        return ">5 and <10"
    elif i < 100:
        return ">10 and <100"
    elif i < 1000:
        return ">100 and <1000"
    else:
        return ">1000"

class_ranges = ["=0", ">0 and <5", ">5 and <10", ">10 and <100", ">100 and <1000", ">1000"]

def multi_classification(cols, feature_name, log=FALSE):
    X = dataset[:,cols]
    if log:
        X_scaled = np.log(X)
    else:
        X_scaled = preprocessing.scale(X)
    y = dataset[:,[11,44,9]] # url_id, all clicks, all impressions
    y[np.where(np.isnan(y))] = 0
    x_train, x_test, y_train_temp, y_test_temp = sklearn.cross_validation.train_test_split(X_scaled, y, test_size=0.5, random_state=42)
    y_train_byurl = dict()
    y_test_byurl = dict()
    for y_t in y_train_temp:
        y_train_byurl[str(int(y_t[0]))] = y_t
    for y_t in y_test_temp:
        y_test_byurl[str(int(y_t[0]))] = y_t
    #if "avg usr ctr" in feature_name: # compute average user ctr WITHOUT the urls in the test set
    # y has both the class and url id
    avg_user_ctr_from_training = dict()
    for user, urls in users_posting_urls.items():
        url_ctrs = []
        for url in urls:
            if url in y_train_byurl:
                url_ctrs.append(float(y_train_byurl[url][1])/float(y_train_byurl[url][2]))
        if len(url_ctrs) == 0:
            avg_user_ctr_from_training[user] = 0
        else:
            avg_user_ctr_from_training[user] = np.mean(url_ctrs)      
    avg_url_ctr_from_training = dict()
    for url, users in urls_posted_by_users.items():
        user_ctrs = []
        for user in users:
            user_ctrs.append(avg_user_ctr_from_training[user])
        avg_url_ctr_from_training[url] = np.mean(user_ctrs)    
    y_train = []
    y_test = []    
    x_train_user_ctr = [] # avg of all users who post; but need to make this all users who post in a time period
    x_test_user_ctr = []
    for i in y_train_temp:
        y_train.append(getClass(i[1]))
        url_id = str(int(i[0]))
        if url_id not in avg_url_ctr_from_training:
            x_train_user_ctr.append(0)
        else:
            x_train_user_ctr.append(avg_url_ctr_from_training[url_id])
    for i in y_test_temp:
        y_test.append(getClass(i[1]))
        url_id = str(int(i[0]))
        if url_id not in avg_url_ctr_from_training:
            x_test_user_ctr.append(0)
        else:
            x_test_user_ctr.append(avg_url_ctr_from_training[url_id])
    x_train = np.c_[x_train, x_train_user_ctr]
    x_test = np.c_[x_test, x_test_user_ctr]
    clf = sklearn.multiclass.OneVsRestClassifier(sklearn.svm.SVC(random_state=0, class_weight='balanced'))
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print(sklearn.metrics.classification_report(y_test, y_pred))
    prf1s = sklearn.metrics.precision_recall_fscore_support(y_test, y_pred, average=None, labels=class_ranges)
    acc = sklearn.metrics.accuracy_score(y_test, y_pred)
    print class_ranges
    print sklearn.metrics.confusion_matrix(y_test, y_pred, labels=class_ranges)
    cm = sklearn.metrics.confusion_matrix(y_test, y_pred, labels=class_ranges)
    plot_cm(cm, feature_name)
    return [prf1s, acc]

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
    #plt.title("Confusion matrix "+ feature_name)
    plt.savefig("click_multiclass_svmSVCauto_withuserinfo_domain4_cm_"+feature_name+".pdf", bbox_inches='tight')
    plt.close()

#multi_classification([10,13,14,15,16], "test")

fout = open("click_multiclass_svmSVCauto_withuserinfo_log_metrics_domain4.txt", "w+")
fout.write("features\tctr_range\tprecision\trecall\tf1_score\tsupport\taccuracy\n")
for feature_name, feature_cols in features.items():
    print "Features: ", feature_name
    [[precision, recall, f1, support], acc] = multi_classification(feature_cols, feature_name, log=TRUE)
    for i, cr in enumerate(class_ranges):
        fout.write(feature_name+"\t"+cr+"\t"+str(precision[i])+"\t"+str(recall[i])+"\t"+str(f1[i])+"\t"+str(support[i])+"\t"+str(acc)+"\n")

def random_classification(strategy, fout):
    X = dataset
    X_scaled = preprocessing.scale(X)
    y = dataset[:,44]
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
        print i
        fout.write(strategy+"\t"+cr+"\t"+str(precision[i])+"\t"+str(recall[i])+"\t"+str(f1[i])+"\t"+str(support[i])+"\t"+str(acc)+"\n")

random_classification("stratified", fout)
random_classification("most_frequent", fout)

fout.close()

