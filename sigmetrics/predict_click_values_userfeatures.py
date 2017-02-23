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
import sklearn.dummy
import sklearn.tree
import sklearn.neighbors
import utils

data_file = "urls_newsmedia_dataset.domain4.txt"
user_file = "/home/mgabielk/twitter_bitly_crawler/users_data.txt"
user_url_file = "/home/mgabielk/twitter_bitly_crawler/users_urls.col"

dataset = np.loadtxt(data_file, delimiter="\t", skiprows=1)

file_prefix = sys.argv[1]
predict_ctr_or_clicks = sys.argv[2]
log_of_features = sys.argv[2]


user_features = dict()
with open(user_file, "rb") as f:
    for line in f:
        fields = line.strip().split("\t")
        user = fields[0]
        if user not in user_features:
            user_features[user] = dict()
        user_features[user]["num_followers"] = fields[3]
        user_features[user]["num_urls"] = fields[1]
        user_features[user]["num_primary_urls"] = fields[2]
        user_features[user]["verified"] = fields[8]
        

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

cols_to_add = np.ndarray((1,len(user_features[user].keys())))
# avg the values for every url's users
for url_info in dataset:
    url_id = str(int(url_info[11]))
    if url_id not in urls_posted_by_users:
        cols_to_add = np.vstack((cols_to_add, [0,0,0,0]))
        continue
    users = urls_posted_by_users[url_id]
    sum_of_values = dict()
    for user in users:
        for key, value in user_features[user].items():
            if key not in sum_of_values:
                sum_of_values[key] = 0
            sum_of_values[key] += float(value)
    avg = dict()
    for key, value in sum_of_values.items():
        avg[key] = sum_of_values[key]/len(users)
    cols_to_add = np.vstack((cols_to_add, np.array(avg.values())))

cols_to_add = np.delete(cols_to_add,0,0)
print cols_to_add

newdata = np.c_[dataset, cols_to_add]
print newdata.shape
dataset = newdata

features = dict()
features["avg_num_followers"] = [70]
features["avg_num_urls"] = [72]
features["avg_num_primary_urls"] = [69]
features["fraction_verified"] = [71]
features["avg_num_followers+avg_user_ctr"] = [70]
features["avg_num_followers+score_median"] = [70]
features["avg_num_followers+score_95"] = [70]
features["avg_num_followers+score_median+score_95"] = [70]
features["avg_num_urls+avg_user_ctr"] = [72]
features["avg_num_urls+score_median"] = [72]
features["avg_num_urls+score_95"] = [72]
features["avg_num_urls+score_median+score_95"] = [72]
features["avg_num_primary_urls+score_median"] = [69]
features["avg_num_primary_urls+score_95"] = [69]
features["avg_num_primary_urls+score_median+score_95"] = [69]

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
    else:
        if i < 1e-6:
            return "<1e-6"
        elif i < 1e-5:
            return ">1e-6 and <1e-5"
        elif i < 1e-4:
            return ">1e-5 and <1e-4"
        elif i < 1e-3:
            return ">1e-4 and <1e-3"
        else:
            return ">1e-3"

if predict_ctr_or_clicks == "clicks":
    class_ranges = ["=0", ">0 and <5", ">=5 and <10", ">=10 and <100", ">=100 and <1000", ">=1000"]
else:
    class_ranges = ["<1e-6", ">1e-6 and <1e-5", ">1e-5 and <1e-4", ">1e-4 and <1e-3", ">1e-3"]

def add_user_avg(y_train_temp, y_test_temp):
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
    return avg_url_ctr_from_training

def get_score_median_95(y, y_train_temp):
    url_clicks = dict()
    for i in y_train_temp: # only consider the training urls
        url_clicks[int(i[0])] = float(i[1])/i[2]
    median_ctr = utils.get_percentile_url(url_clicks, 50)
    percentile95_ctr =  utils.get_percentile_url(url_clicks, 95)
    # need to calculate score per user
    user_score_median = dict()
    user_score_95 = dict()
    for user in users_posting_urls:
        num_urls_with_ctr_above_median = 0
        num_urls_posting_in_training = 0
        for u in users_posting_urls[user]:
            if int(u) not in url_clicks: # might not be there because only training urls
                continue
            num_urls_posting_in_training += 1
            if url_clicks[int(u)] > median_ctr:
                num_urls_with_ctr_above_median += 1
        user_score_median[user] = utils.calc_score(0.5, num_urls_with_ctr_above_median, num_urls_posting_in_training)
        num_urls_with_ctr_above_median = 0
        for u in users_posting_urls[user]:
            if int(u) not in url_clicks: # might not be there because only training urls
                continue
            if url_clicks[int(u)] > percentile95_ctr:
                num_urls_with_ctr_above_median += 1
        user_score_95[user] = utils.calc_score(0.05, num_urls_with_ctr_above_median, num_urls_posting_in_training)
        #print user, users_posting_urls[user], user_score_median[user]

    avg_url_score_median_from_training = dict()
    avg_url_score_95_from_training = dict()
    for url, users in urls_posted_by_users.items():
        user_scores = [user_score_median[user] for user in users]
        avg_url_score_median_from_training[url] = np.mean(user_scores)
        user_scores = [user_score_95[user] for user in users]
        avg_url_score_95_from_training[url]= np.mean(user_scores)

    return [avg_url_score_median_from_training, avg_url_score_95_from_training]

def add_features_columns(x_train, x_test, y_train_temp, y_test_temp, avg_url_ctr_from_training):
    x_train_user_ctr = [] # avg of all users who post; but need to make this all users who post in a time period
    x_test_user_ctr = []
    for i in y_train_temp:
        url_id = str(int(i[0]))
        if url_id not in avg_url_ctr_from_training:
            x_train_user_ctr.append(0)
        else:
            x_train_user_ctr.append(avg_url_ctr_from_training[url_id])
    for i in y_test_temp:
        url_id = str(int(i[0]))
        if url_id not in avg_url_ctr_from_training:
            x_test_user_ctr.append(0)
        else:
            x_test_user_ctr.append(avg_url_ctr_from_training[url_id])
    x_train = np.c_[x_train, x_train_user_ctr]
    x_test = np.c_[x_test, x_test_user_ctr]
    return [x_train, x_test]

def multi_classification(cols, feature_name):
    X = dataset[:,cols]
    if log_of_features=='True':
        temp = np.log(X+1)
        X_scaled = preprocessing.scale(temp)
    else:
        X_scaled = preprocessing.scale(X)
    y = dataset[:,[11,44,9]] # url_id, all clicks, all impressions
    y[np.where(np.isnan(y))] = 0
    x_train, x_test, y_train_temp, y_test_temp = sklearn.cross_validation.train_test_split(X_scaled, y, test_size=0.5, random_state=42)

    [avg_url_score_median_from_training, avg_url_score_95_from_training] = get_score_median_95(y, y_train_temp)

    if "avg_user_ctr" in feature_name:
        avg_url_ctr_from_training = add_user_avg(y_train_temp, y_test_temp)
        [x_train, x_test] = add_features_columns(x_train, x_test, y_train_temp, y_test_temp, avg_url_ctr_from_training)
    if "score_median" in feature_name:
        [x_train, x_test] = add_features_columns(x_train, x_test, y_train_temp, y_test_temp, avg_url_score_median_from_training)
    if "score_95" in feature_name:
        [x_train, x_test] = add_features_columns(x_train, x_test, y_train_temp, y_test_temp, avg_url_score_95_from_training)

    y_train = []
    y_test = []
    for i in y_train_temp:
        if predict_ctr_or_clicks == "clicks":
            y_train.append(getClass(i[1]))
        else:
            y_train.append(getClass(float(i[1])/i[2]))
    for i in y_test_temp:
        if predict_ctr_or_clicks == "clicks":
            y_test.append(getClass(i[1]))
        else:
            y_test.append(getClass(float(i[1])/i[2]))

    clf = sklearn.multiclass.OneVsRestClassifier(sklearn.svm.SVC(random_state=0, class_weight='balanced'))
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    #print(sklearn.metrics.classification_report(y_test, y_pred))
    prf1s = sklearn.metrics.precision_recall_fscore_support(y_test, y_pred, average=None, labels=class_ranges)
    acc = sklearn.metrics.accuracy_score(y_test, y_pred)
    #print class_ranges
    #print sklearn.metrics.confusion_matrix(y_test, y_pred, labels=class_ranges)
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
    plt.savefig("./confusion_matrix/multiclass_svmSVCauto_"+file_prefix+"_cm_"+feature_name+".pdf", bbox_inches='tight')
    plt.close()

#multi_classification([10,13,14,15,16], "test")

fout = open("multiclass_svmSVCauto_"+file_prefix+"_metrics_domain4.txt", "w+")
fout.write("features\tctr_range\tprecision\trecall\tf1_score\tsupport\taccuracy\n")
for feature_name, feature_cols in features.items():
    print "Features: ", feature_name
    [[precision, recall, f1, support], acc] = multi_classification(feature_cols, feature_name)
    for i, cr in enumerate(class_ranges):
        fout.write(feature_name+"\t"+cr+"\t"+str(precision[i])+"\t"+str(recall[i])+"\t"+str(f1[i])+"\t"+str(support[i])+"\t"+str(acc)+"\n")

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

