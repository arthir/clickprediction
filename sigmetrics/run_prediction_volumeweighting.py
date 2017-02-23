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

data_file = "urls_newsmedia_dataset.domain4.txt"
user_file = "/home/mgabielk/twitter_bitly_crawler/users_data.txt"
user_url_file = "/home/mgabielk/twitter_bitly_crawler/users_urls.col"

dataset = np.loadtxt(data_file, delimiter="\t", skiprows=1)

file_prefix = sys.argv[1]
predict_ctr_or_clicks = sys.argv[2]
log_of_features = sys.argv[3]
include_user_avg_ctr = sys.argv[4]

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

def get_value_from_class(i):
    if i == "=0":
        return 0
    elif i == ">0 and <5":
        return 2.5
    elif i == ">=5 and <10":
        return 7
    elif i == ">=10 and <100":
        return 55
    elif i == ">=100 and <1000":
        return 550
    else:
        return 1000

if predict_ctr_or_clicks == "clicks":
    class_ranges = ["=0", ">0 and <5", ">=5 and <10", ">=10 and <100", ">=100 and <1000", ">=1000"]
    predicted = [0, 2.5, 7, 55, 550, 1000]
else:
    class_ranges = ["<1e-6", ">1e-6 and <1e-5", ">1e-5 and <1e-4", ">1e-4 and <1e-3", ">1e-3"]
    predicted = [0, 2.5, 7, 55, 550, 1000] # ignore for now

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
    if include_user_avg_ctr == 'True':
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
            if predict_ctr_or_clicks == "clicks":
                y_train.append(getClass(i[1]))
            else:
                y_train.append(getClass(float(i[1])/i[2]))
            url_id = str(int(i[0]))
            if url_id not in avg_url_ctr_from_training:
                x_train_user_ctr.append(0)
            else:
                x_train_user_ctr.append(avg_url_ctr_from_training[url_id])
        for i in y_test_temp:
            if predict_ctr_or_clicks == "clicks":
               y_test.append(getClass(i[1]))
            else:
                y_test.append(getClass(float(i[1])/i[2]))
            url_id = str(int(i[0]))
            if url_id not in avg_url_ctr_from_training:
                x_test_user_ctr.append(0)
            else:
                x_test_user_ctr.append(avg_url_ctr_from_training[url_id])
        x_train = np.c_[x_train, x_train_user_ctr]
        x_test = np.c_[x_test, x_test_user_ctr]
    else:
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
    print(sklearn.metrics.classification_report(y_test, y_pred))
    prf1s = sklearn.metrics.precision_recall_fscore_support(y_test, y_pred, average=None, labels=class_ranges)
    acc = sklearn.metrics.accuracy_score(y_test, y_pred)
    # count difference of predicted number of clicks to actual number of clicks
    #predicted_clicks = [get_value_from_class(i) for i in y_pred]
    #diff_in_prediction = predicted_clicks - y_test_temp[:,1]
    diff_in_prediction = dict()
    diff_in_prediction["underestimate"] = dict()
    diff_in_prediction["overestimate"] = dict()
    diff_in_prediction["frac_underestimate"] = dict()
    diff_in_prediction["frac_overestimate"] = dict()
    diff_in_prediction["total_clicks"] = dict()
    for i, class_name in enumerate(y_pred): # i is the predicted class
        if class_name not in diff_in_prediction["underestimate"]:
            diff_in_prediction["underestimate"][class_name] = 0
            diff_in_prediction["overestimate"][class_name] = 0
            diff_in_prediction["total_clicks"][class_name] = 0
        predicted_value = get_value_from_class(class_name) 
        real_value =  y_test_temp[i,1]
        diff_in_prediction["total_clicks"][class_name] += real_value
        if predicted_value <= real_value:
            diff_in_prediction["underestimate"][class_name] += real_value - predicted_value
        else:
            diff_in_prediction["overestimate"][class_name] += predicted_value - real_value
    for class_name in class_ranges:
        diff_in_prediction["frac_underestimate"][class_name] = diff_in_prediction["underestimate"][class_name]/diff_in_prediction["total_clicks"][class_name]
        diff_in_prediction["frac_overestimate"][class_name] = diff_in_prediction["overestimate"][class_name]/diff_in_prediction["total_clicks"][class_name]
    #print diff_in_prediction
    #print prf1s
    #print class_ranges
    #print sklearn.metrics.confusion_matrix(y_test, y_pred, labels=class_ranges)
    cm = sklearn.metrics.confusion_matrix(y_test, y_pred, labels=class_ranges)
    plot_cm(cm, feature_name)
    return [prf1s, acc, diff_in_prediction]

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
fout.write("features\tctr_range\tprecision\trecall\tf1_score\tsupport\taccuracy\toverestimate\tunderestimate\ttotal_clicks\tfrac_overestimate\tfrac_underestimate\n")
for feature_name, feature_cols in features.items():
    print "Features: ", feature_name
    [[precision, recall, f1, support], acc, diff_in_prediction] = multi_classification(feature_cols, feature_name)
    for i, cr in enumerate(class_ranges):
        fout.write(feature_name+"\t"+cr+"\t"+str(precision[i])+"\t"+str(recall[i])+"\t"+str(f1[i])+"\t"+str(support[i])+"\t"+str(acc)+"\t"+str(diff_in_prediction["overestimate"][cr])+"\t"+str(diff_in_prediction["underestimate"][cr])+"\t"+str(diff_in_prediction["total_clicks"][cr])+"\t"+str(diff_in_prediction["frac_overestimate"][cr])+"\t"+str(diff_in_prediction["frac_underestimate"][cr])+"\n")

def random_classification(strategy, fout):
    X = dataset
    X_scaled = preprocessing.scale(X)
    if predict_ctr_or_clicks == "clicks":
        y = dataset[:,44]
    else:
        y = dataset[:,44]/dataset[:,9] # all clicks / total impressions
    y[np.where(np.isnan(y))] = 0
    #y_ = []
    #for i in y:
    #    y_.append(getClass(i))
    x_train, x_test, y_train_temp, y_test_temp = sklearn.cross_validation.train_test_split(X_scaled, y, test_size=0.5, random_state=42)
    y_train = []
    y_test = []
    for i in y_train_temp:
        y_train.append(getClass(i))
    for i in y_test_temp:
        y_test.append(getClass(i))
    clf = sklearn.dummy.DummyClassifier(strategy=strategy)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    [precision, recall, f1, support] = sklearn.metrics.precision_recall_fscore_support(y_test, y_pred, average=None, labels=class_ranges)
    acc = sklearn.metrics.accuracy_score(y_test, y_pred)
    diff_in_prediction = dict()
    diff_in_prediction["underestimate"] = dict()
    diff_in_prediction["overestimate"] = dict()
    diff_in_prediction["frac_underestimate"] = dict()
    diff_in_prediction["frac_overestimate"] = dict()
    diff_in_prediction["total_clicks"] = dict()
    for i, class_name in enumerate(y_pred): # i is the predicted class  
        if class_name not in diff_in_prediction["underestimate"]:
            diff_in_prediction["underestimate"][class_name] = 0
            diff_in_prediction["overestimate"][class_name] = 0
            diff_in_prediction["total_clicks"][class_name] = 0
        predicted_value = get_value_from_class(class_name)
        real_value =  float(y_test_temp[i])
        diff_in_prediction["total_clicks"][class_name] += real_value
        if predicted_value <= real_value:
            diff_in_prediction["underestimate"][class_name] += real_value - predicted_value
        else:
            diff_in_prediction["overestimate"][class_name] += predicted_value - real_value
    for class_name in class_ranges:
        if class_name not in diff_in_prediction["underestimate"]:
            diff_in_prediction["underestimate"][class_name]=0
            diff_in_prediction["overestimate"][class_name]=0
            diff_in_prediction["total_clicks"][class_name]=1 # random to avoid div by zero issues            
        diff_in_prediction["frac_underestimate"][class_name] = diff_in_prediction["underestimate"][class_name]/diff_in_prediction["total_clicks"][class_name]
        diff_in_prediction["frac_overestimate"][class_name] = diff_in_prediction["overestimate"][class_name]/diff_in_prediction["total_clicks"][class_name]

    cm = sklearn.metrics.confusion_matrix(y_test, y_pred, labels=class_ranges)
    plot_cm(cm, strategy)
    for i, cr in enumerate(class_ranges):
        fout.write(strategy+"\t"+cr+"\t"+str(precision[i])+"\t"+str(recall[i])+"\t"+str(f1[i])+"\t"+str(support[i])+"\t"+str(acc)+"\t"+str(diff_in_prediction["overestimate"][cr])+"\t"+str(diff_in_prediction["underestimate"][cr])+"\t"+str(diff_in_prediction["total_clicks"][cr])+"\t"+str(diff_in_prediction["frac_overestimate"][cr])+"\t"+str(diff_in_prediction["frac_underestimate"][cr])+"\n")

random_classification("stratified", fout)
random_classification("most_frequent", fout)

fout.close()

