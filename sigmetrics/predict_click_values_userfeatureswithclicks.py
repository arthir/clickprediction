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
early_users_file = "/home/mgabielk/urls_1st_5_posters.txt"

dataset = np.loadtxt(data_file, delimiter="\t", skiprows=1, dtype=float)

file_prefix = sys.argv[1]
predict_ctr_or_clicks = sys.argv[2]
log_of_features = sys.argv[3]

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
num_user_features_stored = len(user_features[user])
        

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

early_users = dict() # [url] -> [user1..user5]
with open(early_users_file, "rb") as f:
    for line in f:
        fields = line.strip().split("\t")
        url = fields[0]
        early_users[url] = []
        for u in fields[1:]:
            early_users[url].append(u)

# add for the first 5 users to post
cols_to_add = np.ndarray((1,5*num_user_features_stored))
# avg the values for every url's users
for url_info in dataset:
    url_id = str(int(url_info[11]))
    if url_id not in early_users:
        cols_to_add = np.vstack((cols_to_add, np.zeros(5*num_user_features_stored)))
        continue
    # take the first five users
    values_to_add = []
    for user in early_users[url_id]:
        if user == "-1":
            for i in xrange(0,num_user_features_stored):
                values_to_add.append(0)
            continue
        for key, value in user_features[user].items():
            values_to_add.append(value)
    cols_to_add = np.vstack((cols_to_add, values_to_add))

cols_to_add = np.delete(cols_to_add,0,0) # remove the dummy first row

newdata = np.c_[dataset, cols_to_add]
dataset = newdata

features = dict()
share_features = dict()
share_features['1hr of shares'] = [13]
share_features['4hr of shares'] = [13,14,15,16]                                                                               
share_features['1hr of impressions'] = [45]
share_features['4hr of impressions'] = [45,46,47,48] 
share_features['1hr of shares+impressions'] = [13,45] 
share_features['4hr of shares+impressions'] = [13,14,15,16,45,46,47,48] 
share_features['1hr of clicks'] = [38] 
share_features['1hr of clicks+impressions'] = [38,45]                                                                                
share_features['4hr of clicks'] = [38, 39, 40, 41] 
share_features['4hr of clicks+impressions'] = [38, 39, 40, 41,45,46,47,48]
features["num_primary_urls_x_users_1-5"] = [69,73, 77, 81, 85]
features["num_followers_x_users_1-5"] = [70, 74, 78, 82, 86]
features["verified_x_users_1-5"] = [71, 75, 79, 83, 87]
features["num_urls_x_users_1-5"] = [72, 76, 80, 84, 88]
features["score_median_x_users_1-5"] = [89, 90, 91, 92, 93]
features["score_95_x_users_1-5"] = [94, 95, 96, 97, 98]
features["score_median+score_95_x_users_1-5"] = [89, 90, 91, 92, 93,94, 95, 96, 97, 98]
features["all_x_users_1-5"] = [69,73, 77, 81, 85,70, 74, 78, 82, 86,71, 75, 79, 83, 87,72, 76, 80, 84, 88,89, 90, 91, 92, 93,94, 95, 96, 97, 98]
features["all_except_scores_x_users_1-5"] = [69,73, 77, 81, 85,70, 74, 78, 82, 86,71, 75, 79, 83, 87,72, 76, 80, 84, 88]

def getClass(i):
    if predict_ctr_or_clicks == "clicks":
        if i==0:
            return "=0"
        elif i < 10:
            return ">0 and <10"
        elif i < 100:
            return ">=10 and <100"
        elif i < 1000:
            return ">=100 and <1000"
        elif i < 5000:
            return ">=1000 and <5000"
        else:
            return ">=5000"
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
    class_ranges = ["=0", ">0 and <10", ">=10 and <100", ">=100 and <1000", ">=1000 and <5000", ">=5000"]
else:
    class_ranges = ["<1e-6", ">1e-6 and <1e-5", ">1e-5 and <1e-4", ">1e-4 and <1e-3", ">1e-3"]

def get_score_median_95(y, y_train_temp):
    url_clicks = dict()
    for i in y_train_temp: # only consider the training urls
        url_clicks[int(float(i[0]))] = float(i[1])/float(i[2])
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

    return [user_score_median, user_score_95]

def add_post_calculated_col(x, feature): # features: [user] -> value
    cols_to_add = np.ndarray((1,5)) # initialize a 'row' 
    for row in x:
        url_id = str(int(float(row[11])))
        if url_id not in early_users:
            cols_to_add = np.vstack((cols_to_add, np.zeros(5)))
            continue
        values_to_add = []
        for user in early_users[url_id]:
            if user == "-1":
                values_to_add.append(0)
                continue
            if user not in feature: # might not be there since this is only calculated off of training
                values_to_add.append(1)
            else:
                values_to_add.append(feature[user])
        cols_to_add = np.vstack((cols_to_add, values_to_add))
    cols_to_add = np.delete(cols_to_add,0,0) # remove the dummy first row 
    newdata = np.c_[x, cols_to_add]
    return newdata

def get_value_from_class(i):
    if i == "=0":
        return 0
    elif i == ">0 and <10":
        return 5
    elif i == ">=10 and <100":
        return 55
    elif i == ">=100 and <1000":
        return 550
    elif i == ">=1000 and <5000":
        return 3000
    else:
        return 5000


def multi_classification(cols, feature_name, fout2):
    X = dataset
    X_scaled = X #preprocessing.scale(X) 
    y = dataset[:,[11,44,9]] # url_id, all clicks, all impressions
    #y[np.where(np.isnan(y))] = 0
    x_train, x_test, y_train_temp, y_test_temp = sklearn.cross_validation.train_test_split(X_scaled, y, test_size=0.5, random_state=42)

    [user_score_median, user_score_95] = get_score_median_95(y, y_train_temp)
    # add user scores and ctrs to X (or rather to x_train and x_test)
    x_train = add_post_calculated_col(x_train, user_score_median)
    x_train = add_post_calculated_col(x_train, user_score_95)
    x_test = add_post_calculated_col(x_test, user_score_median)
    x_test = add_post_calculated_col(x_test, user_score_95)
    # then take the subset of cols that are useful
    #print cols
    #print x_train.shape, x_test.shape
    if log_of_features:
        print "HERE!!!"
        #x_train = preprocessing.scale(np.log(x_train[:,cols]+1))
        #x_test = preprocessing.scale(np.log(x_test[:,cols]+1))
        x_train = np.log(x_train[:,cols].astype(float)+1)
        x_test = np.log(x_test[:,cols].astype(float)+1)
    else:
        x_train = x_train[:,cols]
        x_test = x_test[:,cols]

    # get rid fo useless y column
    y_train = []
    y_test = []    
    for i in y_train_temp:
        y_train.append(getClass(float(i[1])))
    for i in y_test_temp:
        y_test.append(getClass(float(i[1])))

    clf = sklearn.multiclass.OneVsRestClassifier(sklearn.svm.SVC(random_state=0, class_weight='balanced'))
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    #print(sklearn.metrics.classification_report(y_test, y_pred))
    for i, pred in enumerate(y_pred):
        fout2.write(feature_name + "\t" + str(float(y_test_temp[i][1])) + "\t" + str(get_value_from_class(y_pred[i])) + "\n")
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
fout2 = open("multiclass_svmSVCauto_"+file_prefix+"_results_domain4.txt", "w+")
fout2.write("features\treal_value\tpredicted_value\n")
for click_feature_name, click_feature_cols in share_features.items():    
    [[precision, recall, f1, support], acc] = multi_classification(click_feature_cols, click_feature_name, fout2)
    for i, cr in enumerate(class_ranges):
        fout.write(click_feature_name+"\t"+cr+"\t"+str(precision[i])+"\t"+str(recall[i])+"\t"+str(f1[i])+"\t"+str(support[i])+"\t"+str(acc)+"\n")
    for feature_name, feature_cols in features.items():
        #print "Features: ", feature_name
        combo_feature_name = click_feature_name + " and " + feature_name
        combo_feature_cols = click_feature_cols + feature_cols
        print combo_feature_name, combo_feature_cols
        [[precision, recall, f1, support], acc] = multi_classification(combo_feature_cols, combo_feature_name, fout2)
        for i, cr in enumerate(class_ranges):
            fout.write(combo_feature_name+"\t"+cr+"\t"+str(precision[i])+"\t"+str(recall[i])+"\t"+str(f1[i])+"\t"+str(support[i])+"\t"+str(acc)+"\n")
for feature_name, feature_cols in features.items():
    [[precision, recall, f1, support], acc] = multi_classification(feature_cols, feature_name, fout2)
    for i, cr in enumerate(class_ranges):
        fout.write(feature_name+"\t"+cr+"\t"+str(precision[i])+"\t"+str(recall[i])+"\t"+str(f1[i])+"\t"+str(support[i])+"\t"+str(acc)+"\n")

exit()



def random_classification(strategy, fout):
    X = dataset
    X_scaled = preprocessing.scale(X)
    if predict_ctr_or_clicks == "clicks":
        y = dataset[:,44]
    else:
        y = dataset[:,44]/dataset[:,9] # all clicks / total impressions
    #y[np.where(np.isnan(y))] = 0
    y_ = []
    for i in y:
        y_.append(getClass(float(i)))
    x_train, x_test, y_train, y_test = sklearn.cross_validation.train_test_split(X_scaled, y_, test_size=0.5, random_state=42)
    clf = sklearn.dummy.DummyClassifier(strategy=strategy)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print(sklearn.metrics.classification_report(y_test, y_pred))
    [precision, recall, f1, support] = sklearn.metrics.precision_recall_fscore_support(y_test, y_pred, average=None, labels=class_ranges)
    acc = sklearn.metrics.accuracy_score(y_test, y_pred)
    cm = sklearn.metrics.confusion_matrix(y_test, y_pred, labels=class_ranges)
    plot_cm(cm, strategy)
    for i, cr in enumerate(class_ranges):
        fout.write(strategy+"\t"+cr+"\t"+str(precision[i])+"\t"+str(recall[i])+"\t"+str(f1[i])+"\t"+str(support[i])+"\t"+str(acc)+"\n")

random_classification("stratified", fout)
random_classification("most_frequent", fout)

fout.close()

