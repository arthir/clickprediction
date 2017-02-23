# take in features as arguments (or in file)
# what field to predict? The final CTR?
# not including user level info yet
# final results: accuracy? AUC?
# classification: classes of ctr (for now)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pylab as pl

import numpy as np
import sklearn as sklearn
from sklearn import *
from sklearn.multiclass import OneVsRestClassifier
import sklearn.dummy
import sklearn.tree
import sklearn.neighbors

data_file = "urls_newsmedia_dataset.domain4.txt"
# read data and center/normalize
dataset = np.loadtxt(data_file, delimiter="\t", skiprows=1)

# linear regression
def linear_regression(cols):
    X = dataset[:,cols]
    X_scaled = preprocessing.scale(X)
    y = dataset[:,44]/dataset[:,9] # all clicks / total impressions                                                    
    y[np.where(np.isnan(y))] = 0
    x_train, x_test, y_train, y_test = sklearn.cross_validation.train_test_split(X_scaled, y, test_size=0.33, random_state=42)
    regr = linear_model.LinearRegression()
    regr.fit(x_train, y_train)
    print('Coefficients: ', regr.coef_)
    # The mean square error
    print("Residual sum of squares: %f" % np.mean((regr.predict(x_test) - y_test) ** 2))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %f' % regr.score(x_test, y_test))

#linear_regression([10,13,14,15,16,17]) # is_primary + 5 hr of tweets
#linear_regression([10]) # is_primary

features = dict()
#features['is_primary'] = [10]
features['1hr of tweets'] = [13]
#features['is_primary+1hr of tweets'] = [10,13]
features['4hr of tweets'] = [13,14,15,16]
#features['is_primary+4hr of tweets'] = [10,13,14,15,16]
features['24hr of tweets'] = [13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,8,29,30,31,32,33,34,35,36]
#features['is_primary+24hr of tweets'] = [10,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,8,29,30,31,32,33,34,35,36]
features['1hr of impressions'] = [45]
#features['is_primary+1hr of impressions'] = [10,45]
features['4hr of impressions'] = [45,46,47,48]
#features['is_primary+4hr of impressions'] = [10,45,46,47,48]
features['1hr of clicks'] = [38] # TODO: this one's a little weird since we're calculating based on this
#features['is_primary+1hr of clicks'] = [10,38]
features['1hr of tweets+impressions'] = [13,45]
features['4hr of tweets+impressions'] = [13,14,15,16,45,46,47,48]
features['4hr of clicks'] = [38, 39, 40, 41]

def bin_classification(threshold, cols):
    X = dataset[:,cols]
    X_scaled = preprocessing.scale(X)
    y = dataset[:,44]/dataset[:,9] # all clicks / total impressions    
    y[np.where(np.isnan(y))] = 0
    y_ = ['High' if i > threshold else 'Low' for i in y] 
    x_train, x_test, y_train, y_test = sklearn.cross_validation.train_test_split(X_scaled, y_, test_size=0.33, random_state=42)
    clf = svm.SVC()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print(sklearn.metrics.classification_report(y_test, y_pred))
    f1 = sklearn.metrics.f1_score(y_test, y_pred, average=None)
    return {'High':f1[0], 'Low':f1[1]}

def run_bin_class():
    f1_scores_hi = dict()
    f1_scores_lo = dict()

    for f in features.keys():
        f1_scores_hi[f] = dict()
        f1_scores_lo[f] = dict()

    for i in [1e3,1e4]:#,1e5,1e6]:
        ctr = 1.0/i
        for feature_name, feature_cols in features.items():
            print "CTR: ", ctr, "\tFeatures: ", feature_name
            hilo = bin_classification(ctr, feature_cols)
            f1_scores_hi[feature_name][ctr] = hilo['High']
            f1_scores_lo[feature_name][ctr] = hilo['Low']


    fout = open("binaryClass_f1Score_domain4.txt", "w+")
    fout.write("features\tctr\tf1\tclass\n")

    for f in features.keys():
        #plt.plot(sorted(f1_scores_hi[f].keys()), [f1_scores_hi[f][i] for i in sorted(f1_scores_hi[f].keys())], label=f+"(High)", marker='o')
        #plt.plot(sorted(f1_scores_lo[f].keys()), [f1_scores_lo[f][i] for i in sorted(f1_scores_lo[f].keys())], label=f+"(Low)", marker='+')    
        for ctr in sorted(f1_scores_hi[feature_name].keys()):
            fout.write(f + "\t"+str(ctr)+"\t"+str(f1_scores_hi[f][ctr])+"\tHigh\n")
            fout.write(f + "\t"+str(ctr)+"\t"+str(f1_scores_lo[f][ctr])+"\tLow\n")
            
    fout.close()

def getClass(i):
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

class_ranges = ["<1e-6", ">1e-6 and <1e-5", ">1e-5 and <1e-4", ">1e-4 and <1e-3", ">1e-3"]

def multi_classification(cols, feature_name):
    X = dataset[:,cols]
    X_scaled = preprocessing.scale(X)
    y = dataset[:,44]/dataset[:,9] # all clicks / total impressions
    y[np.where(np.isnan(y))] = 0
    y_ = []
    for i in y:
        y_.append(getClass(i))
    x_train, x_test, y_train, y_test = sklearn.cross_validation.train_test_split(X_scaled, y_, test_size=0.5, random_state=42)
    clf = sklearn.multiclass.OneVsRestClassifier(sklearn.svm.SVC(random_state=0, class_weight='auto'))
    #clf = sklearn.multiclass.OneVsRestClassifier(sklearn.tree.DecisionTreeClassifier())
    #clf = sklearn.multiclass.OneVsRestClassifier(sklearn.neighbors.KNeighborsClassifier())
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
    plt.savefig("multiclass_svmSVCauto_domain4_cm_"+feature_name+".pdf", bbox_inches='tight')
    plt.close()

#multi_classification([10,13,14,15,16], "test")

fout = open("multiclass_svmSVCauto_metrics_domain4.txt", "w+")
#fout = open("multiclass_treeDecisionTree_metrics_domain4.txt", "w+")
#fout = open("multiclass_knn_metrics_domain4.txt", "w+")
fout.write("features\tctr_range\tprecision\trecall\tf1_score\tsupport\taccuracy\n")
for feature_name, feature_cols in features.items():
    print "Features: ", feature_name
    [[precision, recall, f1, support], acc] = multi_classification(feature_cols, feature_name)
    for i, cr in enumerate(class_ranges):
        fout.write(feature_name+"\t"+cr+"\t"+str(precision[i])+"\t"+str(recall[i])+"\t"+str(f1[i])+"\t"+str(support[i])+"\t"+str(acc)+"\n")

def random_classification(strategy, fout):
    X = dataset
    X_scaled = preprocessing.scale(X)
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
        print i
        fout.write(strategy+"\t"+cr+"\t"+str(precision[i])+"\t"+str(recall[i])+"\t"+str(f1[i])+"\t"+str(support[i])+"\t"+str(acc)+"\n")

random_classification("stratified", fout)
random_classification("most_frequent", fout)

fout.close()

