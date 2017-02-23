import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model

regr = sklearn.linear_model.LinearRegression(fit_intercept=True)
def r2(pred, actual):
    return 1 - ( np.sum( (pred - actual)**2)/np.sum( (actual - np.mean(actual))**2 ) )

def mape(pred, actual):
    return np.median(np.abs((pred - actual)/actual))


new_df = pd.read_csv('fitted_temporal_dataset.csv')

# RANDOMLY SHUFFLE ROWS OF DF TO CREATE k-folds
np.random.seed(88)
new_df = new_df.reindex(np.random.permutation(new_df.index))
new_df = new_df[new_df['hour1_actual_imp'] > 0]
new_df = new_df.dropna()
new_df['log_primary_impressions'] = np.log(new_df['primary_impressions'])
new_df = new_df.replace([np.inf, -np.inf], 0)
new_df[new_df['log_hour1_est_clicks'] == -np.inf]

# create k-folds
num_folds = 20
subset_size = len(new_df)/num_folds
tests = []
trains = []
for i in range(num_folds):
    pre_test = new_df[i*subset_size:][:subset_size]
    if i == 0:
        pre_train = new_df[(i+1)*subset_size:]
    else:
        pre_train = new_df[:i*subset_size].append(new_df[(i+1)*subset_size:])
        tests.append(pre_test)
        trains.append(pre_train)
        
# define subset selection functions, based on MAPE and R2
def mape_subset_select(x, y, tests, trains, log_to_lin = True):
    r_values = {}
    feature_set = []

    for subset_size in range(1,len(x)+1):
        r_values[subset_size] = {}
        for feature in x:
            if feature not in feature_set:
                new_feature_set = feature_set + [feature]
                feature_set_fold_r = []
                for i in range(len(trains)):
                    train = trains[i]
                    test = tests[i]
                    regr.fit(train[new_feature_set], train[y])
                    if log_to_lin is True:
                        pred = np.exp(regr.predict(test[new_feature_set]))
                        actual = np.exp(test[y])
                    else:
                        pred = regr.predict(test[new_feature_set])
                        actual = test[y]
                    feature_set_fold_r.append(mape(pred, actual))
                feature_set_r = np.mean(feature_set_fold_r)
                r_values[subset_size][tuple(new_feature_set)] = feature_set_r
        best_feature = min(r_values[subset_size].iterkeys(), key=r_values[subset_size].get)
        feature_set.extend(list(best_feature))
        feature_set = list(set(feature_set))

    print 'MAPE'
    for subset_size, values in r_values.items():
        print subset_size
        print min(r_values[subset_size].iterkeys(), key=r_values[subset_size].get)
        print min(r_values[subset_size].itervalues())

def r2_subset_select(x, y, tests, trains, log_to_lin = True):
    r_values = {}
    feature_set = []

    for subset_size in range(1,len(x)+1):
        r_values[subset_size] = {}
        for feature in x:
            if feature not in feature_set:
                new_feature_set = feature_set + [feature]
                feature_set_fold_r = []
                for i in range(len(trains)):
                    train = trains[i]
                    test = tests[i]
                    regr.fit(train[new_feature_set], train[y])
                    if log_to_lin is True:
                        pred = np.exp(regr.predict(test[new_feature_set]))
                        actual = np.exp(test[y])
                    else:
                        pred = regr.predict(test[new_feature_set])
                        actual = test[y]
                    feature_set_fold_r.append(r2(pred, actual))
                feature_set_r = np.mean(feature_set_fold_r)
                r_values[subset_size][tuple(new_feature_set)] = feature_set_r
        best_feature = max(r_values[subset_size].iterkeys(), key=r_values[subset_size].get)
        feature_set.extend(list(best_feature))
        feature_set = list(set(feature_set))

    print 'R2'
    for subset_size, values in r_values.items():
        print subset_size
        print max(r_values[subset_size].iterkeys(), key=r_values[subset_size].get)
        print max(r_values[subset_size].itervalues())
        
# DAY AHEAD MODEL
x = ['log_hour1_f_imp', 'log_hour1_f_clicks']
y = 'log_total_clicks'
#mape_subset_select(x, y, tests, trains, log_to_lin=True)
r2_subset_select(x, y, tests, trains, log_to_lin=True)

# SZABO MODEL
x = ['log_hour1_f_clicks']
y = 'log_total_clicks'
#mape_subset_select(x, y, tests, trains, log_to_lin=True)
r2_subset_select(x, y, tests, trains, log_to_lin=True)

# PINTO MODEL
x = ['hour1_f_clicks','hour2_f_clicks','hour3_f_clicks', 'hour4_f_clicks']
y = 'total_clicks'
mape_subset_select(x, y, tests, trains, log_to_lin=False)
r2_subset_select(x, y, tests, trains, log_to_lin=False) 
