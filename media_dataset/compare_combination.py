'''
Combine method to estimate shares with our method to estimate clicks from shares.
To be careful - be clear on what the inputs and outputs are

run: workon compare
'''

import sklearn
from sklearn import linear_model
from sklearn.model_selection import cross_val_predict
import matplotlib.pyplot as plt

import csv
import pandas
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics

def mean_absolute_percentage_error(y_true, y_pred): 
    # not going to account for zeros cleanly
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def run_lin_prediction(X, y):
    #print "train vs test: ", len(X_train),len(X_test)

    model = sklearn.linear_model.LinearRegression(fit_intercept=True)
    if len(X.shape) <= 1:
        #model.fit(X_train.reshape(-1, 1), y_train)
        #y_pred = model.predict(X_test.reshape(-1, 1))
        predicted = cross_val_predict(model, X.reshape(-1,1), y, cv=10)
    else:
        #model.fit(X_train, y_train)
        #y_pred = model.predict(X_test)
        predicted = cross_val_predict(model, X, y, cv=10)
    return predicted #y_pred

fout = open("szabopintotransform.txt", "w+")
#print "method\tcps_calc\tfeatures\tto_predict\tMAE\tMSE\tmedianAE\tR2\tMAPE"    
fout.write("method\tcps_calc\tfeatures\tto_predict\tMAE\tMSE\tmedianAE\tR2\tMAPE\n")
def print_metrics(y_test, y_pred, method, feature, predictor, cps="-"):
    mae = metrics.mean_absolute_error(y_test, y_pred)
    mse =  metrics.mean_squared_error(y_test, y_pred)
    medianae = metrics.median_absolute_error(y_test, y_pred) 
    r2 = metrics.r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred) 
    #print method, "\t", cps, "\t", feature, "\t", predictor, "\t", mae, "\t",  mse, "\t", metrics.median_absolute_error(y_test, y_pred) , "\t",  metrics.r2_score(y_test, y_pred), "\t", mean_absolute_percentage_error(y_test, y_pred)

    print ("%s\t%s\t%s\t%s\t%0.3f\t%0.3f\t%0.3f\t%0.3f\t%0.3f" % (method, cps, feature, predictor, mae, mse, medianae, r2, mape))
    fout.write(("%s\t%s\t%s\t%s\t%0.3f\t%0.3f\t%0.3f\t%0.3f\t%0.3f\n" % (method, cps, feature, predictor, mae, mse, medianae, r2, mape)))
    #fig, ax = plt.subplots()
    #ax.scatter(y_test, y_pred)
    #ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
    #ax.set_xlabel('Measured')
    #ax.set_ylabel('Predicted')
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred)
    ax.set_xlabel("True value of " + feature)
    ax.set_ylabel("Predicted value of " + feature)
    ax.set_title(method + " \nmethod to predict " + predictor)
    y = y_test
    #ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
    
    fig.savefig(method.split(";")[0]+"_"+feature+"_"+predictor+".png")

    x = y_test
    y = y_pred
    #x = np.random.randn(8873)
    #y = np.random.randn(8873)
    '''
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=50)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    plt.clf()
    plt.imshow(heatmap.T, extent=extent, origin='lower')
    plt.show()
    '''


# run other method to estimate shares
def estimate_szabo(data):
    # number of shares at 24hr = linear regression on log(number of shares at 1hr)
    print "-------------------"
    print "Szabo method: log(shares at 24hr) = a log(shares at 1hr) + b"
    d = pandas.merge( data[data.hour==1],  data[data.hour==24][['_id', 'total_retweets']], on='_id')

    X=np.log(d['est_retweets']+1)
    #X=X.replace('-inf', 0)
    y=np.log(d['total_retweets_y']+1)

    print len(X), len(y)
    if len(X) != len(y):
        print "ERROR: the features and class should have the same number of datapoint"
    lr = sklearn.linear_model.LinearRegression()
    #predicted = cross_val_predict(lr, X.reshape(-1, 1), y, cv=10)
    #print predicted

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    y_pred = run_lin_prediction(X_train, y_train, X_test, y_test)
    y_test = np.exp(y_test)
    y_pred = np.exp(y_pred)
    print_metrics(y_test, y_pred)

def estimate_szabo_clicks(data):
    # number of clicks at 24hr = linear regression on log(number of clicks at 1hr)
    print "-------------------"
    print "Szabo method: log(clicks at 24hr) = a log(clicks at 1hr) + b"
    d = pandas.merge( data[data.hour==1],  data[data.hour==24][['_id', 'total_clicks']], on='_id')

    #d = d.fillna(0) # getting rid of the empty data points
    d = d[np.isfinite(d['est_clicks'])]
    X=np.log(d['est_clicks']+1)
    y=np.log(d['total_clicks_y']+1)

    print "Dataset lengths: ", len(X), len(y)
    #print np.isnan(X.any()), np.all(np.isfinite(X)), X[~np.isfinite(X)]

    if len(X) != len(y):
        print "ERROR: the features and class should have the same number of datapoint"

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    y_pred = run_lin_prediction(X_train, y_train, X_test, y_test)
    y_test = np.exp(y_test)
    y_pred = np.exp(y_pred)
    print_metrics(y_test, y_pred)

def estimate_pinto_general(data, feature, predictor):
    #print "-------------------"
    #print "Pinto method: log("+predictor+" at 24hr) = a log("+feature+" at 1hr) + b log("+feature+" at 4hr) + c "
    d = pandas.merge( data[data.hour==1],  data[data.hour==24][['_id', predictor]], on='_id')
    d = pandas.merge(d,  data[data.hour==4][['_id', feature]], on="_id")

    d = d[np.isfinite(d[feature+"_x"])]
    d = d[np.isfinite(d[feature+"_y"])]
    if 'est_clicks' in d.columns:
        d = d[np.isfinite(d['est_clicks'])]
    X=np.log(d[[feature+'_x', feature+"_y"]]+1)
    y=np.log(d[predictor+'_y']+1)

    #print len(X), len(y)
    if len(X) != len(y):
        print "ERROR: the features and class should have the same number of datapoint"

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    y_pred = run_lin_prediction(X_train, y_train, X_test, y_test)
    y_test = np.exp(y_test)-1
    y_pred = np.exp(y_pred)-1
    print_metrics(y_test, y_pred, "pinto", feature+"_hr1&hr4", predictor)



def estimate_szabo_general(data, feature, predictor):
    # number of clicks at 24hr = linear regression on log(number of clicks at 1hr)
    #print "-------------------"
    #print "Szabo method: log("+predictor+" at 24hr) = a log("+feature+" at 1hr) + b"
    '''
    d = pandas.merge( data[data.hour==1],  data[data.hour==24][['_id', predictor]], on='_id')

    d = d[np.isfinite(d[feature])]
    d = d[np.isfinite(d['est_clicks'])]
    X=np.log(d[feature]+1)
    y=np.log(d[predictor+"_y"]+1)
    '''
    d = data
    X = np.log(d["hour1_"+feature]+1)        
    y = np.log(d["hour24_" + predictor]+1)      
    #print "Dataset lengths: ", len(X), len(y)
    if len(X) != len(y):
        print "ERROR: the features and class should have the same number of datapoint"

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    y_pred = run_lin_prediction(X_train, y_train, X_test, y_test)
    y_test = np.exp(y_test)
    y_pred = np.exp(y_pred)
    print_metrics(y_test, y_pred, "szabo", feature+"_hr1", predictor)


# run our old method to estimate clicks from shares
def transfer_impressions_to_clicks_hourly(data):
    # num clicks at hr i = cpi at hr1 * num impressions at hr i
    # i for now = 24 hr
    print "-------------------"
    print "Szabo method to predict impressions; ours to convert to clicks"
    feature = "est_retweets"
    predictor = "total_impressions"
    d = pandas.merge( data[data.hour==1],  data[data.hour==24][['_id', predictor, "total_clicks"]], on='_id')

    d = d[np.isfinite(d[feature])]
    d = d[np.isfinite(d['est_clicks'])]
    #d[~np.isfinite(d['est_clicks'])]['est_clicks'] = 0
    d['cpi_1'] = d['est_clicks']/d['est_impressions']
    #d[~np.isfinite(d['cpi_1'])]['cpi_1'] = 0 
    #d[~np.isfinite(d['cpi_1'])] = 0
    d['log_feature'] = np.log(d[feature]+1)
    d['log_predictor'] = np.log(d[[predictor+"_y"]]+1)
    X = d[['log_feature', 'cpi_1']]
    y = d[['log_predictor', 'total_clicks_y']]

    print "Dataset lengths: ", len(X), len(y)
    if len(X) != len(y):
        print "ERROR: the features and class should have the same number of datapoint"

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    y_pred = run_lin_prediction(X_train['log_feature'], y_train['log_predictor'], X_test['log_feature'], y_test['log_predictor'])

    print_metrics(np.exp(y_test['log_predictor'])-1, np.exp(y_pred)-1, "szabo", feature, predictor)

    pred_clicks = X_test['cpi_1'] * (np.exp(y_pred)-1)
    #print X_test[~np.isfinite(X_test)]
    #print "\t\t\t--------------"
    print_metrics(y_test['total_clicks_y'], pred_clicks, "szabo+our transform (impressions->clicks)", feature, predictor+"then total_clicks")
    print "here"

def convert_and_estimate_szabo(data, feature, predictor):
    # can't do without more dataq
    estimate_szabo_general(data, est_feature, predictor)



def transfer_impressions_to_clicks_dayahead():
    # features: 
    print "ignoreing for now"


def transfer_shares_to_clicks_hourly(data, predictor, cps_clicks, cps_shares):
    #print "-------------------"
    #print "Szabo method to predict shares; ours to convert to clicks"
    d = data
    d['log_hr1_est_retweets'] = np.log(d["hour1_est_retweets"]+1)
    d['log_hr24_est_retweets'] = np.log(d["hour24_est_retweets"]+1)
    d['cps_all'] = d['total_clicks']/(d['hour24_est_retweets']+1) # double check this??
    d['cps_all'] = d[cps_clicks]/(d[cps_shares]+1) # double check this??
    #print d[['cps_all', 'hour24_est_clicks', 'hour24_est_retweets', 'total_clicks']].head(20)
    X = d[['log_hr1_est_retweets', 'cps_all']]
    y = d[['log_hr24_est_retweets', predictor]]
    
    if len(X) != len(y):
        print "ERROR: the features and class should have the same number of datapoint"
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    y_pred = run_lin_prediction(X['log_hr1_est_retweets'], y['log_hr24_est_retweets'])

    pred_shares = np.exp(y_pred) # do we need the -1????
    #print_metrics(np.exp(y_test['log_hr24_est_retweets']), pred_shares, "szabo", "hour1_est_retweets", predictor)
    print_metrics(np.exp(y['log_hr24_est_retweets']), pred_shares, "szabo", "hour1_est_retweets", predictor)

    pred_clicks = X['cps_all'] * (pred_shares-1)
    #print "\t\t\t--------------"
    print_metrics(y[predictor], pred_clicks, "szabo+transform(shares->clicks)", "hour1_est_retweets", predictor, "CPS="+cps_clicks+"/"+cps_shares)



def transfer_shares_to_clicks_hourly_pinto(data, predictor, cps_clicks, cps_shares):
    d = data
    d['log_hr1_est_retweets'] = np.log(d["hour1_est_retweets"]+1)
    d['log_hr2_est_retweets'] = np.log(d["hour2_est_retweets"]+1)
    d['log_hr3_est_retweets'] = np.log(d["hour3_est_retweets"]+1)
    d['log_hr4_est_retweets'] = np.log(d["hour4_est_retweets"]+1)
    d['log_hr24_est_retweets'] = np.log(d["hour24_est_retweets"]+1)
    d['cps_all'] = d['total_clicks']/(d['hour24_est_retweets']+1) # double check this??
    d['cps_all'] = d[cps_clicks]/(d[cps_shares]+1) # double check this??
    #print d[['cps_all', 'hour24_est_clicks', 'hour24_est_retweets', 'total_clicks']].head(20)
    X = d[['log_hr1_est_retweets', 'log_hr2_est_retweets', 'log_hr3_est_retweets', 'log_hr4_est_retweets', 'cps_all']]
    y = d[['log_hr24_est_retweets', predictor]]
    
    if len(X) != len(y):
        print "ERROR: the features and class should have the same number of datapoint"

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    y_pred = run_lin_prediction(X[['log_hr1_est_retweets', 'log_hr2_est_retweets', 'log_hr3_est_retweets', 'log_hr4_est_retweets']], y['log_hr24_est_retweets'])
    pred_shares = np.exp(y_pred)# do we need the -1????
    print_metrics(np.exp(y['log_hr24_est_retweets']), pred_shares, "pinto", "hour1_est_retweets", predictor)

    pred_clicks = X['cps_all'] * (pred_shares-1)
    #print "\t\t\t--------------"
    print_metrics(y[predictor], pred_clicks, "pinto + transform (shares->clicks)", "hour1_est_retweets,hour2_est_retweets,hour3_est_retweets,hour4_est_retweets", predictor, "CPS="+cps_clicks+"/"+cps_shares)


# compute MAPE, r2 for part1 and part2

# read data

data = pandas.read_csv("fitted_temporal_dataset.csv")
#print data.head()['actual_clicks']
#print data[data.hour==1][['actual_clicks', 'hour']].head()

#estimate_szabo(data)
#estimate_szabo_clicks(data)
'''
estimate_szabo_general(data, "est_clicks", "total_clicks")
estimate_szabo_general(data, "est_retweets", "total_retweets")
#transfer_shares_to_clicks_hourly(data) # doesn't yet work
estimate_szabo_general(data, "est_impressions", "total_impressions")

estimate_szabo_general(data, "est_retweets", "total_clicks")
estimate_szabo_general(data, "est_retweets", "total_impressions")
'''
#transfer_impressions_to_clicks_hourly(data)
#estimate_szabo_general(data, "est_retweets", "est_retweets")

cps_pairs = [
    ('hour1_est_clicks', 'hour1_est_retweets'),
    ( 'hour1_actual_clicks', 'hour1_est_retweets'),
    ('hour24_est_clicks', 'hour24_est_retweets'),
    ('hour24_actual_clicks', 'hour24_est_retweets'),
    ('total_clicks', 'hour24_est_retweets')
    ]

for cps_pair in cps_pairs:
    #print "==============================================="
    #print cps_pair
    transfer_shares_to_clicks_hourly(data, 'total_clicks', cps_pair[0], cps_pair[1])
    transfer_shares_to_clicks_hourly(data, 'hour24_est_clicks', cps_pair[0], cps_pair[1])
    transfer_shares_to_clicks_hourly_pinto(data, 'total_clicks', cps_pair[0], cps_pair[1])
    transfer_shares_to_clicks_hourly_pinto(data, 'hour24_est_clicks', cps_pair[0], cps_pair[1])

    #print "==============================================="



'''
estimate_pinto_general(data, "est_clicks", "total_clicks")
estimate_pinto_general(data, "est_retweets", "total_retweets")
estimate_pinto_general(data, "est_impressions", "total_impressions")

estimate_pinto_general(data, "est_retweets", "total_clicks")
estimate_pinto_general(data, "est_retweets", "total_impressions")
'''

# can also do pinto on converted bitly clicks (from icwsm paper)
#convert_and_estimate_pinto(data, "est_clicks", "total_clicks")
#estimate_pinto(data)
