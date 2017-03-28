import numpy as np
import xgboost as xgb
from fetcher import *
# label need to be 0 to num_class-1
champion_dict = fetch_champion_dict("champion.json")
champion_num = len(champion_dict)
X_train, y_train, X_test, y_test = fetch_both_sides("arurf", champion_dict)
xg_train = xgb.DMatrix(train_X, label=train_Y)
xg_test = xgb.DMatrix(test_X, label=test_Y)
# setup parameters for xgboost
param = {}
param['booster'] = 'gbtree'
param['silent'] = 1
param['eta'] = 0.1
param['max_depth'] = 8
param['gamma'] = 0.01
param['subsample'] = 0.7
param['lambda'] = 1
param['alpha'] = 1
param['objective'] = 'multi:softmax'
param['num_class'] = 2
watchlist = [ (xg_train,'train'), (xg_test, 'test') ]
num_round = 50
bst = xgb.train(param, xg_train, num_round, watchlist )
# get prediction
pred = bst.predict( xg_test )
print ('predicting, classification error=%f' % (sum( int(pred[i]) != test_Y[i] for i in range(len(test_Y))) / float(len(test_Y)) ))
#st.save_model('aram_xgboost.model')
