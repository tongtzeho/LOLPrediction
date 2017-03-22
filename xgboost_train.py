import numpy as np
import xgboost as xgb
from data import *
# label need to be 0 to num_class-1
champion_num = 134
train_X, train_Y, test_X, test_Y = read_file_both_sides("AramDataSet624.txt", "ChampionList624.txt", champion_num)
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
num_round = 500
bst = xgb.train(param, xg_train, num_round, watchlist )
# get prediction
pred = bst.predict( xg_test )
print ('predicting, classification error=%f' % (sum( int(pred[i]) != test_Y[i] for i in range(len(test_Y))) / float(len(test_Y)) ))
bst.save_model('aram_xgboost.model')