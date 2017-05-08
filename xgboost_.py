import numpy as np
import xgboost as xgb
from fetcher import *
# label need to be 0 to num_class-1
champion_dict = fetch_champion_dict("champion136.json")
champion_num = len(champion_dict)

#X_train, y_train, X_test, y_test = read_file_one_side_old('AramDataSet38W.txt', 'ChampionList624.txt', 134)
#X_train, y_train, X_test, y_test = read_file_one_side('AramDataSet624.txt', 'ChampionList624.txt', 134)
#X_train, y_train, X_test, y_test = fetch_one_side_tgp('arurf', champion_dict)
#X_train, y_train, X_test, y_test = fetch_one_side_riot('11', 'MATCHED_GAME', 'RANKED_SOLO_5x5', 'CLASSIC', ('1490241600000', '1491321600000'), champion_dict)
#X_train, y_train, X_test, y_test = fetch_one_side_riot('11', 'MATCHED_GAME', 'NORMAL', 'CLASSIC', ('1490241600000', '1491321600000'), champion_dict)
#X_train, y_train, X_test, y_test = fetch_one_side_riot('12', 'MATCHED_GAME', 'ARAM_UNRANKED_5x5', 'ARAM', ('1490241600000', '1491321600000'), champion_dict)

#X_train, y_train, X_test, y_test = fetch_one_side_riot('11', 'MATCHED_GAME', 'RANKED_SOLO_5x5', 'CLASSIC', ('1491451200000', '1492531200000'), champion_dict)
#X_train, y_train, X_test, y_test = fetch_one_side_riot('11', 'MATCHED_GAME', 'RANKED_FLEX_SR', 'CLASSIC', ('1491451200000', '1492531200000'), champion_dict)
#X_train, y_train, X_test, y_test = fetch_one_side_riot('11', 'MATCHED_GAME', 'NORMAL', 'CLASSIC', ('1491451200000', '1492531200000'), champion_dict)
#X_train, y_train, X_test, y_test = fetch_one_side_riot('12', 'MATCHED_GAME', 'ARAM_UNRANKED_5x5', 'ARAM', ('1491451200000', '1492531200000'), champion_dict)
#X_train, y_train, X_test, y_test = fetch_one_side_riot('11', 'MATCHED_GAME', 'ARSR', 'ARSR', ('1491451200000', '1492531200000'), champion_dict)

#X_train, y_train, X_test, y_test = fetch_one_side_riot('12', 'MATCHED_GAME', 'ARAM_UNRANKED_5x5', 'ARAM', ('1492660800000', '1493740800000'), champion_dict)
X_train, y_train, X_test, y_test = fetch_one_side_riot('12', 'MATCHED_GAME', 'KING_PORO', 'KINGPORO', ('1492660800000', '1493740800000'), champion_dict)

#X_train, y_train, X_test, y_test = read_file_both_sides_old('AramDataSet38W.txt', 'ChampionList624.txt', 134)
#X_train, y_train, X_test, y_test = read_file_both_sides('AramDataSet624.txt', 'ChampionList624.txt', 134)
#X_train, y_train, X_test, y_test = fetch_both_sides_tgp('arurf', champion_dict)
#X_train, y_train, X_test, y_test = fetch_both_sides_riot('11', 'MATCHED_GAME', 'RANKED_SOLO_5x5', 'CLASSIC', ('1490241600000', '1491321600000'), champion_dict)
#X_train, y_train, X_test, y_test = fetch_both_sides_riot('11', 'MATCHED_GAME', 'NORMAL', 'CLASSIC', ('1490241600000', '1491321600000'), champion_dict)
#X_train, y_train, X_test, y_test = fetch_both_sides_riot('12', 'MATCHED_GAME', 'ARAM_UNRANKED_5x5', 'ARAM', ('1490241600000', '1491321600000'), champion_dict)

#X_train, y_train, X_test, y_test = fetch_both_sides_riot('11', 'MATCHED_GAME', 'RANKED_SOLO_5x5', 'CLASSIC', ('1491451200000', '1492531200000'), champion_dict)
#X_train, y_train, X_test, y_test = fetch_both_sides_riot('11', 'MATCHED_GAME', 'RANKED_FLEX_SR', 'CLASSIC', ('1491451200000', '1492531200000'), champion_dict)
#X_train, y_train, X_test, y_test = fetch_both_sides_riot('11', 'MATCHED_GAME', 'NORMAL', 'CLASSIC', ('1491451200000', '1492531200000'), champion_dict)
#X_train, y_train, X_test, y_test = fetch_both_sides_riot('12', 'MATCHED_GAME', 'ARAM_UNRANKED_5x5', 'ARAM', ('1491451200000', '1492531200000'), champion_dict)
#X_train, y_train, X_test, y_test = fetch_both_sides_riot('11', 'MATCHED_GAME', 'ARSR', 'ARSR', ('1491451200000', '1492531200000'), champion_dict)

#X_train, y_train, X_test, y_test = fetch_both_sides_riot('12', 'MATCHED_GAME', 'ARAM_UNRANKED_5x5', 'ARAM', ('1492660800000', '1493740800000'), champion_dict)
#X_train, y_train, X_test, y_test = fetch_both_sides_riot('12', 'MATCHED_GAME', 'KING_PORO', 'KINGPORO', ('1492660800000', '1493740800000'), champion_dict)

xg_train = xgb.DMatrix(X_train, label=y_train)
xg_test = xgb.DMatrix(X_test, label=y_test)
# setup parameters for xgboost
param = {}
param['booster'] = 'gbtree'
param['silent'] = 1
param['eta'] = 0.1
param['max_depth'] = 10
param['gamma'] = 0.01
param['subsample'] = 0.7
param['lambda'] = 1
param['alpha'] = 1
param['objective'] = 'multi:softmax'
param['num_class'] = 2
watchlist = [ (xg_train,'train'), (xg_test, 'test') ]
num_round = 800
bst = xgb.train(param, xg_train, num_round, watchlist )
# get prediction
pred = bst.predict( xg_test )
print ('predicting, classification error=%f' % (sum( int(pred[i]) != y_test[i] for i in range(len(y_test))) / float(len(y_test)) ))
#st.save_model('aram_xgboost.model')
