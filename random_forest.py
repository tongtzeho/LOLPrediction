from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from fetcher import *
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

clf = RandomForestClassifier(n_estimators=500, n_jobs=2)
clf.fit(X_train, y_train)
train_score = clf.score(X_train, y_train)
print ("Train Score = "+str(train_score))
test_score = clf.score(X_test, y_test)
print ("Test Score = "+str(test_score))
#joblib.dump(clf, "aram.m")
