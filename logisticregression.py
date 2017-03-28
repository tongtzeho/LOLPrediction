from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from fetcher import *
champion_dict = fetch_champion_dict("champion.json")
champion_num = len(champion_dict)
X_train, y_train, X_test, y_test = fetch_both_sides("arurf", champion_dict)
clf = LogisticRegression()
clf.fit(X_train, y_train)
train_score = clf.score(X_train, y_train)
print ("Train Score = "+str(train_score))
test_score = clf.score(X_test, y_test)
print ("Test Score = "+str(test_score))
#joblib.dump(clf, "aram.m")