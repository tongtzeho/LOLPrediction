from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from data import *
champion_num = 134
X_train, y_train, X_test, y_test = read_file_both_sides("AramDataSet624.txt", "ChampionList624.txt", champion_num)
#clf = RandomForestClassifier(n_estimators=8, n_jobs=-1)
#clf = svm.SVC()
clf = LogisticRegression()
clf.fit(X_train, y_train)
train_score = clf.score(X_train, y_train)
print ("Train Score = "+str(train_score))
test_score = clf.score(X_test, y_test)
print ("Test Score = "+str(test_score))
joblib.dump(clf, "aram.m")
