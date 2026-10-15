import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier


x, y = load_iris(return_X_y=True)
clf = RandomForestClassifier()
clf.fit(x, y)
joblib.dump(clf, "model.pkl")