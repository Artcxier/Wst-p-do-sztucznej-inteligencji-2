from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.3, random_state=42)

from sklearn.svm import SVC
clf = SVC(C=1.0, kernel='linear')
clf.fit(X_train, y_train)

from sklearn.metrics import accuracy_score
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

from sklearn.model_selection import GridSearchCV
param_grid = {'C': [0.1, 1, 10, 100], 'kernel': ['linear', 'rbf']}
grid = GridSearchCV(SVC(), param_grid, cv=5)
grid.fit(X_train, y_train)


clf_linear = SVC(C=grid.best_params_['C'], kernel='linear')
clf_linear.fit(X_train, y_train)

clf_rbf = SVC(C=grid.best_params_['C'], kernel='rbf')
clf_rbf.fit(X_train, y_train)

from sklearn.metrics import confusion_matrix
y_pred_linear = clf_linear.predict(X_test)
y_pred_rbf = clf_rbf.predict(X_test)

conf_matrix_linear = confusion_matrix(y_test, y_pred_linear)
conf_matrix_rbf = confusion_matrix(y_test, y_pred_rbf)
