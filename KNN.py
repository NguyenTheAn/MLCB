from __future__ import print_function
import numpy as np
from sklearn import neighbors, datasets
from sklearn.model_selection import train_test_split # for splitting data
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
# for evaluating results

np.random.seed(7)
iris = datasets.load_iris()
iris_X = iris.data
iris_Y = iris.target
print("Labels: ", np.unique(iris_Y))

def myweight(distances):
    sigma2 = .4 # we can change this number
    return np.exp(-distances**2/sigma2)

X_train, X_test, Y_train, Y_test = train_test_split(iris_X, iris_Y, test_size = 130)
print("Train size:", X_train.shape[0], ", test size:", X_test.shape[0])
model = DecisionTreeClassifier()
# model = neighbors.KNeighborsClassifier(n_neighbors = 7, p = 2, weights = "distance")
model.fit(X_train, Y_train)
y_pred = model.predict(X_test)
print("Accuracy of 7NN: %.2f %%" %(100*accuracy_score(Y_test, y_pred)))
