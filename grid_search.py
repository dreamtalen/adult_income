import numpy as np
from sklearn import svm
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV

train_set = np.loadtxt('adult.data.csv', delimiter=',')
test_set = np.loadtxt('adult.test.csv', delimiter=',')

train_set_size = 30000
test_set_size = 1000

X = train_set[:train_set_size, :-1]
y = train_set[:train_set_size, -1]

#scale the data attributes
X = preprocessing.scale(X)

# normalize the data attributes
# X = preprocessing.normalize(X)


test_X = test_set[:test_set_size, :-1]

test_y = test_set[:test_set_size, -1]

#scale the data attributes
test_X = preprocessing.scale(test_X)

# normalize the data attributes
# test_X = preprocessing.normalize(test_X)

parameters = {"kernel": ["rbf"], "C": np.logspace(-9, 20, 30, base=2), "gamma": np.logspace(-14, 15, 30, base=2)}
clf = GridSearchCV(svm.SVC(cache_size=1000), parameters, n_jobs=8)
# print clf
# clf.fit(X, y)
clf.fit(X, y)

print clf.best_estimator_
print clf.best_score_
# print clf.cv_results_
print clf.best_params_

# predicted = clf.predict(test_X)

# wrong_num = sum(1 for i, j in zip(predicted, test_y) if i != j)

# accurate = 1 - wrong_num/float(test_set_size)