import numpy as np
from sklearn import svm
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
import math
import pickle

train_set = np.loadtxt('adult.data.csv', delimiter=',')
test_set = np.loadtxt('adult.test.csv', delimiter=',')

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('train_set_size')
args = parser.parse_args()

train_set_size = int(args.train_set_size)
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
# C_list = [i/2.0 for i in range(0, 12)]
# gamma_list = [i/2.0 for i in range(-9, -3)]

# C_list = range(-5, 30)
C_list = [i*2 for i in range(4, 10)]
# gamma_list = range(-30, 5)
gamma_list = [-7]

parameters = {"C": [2**i for i in C_list], "gamma": [2**i for i in gamma_list]}
clf = GridSearchCV(svm.SVC(cache_size=1000), parameters, n_jobs=8)
# print clf
# clf.fit(X, y)
clf.fit(X, y)

# print clf.best_estimator_
# print pd.DataFrame(clf.cv_results_)
print clf.best_params_
print math.log(clf.best_params_['C'], 2), math.log(clf.best_params_['gamma'], 2)
print clf.best_score_

with open('search_result'+str(C_list[0])+'_'+str(C_list[-1])+'_gamma_'+str(gamma_list[0])+'_'+str(gamma_list[-1])+'_number'+str(train_set_size)+'.pkl', 'wb') as f:
    pickle.dump(clf.cv_results_, f)

# print clf.cv_results_