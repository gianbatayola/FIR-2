# Gian Batayola

# Goal: Rank the importance of features for classification performance to determine which features
# are most impactful

# Two Potential Goals:
# 1. Understanding which features are most impactful for a fixed model
# 2. Selecting features for training a new model

# Steps:
# 1. Make some pseudo data with clear good and bad features
# 2. Train a model with combined good+bad data
# 3. Rank features with amazing method
#   3.1 Permutation based
#   3.2 Gradient based?
#   3.3 Perhaps with correlation stuff
# 4. Determine if ranking matches my expectations


from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from FIR_model import Model
from FIR_functions import add_noise
from itertools import combinations

import numpy as np

# extract the data
dataset = load_breast_cancer()

features = dataset['feature_names']
# 10 features might make more sense, 11-30 are error and worst values won't really work with noise
features = features[:10]

X = dataset['data']
X = np.delete(X, slice(10, 30, 1), 1)
# print(X[0])
# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html
scale = StandardScaler()
scale.fit(X)
X = scale.transform(X)
# print(X[0])
y = dataset['target']

# scale = MinMaxScaler(feature_range=(0, 1))
# scale.fit(X)
# X = scale.transform(X)
# print(X[0])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 / 3, random_state=0)
n_test_samples = len(X_test)

# set up
ncr_acc = []
ncr_feats = []
perm = combinations([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 5)
perm = list(perm)

# use the model
for i in range(len(perm)):
    model = Model(10, 15, 12, n_test_samples)
    # model.evaluate(X_train, y_train, X_test, y_test)
    # model.rank(X_train, y_train, features)

    # add noise
    X_test = add_noise(X_test, features, 5, 0, random_state=None, noisy_indices=perm[i])

    acc = model.evaluate(X_train, y_train, X_test, y_test)
    feats = model.rank(X_test, y_test, features, 5)

    ncr_acc.append(acc)
    ncr_feats.append(feats)

best_acc = max(ncr_acc)
best_acc_index = ncr_acc.index(best_acc)
print('Best accuracy was', best_acc)
print('Features were', ncr_feats[best_acc_index])
