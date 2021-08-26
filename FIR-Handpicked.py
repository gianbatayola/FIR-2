import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from FIR_model import Model
from itertools import combinations
import time

dataset = pd.read_csv('heart.csv')

X = dataset.iloc[:, [0, 3, 4, 6, 8, 9, 11]].values
features = dataset.iloc[:, [0, 3, 4, 6, 8, 9, 11]].columns
y = dataset.iloc[:, 13].values
scale = StandardScaler()
X = StandardScaler().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 / 3, random_state=0)
n_test_samples = len(X_test)

model = Model(7, 8, 5, n_test_samples)
ev = model.evaluate(X_train, y_train, X_test, y_test)
ranks = model.rank(X_test, y_test, features)
print(ev, ranks)

ncr_acc = []
ncr_feats = []
perm = combinations([0, 1, 2, 3, 4, 5, 6], 4)
perm = list(perm)

startTime = time.time()
for i in range(len(perm)):
    X_copy = X
    y_copy = y
    features_copy = features

    features_copy = np.delete(features, perm[i], 0)

    X_copy = np.delete(X_copy, perm[i], 1)

    X_train, X_test, y_train, y_test = train_test_split(X_copy, y_copy, test_size=1 / 3,
                                                        random_state=0)

    n_test_samples = len(X_test)

    model = Model(3, 8, 8, n_test_samples)
    acc = model.evaluate(X_train, y_train, X_test, y_test)
    feats = model.rank(X_test, y_test, features_copy)

    ncr_acc.append(acc)
    ncr_feats.append(feats)
print(time.time() - startTime)

best_acc = max(ncr_acc)
best_acc_index = ncr_acc.index(best_acc)
worst_acc = min(ncr_acc)
worst_acc_index = ncr_acc.index(worst_acc)

print('Best accuracy was', best_acc)
print('Best features were', ncr_feats[best_acc_index])
print('')
print('Worst accuracy was', worst_acc)
print('Worst features were', ncr_feats[worst_acc_index])

