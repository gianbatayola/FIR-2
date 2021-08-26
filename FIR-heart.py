from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from FIR_model import Model
from itertools import combinations
import time
import pandas as pd
import numpy as np

df = pd.read_csv('heart.csv')

features = df.columns[:13]

X = df.values[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]
# scale = StandardScaler()
# scale.fit(X)
X = StandardScaler().fit_transform(X)
y = df.values[:, 13]

ncr_acc = []
ncr_feats = []
perm = combinations([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 10)
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
