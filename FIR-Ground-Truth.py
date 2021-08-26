from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from FIR_model import Model
from itertools import combinations

import numpy as np

dataset = load_breast_cancer()

features = dataset['feature_names']
features = features[:10]

X = dataset['data']
X = np.delete(X, slice(10, 30, 1), 1)
scale = StandardScaler()
scale.fit(X)
X = scale.transform(X)
y = dataset['target']

ncr_acc = []
ncr_feats = []
perm = combinations([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 5)
perm = list(perm)

# baseline test
for i in range(len(perm)):
    X_copy = X
    y_copy = y

    X_copy = np.delete(X_copy, perm[i], 1)

    X_train, X_test, y_train, y_test = train_test_split(X_copy, y_copy, test_size=1 / 3,
                                                        random_state=0)
    n_test_samples = len(X_test)

    model = Model(10, 15, 12, n_test_samples)
    acc = model.evaluate(X_train, y_train, X_test, y_test)
    feats = model.rank(X_test, y_test, features, 6)

    ncr_acc.append(acc)
    ncr_feats.append(feats)

best_acc = max(ncr_acc)
best_acc_index = ncr_acc.index(best_acc)
worst_acc = min(ncr_acc)
worst_acc_index = ncr_acc.index(worst_acc)


print('Best accuracy was', best_acc)
print('Best features were', ncr_feats[best_acc_index])
print('')
print('Worst accuracy was', worst_acc)
print('Worst features were', ncr_feats[worst_acc_index])
