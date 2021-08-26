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
from FIR_Rand_model import Model
#from FIR_functions import add_noise

import numpy as np

# extract the data
dataset = load_breast_cancer()

features = dataset['feature_names']
features = features[:10]

X = dataset['data']
X = np.delete(X, slice(10, 30, 1), 1)
scale = StandardScaler()
scale.fit(X)
X = scale.transform(X)
y = dataset['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 / 3, random_state=0)
n_test_samples = len(X_test)

# use the model
model = Model(10, 15, 12, n_test_samples)
model.evaluate(X_train, y_train, X_test, y_test)
model.rank(X_train, y_train, features)

