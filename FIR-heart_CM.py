import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from FIR_model import Model

dataset = pd.read_csv('heart.csv')

features = dataset.columns[:14]

df = pd.DataFrame(dataset, columns=features)
corr = df.corr()

sns.heatmap(corr, annot=True, annot_kws={"fontsize": 5})
# plt.savefig('heatmap.png')

sns.pairplot(df, hue='target')
# plt.savefig('scatter.png')

X = dataset.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]].values
scale = StandardScaler()
scale.fit(X)
X = scale.transform(X)
y = dataset.iloc[:, 13].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 / 3, random_state=0)
n_test_samples = len(X_test)

model = Model(13, 8, 5, n_test_samples)
evs = model.evaluate(X_train, y_train, X_test, y_test)
ranks = model.rank(X_test, y_test, features)
print(evs, ranks)

plt.show()
