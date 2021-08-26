import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

dataset = pd.read_csv('heart.csv')

X = dataset.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]].values
y = dataset.iloc[:, 13].values

X = StandardScaler().fit_transform(X)

pca = PCA(n_components=2)

pc = pca.fit_transform(X)

pdf = pd.DataFrame(data=pc, columns=['principal component 1', 'principal component 2'])

fdf = pd.concat([pdf, dataset['target']], axis=1)

plt.title('2 component PCA')
plt.xlabel('Principal component 1')
plt.ylabel('Principal component 2')
plt.scatter(fdf['principal component 1'], fdf['principal component 2'], c=fdf['target'])
plt.savefig('pca.png')

plt.show()
