import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import json

with open('uniqueProbabilities.json', 'r') as file:
    data = json.load(file)

artist_and_title = [str(obj['artist']) + ' - ' + str(obj['title']) for obj in data][:5000]
data = [obj['probabilityArray'] for obj in data][:5000]
all_indices = set()
for song in data:
    all_indices.update(song.keys())
all_indices = sorted(list(all_indices))

X = np.zeros((len(data), len(all_indices)))
for i, song in enumerate(data):
    for j, index in enumerate(all_indices):
        X[i, j] = song.get(index, 0)

scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_normalized)

plt.figure(figsize=(10, 8))
for (x, y), label in zip(X_pca, artist_and_title):
    plt.scatter(x, y)
    plt.annotate(label, (x, y), textcoords="offset points", xytext=(0, 10), ha='center')

embedding_data = [
    {'artist_and_title': label, 'x': float(x), 'y': float(y)} for (x, y), label in zip(X_pca, artist_and_title)
]

with open('pca_embedding.json', 'w') as f:
    json.dump(embedding_data, f)

plt.scatter(X_pca[:, 0], X_pca[:, 1])
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('2D PCA Embedding of Songs')
plt.grid(True)
plt.show()
