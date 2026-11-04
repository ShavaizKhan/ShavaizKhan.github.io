import numpy as np
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn.metrics import silhouette_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline

# UCI Machine Learning Repository: Iris (dataset ID 53)
iris = fetch_ucirepo(id=53)
X = iris.data.features.to_numpy(dtype=float)
y = iris.data.targets.iloc[:, 0].to_numpy()
names = np.unique(y)
print('Feature order:', iris.data.features.columns.tolist())
print('Dataset shape:', X.shape)
print('Species:', names)
colors = ['tab:blue', 'tab:orange', 'tab:green']

# Part 2: Original feature projections
fig = plt.figure(figsize=(16, 4.5))
ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133, projection='3d')
for i, species in enumerate(names):
    mask = y == species
    ax1.scatter(X[mask, 2], np.full(mask.sum(), i), color=colors[i], alpha=0.7, label=species)
    ax2.scatter(X[mask, 2], X[mask, 3], color=colors[i], alpha=0.7)
    ax3.scatter(X[mask, 2], X[mask, 3], X[mask, 0], color=colors[i], alpha=0.7)
ax1.set(xlabel='Petal length (cm)', ylabel='Species (display only)', title='1D: Petal length')
ax1.set_yticks(range(len(names)), names)
ax2.set(xlabel='Petal length (cm)', ylabel='Petal width (cm)', title='2D: Original features')
ax3.set(xlabel='Petal length', ylabel='Petal width', zlabel='Sepal length', title='3D: Original features')
fig.legend(loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.04))
fig.tight_layout()
fig.savefig('3/media/part2_features.png', dpi=200, bbox_inches='tight')

# Parts 3 and 4: PCA on centered, unstandardized Iris features
pca = PCA(n_components=4)
X_pca = pca.fit_transform(X)
print('PCA axes (rows PC1-PC4; columns in original feature order):')
print(np.round(pca.components_, 4))
print('Explained variance ratios:', np.round(pca.explained_variance_ratio_, 6))
print(f'Two-PC variance retained: {pca.explained_variance_ratio_[:2].sum():.2%}')
print('PCs needed for >=50%:', np.searchsorted(np.cumsum(pca.explained_variance_ratio_), 0.5) + 1)

fig = plt.figure(figsize=(16, 4.5))
ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133, projection='3d')
for i, species in enumerate(names):
    mask = y == species
    ax1.scatter(X_pca[mask, 0], np.full(mask.sum(), i), color=colors[i], alpha=0.7, label=species)
    ax2.scatter(X_pca[mask, 0], X_pca[mask, 1], color=colors[i], alpha=0.7)
    ax3.scatter(X_pca[mask, 0], X_pca[mask, 1], X_pca[mask, 2], color=colors[i], alpha=0.7)
ax1.set(xlabel='PC1', ylabel='Species (display only)', title='1D PCA')
ax1.set_yticks(range(len(names)), names)
ax2.set(xlabel='PC1', ylabel='PC2', title='2D PCA')
ax3.set(xlabel='PC1', ylabel='PC2', zlabel='PC3', title='3D PCA')
fig.legend(loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.04))
fig.tight_layout()
fig.savefig('3/media/part3_pca.png', dpi=200, bbox_inches='tight')

# Part 5: Isomap using exactly 3 nearest neighbors
isomap = Isomap(n_neighbors=3, n_components=2)
X_isomap = isomap.fit_transform(X)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for i, species in enumerate(names):
    mask = y == species
    axes[0].scatter(X_pca[mask, 0], X_pca[mask, 1], color=colors[i], alpha=0.7, label=species)
    axes[1].scatter(X_isomap[mask, 0], X_isomap[mask, 1], color=colors[i], alpha=0.7)
axes[0].set(xlabel='PC1', ylabel='PC2', title='PCA (2D)')
axes[1].set(xlabel='Isomap 1', ylabel='Isomap 2', title='Isomap (3 neighbors)')
fig.legend(loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.05))
fig.tight_layout()
fig.savefig('3/media/part5_isomap_vs_pca.png', dpi=200, bbox_inches='tight')
print(f'PCA 2D silhouette score (species labels): {silhouette_score(X_pca[:, :2], y):.4f}')
print(f'Isomap 2D silhouette score (species labels): {silhouette_score(X_isomap, y):.4f}')

# Classification comparison: fit dimensionality reduction inside each training fold
# to avoid data leakage from the held-out test fold.
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

pca_classifier = make_pipeline(PCA(n_components=2), KNeighborsClassifier(n_neighbors=5))
isomap_classifier = make_pipeline(Isomap(n_neighbors=3, n_components=2), KNeighborsClassifier(n_neighbors=5))

pca_accuracy = cross_val_score(pca_classifier, X, y, cv=cv, scoring='accuracy')
isomap_accuracy = cross_val_score(isomap_classifier, X, y, cv=cv, scoring='accuracy')

print('PCA classification accuracies by fold:', np.round(pca_accuracy, 4))
print(f'PCA mean classification accuracy: {pca_accuracy.mean():.2%} (+/- {pca_accuracy.std():.2%} SD)')
print('Isomap classification accuracies by fold:', np.round(isomap_accuracy, 4))
print(f'Isomap mean classification accuracy: {isomap_accuracy.mean():.2%} (+/- {isomap_accuracy.std():.2%} SD)')

plt.show()