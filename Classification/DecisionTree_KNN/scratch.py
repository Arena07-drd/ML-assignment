"""From-scratch NumPy KNN, Decision Tree (CART) and PCA.

They follow the scikit-learn API (fit / predict / predict_proba / transform) only so that
GridSearchCV and Pipeline work for both the library and the NumPy versions. Mixins go before BaseEstimator.
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin


class KNN(ClassifierMixin, BaseEstimator):
    def __init__(self, n_neighbors=5, weights="uniform"):
        self.n_neighbors = n_neighbors
        self.weights = weights

    def fit(self, X, y):
        self.classes_, self.y_ = np.unique(y, return_inverse=True)
        self.X_ = np.asarray(X, dtype=float)
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        # squared Euclidean distance to every training point: |a|^2 - 2ab + |b|^2
        d2 = (X ** 2).sum(1)[:, None] - 2 * X @ self.X_.T + (self.X_ ** 2).sum(1)[None]
        nn = np.argpartition(d2, self.n_neighbors - 1, axis=1)[:, :self.n_neighbors]  # k nearest, unordered
        w = np.ones(nn.shape)
        if self.weights == "distance":  # closer neighbours vote more
            w = 1 / (np.sqrt(np.maximum(np.take_along_axis(d2, nn, 1), 0)) + 1e-12)
        votes = np.eye(len(self.classes_))[self.y_[nn]] * w[..., None]  # (n_test, k, n_classes)
        votes = votes.sum(1)
        return votes / votes.sum(1, keepdims=True)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]


class DecisionTree(ClassifierMixin, BaseEstimator):
    """CART with Gini impurity. A node is a leaf (class-count array) or a tuple (feature, threshold, left, right)."""

    def __init__(self, max_depth=None, min_samples_leaf=1):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf

    def fit(self, X, y):
        self.classes_, y = np.unique(y, return_inverse=True)
        self.tree_ = self._grow(np.asarray(X, dtype=float), y, 0)
        return self

    def _grow(self, X, y, depth):
        counts = np.bincount(y, minlength=len(self.classes_)).astype(float)
        if counts.max() == len(y) or depth == self.max_depth:  # pure node or depth limit
            return counts
        split = self._best_split(X, y)
        if split is None:
            return counts
        j, t = split
        left = X[:, j] <= t
        return (j, t, self._grow(X[left], y[left], depth + 1), self._grow(X[~left], y[~left], depth + 1))

    def _best_split(self, X, y):
        """Try every feature and every threshold at once; return (feature, threshold) with lowest weighted Gini."""
        n, n_feat = X.shape
        onehot = np.eye(len(self.classes_), dtype=np.float32)[y]
        order = np.argsort(X, axis=0)  # sort each feature column
        xs = np.take_along_axis(X, order, 0)
        left = np.cumsum(onehot[order], axis=0)[:-1]  # (n-1, F, K) class counts left of each cut
        right = onehot.sum(0) - left
        nl = np.arange(1, n, dtype=np.float32)[:, None]
        nr = n - nl
        # n_left * gini_left + n_right * gini_right, with gini = 1 - sum(p^2) = 1 - sum(count^2) / n^2
        cost = (nl - np.einsum("ijk,ijk->ij", left, left) / nl) + (nr - np.einsum("ijk,ijk->ij", right, right) / nr)
        valid = (xs[1:] != xs[:-1]) & (nl >= self.min_samples_leaf) & (nr >= self.min_samples_leaf)
        cost = np.where(valid, cost, np.inf)
        i, j = np.unravel_index(cost.argmin(), cost.shape)
        if not np.isfinite(cost[i, j]):
            return None
        return j, (xs[i, j] + xs[i + 1, j]) / 2  # cut halfway between neighbouring values

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        out = np.empty((len(X), len(self.classes_)))
        for r, x in enumerate(X):
            node = self.tree_
            while isinstance(node, tuple):
                j, t, l, rt = node
                node = l if x[j] <= t else rt
            out[r] = node / node.sum()
        return out

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]


class PCA(TransformerMixin, BaseEstimator):
    """PCA via eigen-decomposition of the covariance matrix. n_components: int, or float in (0,1) = variance to keep."""

    def __init__(self, n_components=2):
        self.n_components = n_components

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(0)
        Xc = X - self.mean_
        cov = Xc.T @ Xc / (len(X) - 1)  # (F, F) covariance matrix
        vals, vecs = np.linalg.eigh(cov)  # symmetric matrix -> eigh; ascending order
        idx = np.argsort(vals)[::-1]
        self.eigenvalues_ = np.maximum(vals[idx], 0)
        vecs = vecs[:, idx]
        self.explained_variance_ratio_ = self.eigenvalues_ / self.eigenvalues_.sum()
        k = self.n_components
        if isinstance(k, float):  # smallest k that reaches the requested cumulative variance
            k = int(np.searchsorted(np.cumsum(self.explained_variance_ratio_), k) + 1)
        self.n_components_ = k
        self.components_ = vecs[:, :k]  # (F, k) principal axes as columns
        return self

    def transform(self, X):
        return (np.asarray(X, dtype=float) - self.mean_) @ self.components_
