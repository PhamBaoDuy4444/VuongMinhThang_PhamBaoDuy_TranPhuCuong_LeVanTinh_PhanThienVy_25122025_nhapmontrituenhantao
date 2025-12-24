# File: email-spam-project/src/custom_models.py

import numpy as np
from sklearn.tree import DecisionTreeRegressor

class CustomMultinomialNB:
    """
    Triển khai thuật toán Multinomial Naive Bayes từ đầu.
    """
    def __init__(self, alpha=1.0):
        self.alpha = alpha  # Tham số làm mượt Laplace
        self._classes = None
        self._class_priors = None
        self._feature_log_prob = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)

        # Khởi tạo các biến
        self._class_priors = np.zeros(n_classes)
        self._feature_counts = np.zeros((n_classes, n_features))

        # Tính toán xác suất tiên nghiệm và đếm từ cho mỗi lớp
        for i, c in enumerate(self._classes):
            X_c = X[y == c]
            self._class_priors[i] = len(X_c) / n_samples
            self._feature_counts[i, :] = X_c.sum(axis=0)

        # Áp dụng Laplace Smoothing và tính log likelihood
        smoothed_feature_counts = self._feature_counts + self.alpha
        total_words_per_class = self._feature_counts.sum(axis=1) + self.alpha * n_features
        
        self._feature_log_prob = np.log(smoothed_feature_counts / total_words_per_class[:, np.newaxis])

    def _predict_log_proba(self, X):
        return X @ self._feature_log_prob.T + np.log(self._class_priors)

    def predict(self, X):
        log_probas = self._predict_log_proba(X)
        return self._classes[np.argmax(log_probas, axis=1)]


class SimpleGradientBoostingClassifier:
    """
    Triển khai một phiên bản đơn giản của Gradient Boosting Machine từ đầu.
    """
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self._initial_prediction = None
        self._trees = []

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def fit(self, X, y):
        n_samples = X.shape[0]
        
        # 1. Khởi tạo dự đoán ban đầu (log-odds)
        self._initial_prediction = np.log(y.mean() / (1 - y.mean()))
        current_predictions = np.full(n_samples, self._initial_prediction)

        # 2. Lặp và xây dựng các cây
        for _ in range(self.n_estimators):
            probabilities = self._sigmoid(current_predictions)
            residuals = y - probabilities
            
            tree = DecisionTreeRegressor(max_depth=self.max_depth, random_state=42)
            tree.fit(X, residuals)
            self._trees.append(tree)
            
            update = tree.predict(X)
            current_predictions += self.learning_rate * update

    def predict_proba(self, X):
        current_predictions = np.full(X.shape[0], self._initial_prediction)
        
        for tree in self._trees:
            current_predictions += self.learning_rate * tree.predict(X)
            
        probabilities = self._sigmoid(current_predictions)
        return np.vstack((1 - probabilities, probabilities)).T

    def predict(self, X):
        probabilities = self.predict_proba(X)[:, 1]
        return (probabilities >= 0.5).astype(int)