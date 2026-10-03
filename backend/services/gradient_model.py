from __future__ import annotations

import numpy as np


class GradientBoostingRegressor:
    def __init__(self, n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        self.coef_ = None
        self.intercept_ = 0.0
        self.feature_importances_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.intercept_ = float(np.mean(y))
        centered_X = X - X.mean(axis=0)
        centered_y = y - self.intercept_

        prediction = np.full_like(y, self.intercept_, dtype=float)
        coef = np.zeros(X.shape[1], dtype=float)

        for _ in range(self.n_estimators):
            residual = y - prediction
            grad_coef, *_ = np.linalg.lstsq(centered_X, residual, rcond=None)
            update = self.learning_rate * grad_coef
            coef += update
            prediction = self.intercept_ + centered_X @ coef

        self.coef_ = coef
        abs_importance = np.abs(coef)
        total_importance = float(abs_importance.sum()) or 1.0
        self.feature_importances_ = abs_importance / total_importance
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if self.coef_ is None:
            raise ValueError('The model has not been fitted yet.')
        return np.clip(X @ self.coef_ + self.intercept_, 0.0, 125.0)
