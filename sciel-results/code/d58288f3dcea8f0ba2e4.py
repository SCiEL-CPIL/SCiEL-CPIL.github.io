"""Feature-based ridge regression with polynomial and trigonometric features."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def fit(X_train, y_train):
    t_start = time.time()
    deadline = 270.0  # safety margin before 330s limit

    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    n, d = X.shape

    # Center and scale X for stable feature computation
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_std[X_std < 1e-10] = 1.0
    X_scaled = (X - X_mean) / X_std

    def build_features(Xs):
        """Build a rich feature set from scaled inputs."""
        n_samples = Xs.shape[0]
        cols = [np.ones(n_samples)]
        
        # Raw features
        for j in range(d):
            cols.append(Xs[:, j])
        
        # Squared features
        for j in range(d):
            cols.append(Xs[:, j] ** 2)
        
        # Cubic features
        for j in range(d):
            cols.append(Xs[:, j] ** 3)
        
        # Quartic features
        for j in range(d):
            cols.append(Xs[:, j] ** 4)
        
        # Pairwise products
        for i in range(d):
            for j in range(i + 1, d):
                cols.append(Xs[:, i] * Xs[:, j])
        
        # Trigonometric features (sine and cosine)
        for j in range(d):
            for freq in [0.5, 1.0, 2.0]:
                cols.append(np.sin(freq * Xs[:, j]))
                cols.append(np.cos(freq * Xs[:, j]))
        
        # Interaction terms with trig
        for j in range(d):
            cols.append(Xs[:, j] * np.sin(Xs[:, j]))
            cols.append(Xs[:, j] * np.cos(Xs[:, j]))
        
        # Pairwise trig interactions
        for i in range(d):
            for j in range(i + 1, d):
                cols.append(np.sin(Xs[:, i]) * np.cos(Xs[:, j]))
                cols.append(np.cos(Xs[:, i]) * np.sin(Xs[:, j]))
        
        return np.column_stack(cols)

    # Build feature matrix for training data
    F_train = build_features(X_scaled)
    n_features = F_train.shape[1]

    # Simple K-fold cross-validation for hyperparameter selection
    def kfold_split(n, k, seed=42):
        rng = np.random.RandomState(seed)
        indices = rng.permutation(n)
        folds = np.array_split(indices, k)
        for i in range(k):
            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
            yield train_idx, val_idx

    kf_splits = list(kfold_split(n, 5))

    # Grid search over regularization strength
    alphas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0]
    best_alpha = 1e-3
    best_cv_score = np.inf

    for alpha in alphas:
        if time.time() > deadline - 30:
            break
        
        cv_scores = []
        for train_idx, val_idx in kf_splits:
            F_tr = F_train[train_idx]
            y_tr = y[train_idx]
            F_val = F_train[val_idx]
            y_val = y[val_idx]
            
            try:
                # Ridge regression: solve (F^T F + alpha*I) beta = F^T y
                A = F_tr.T @ F_tr + alpha * np.eye(n_features)
                b = F_tr.T @ y_tr
                beta = np.linalg.solve(A, b)
                pred_val = F_val @ beta
                mse = np.mean((pred_val - y_val) ** 2)
                var_y = np.var(y_val)
                if var_y < 1e-10:
                    nmse = mse
                else:
                    nmse = mse / var_y
                cv_scores.append(nmse)
            except np.linalg.LinAlgError:
                cv_scores.append(np.inf)
        
        mean_cv = np.mean(cv_scores)
        if mean_cv < best_cv_score:
            best_cv_score = mean_cv
            best_alpha = alpha

    # Fit on all data with best alpha
    try:
        A = F_train.T @ F_train + best_alpha * np.eye(n_features)
        b = F_train.T @ y
        beta = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        # Fallback: use pseudo-inverse
        beta = np.linalg.lstsq(F_train.T @ F_train + best_alpha * np.eye(n_features), 
                               F_train.T @ y, rcond=None)[0]

    def predict(X_test):
        X_test = np.asarray(X_test, dtype=np.float64)
        X_test_scaled = (X_test - X_mean) / X_std
        F_test = build_features(X_test_scaled)
        preds = F_test @ beta
        # Ensure finite predictions
        preds = np.where(np.isfinite(preds), preds, 0.0)
        return preds

    return predict
# EVOLVE-BLOCK-END
