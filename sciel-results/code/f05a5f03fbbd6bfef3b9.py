"""Sparse Gaussian Process / Kernel Ridge Regression for symbolic regression."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 280.0  # safety margin before 330s limit

    n, d = X_train.shape

    # Normalize inputs to [0, 1] range for better kernel conditioning
    X_min = X_train.min(axis=0)
    X_max = X_train.max(axis=0)
    X_range = X_max - X_min
    X_range[X_range < 1e-12] = 1.0
    X_norm = (X_train - X_min) / X_range

    # Normalize target to zero mean, unit variance for numerical stability
    y_mean = y_train.mean()
    y_std = y_train.std()
    if y_std < 1e-12:
        y_std = 1.0
    y_norm = (y_train - y_mean) / y_std

    # Build composite kernel: RBF + Polynomial + Sigmoid
    # RBF captures smooth local structure
    # Polynomial captures global polynomial trends
    # Sigmoid captures step-like or threshold behaviors

    def rbf_kernel(X1, X2, length_scale):
        sq_dists = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * X1 @ X2.T
        sq_dists = np.maximum(sq_dists, 0)
        return np.exp(-0.5 * sq_dists / (length_scale ** 2))

    def poly_kernel(X1, X2, degree=3, gamma=None, coef0=1.0):
        if gamma is None:
            gamma = 1.0 / X1.shape[1]
        return (gamma * X1 @ X2.T + coef0) ** degree

    def sigmoid_kernel(X1, X2, gamma=None, coef0=1.0):
        if gamma is None:
            gamma = 1.0 / X1.shape[1]
        return np.tanh(gamma * X1 @ X2.T + coef0)

    # Try multiple kernel configurations and select the best via cross-validation
    # Since we have noiseless data, we can also just pick a good fixed configuration

    # For noiseless data with 200 points, a well-tuned RBF with appropriate length scale
    # should interpolate perfectly. Let's use a grid of length scales.

    best_score = np.inf
    best_predict = None

    # Candidate length scales for RBF
    length_scales = [0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0]

    # Also try mixed kernels
    kernel_configs = []

    # Pure RBF with various length scales
    for ls in length_scales:
        kernel_configs.append(('rbf', ls, 1.0, 0.0, 0.0))

    # RBF + small polynomial
    for ls in [0.2, 0.5, 1.0]:
        kernel_configs.append(('mixed1', ls, 1.0, 0.1, 0.0))
        kernel_configs.append(('mixed2', ls, 1.0, 0.0, 0.1))

    # Sigmoid only
    kernel_configs.append(('sigmoid', 0.5, 0.0, 0.0, 1.0))

    # RBF + Sigmoid
    for ls in [0.3, 0.5, 1.0]:
        kernel_configs.append(('mixed3', ls, 0.5, 0.0, 0.5))

    # Regularization parameter (alpha) for KRR
    alphas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1]

    # Use 5-fold cross-validation to select best config
    # But with only 200 points and tight time budget, let's be efficient
    # We'll use a quick holdout validation

    n_folds = 5
    indices = np.arange(n)
    fold_size = n // n_folds

    for ki, config in enumerate(kernel_configs):
        if time.time() > deadline - 30:
            break

        ctype, ls, w_rbf, w_poly, w_sig = config

        def compute_kernel(X1, X2):
            K = np.zeros((X1.shape[0], X2.shape[0]))
            if w_rbf > 0:
                K += w_rbf * rbf_kernel(X1, X2, ls)
            if w_poly > 0:
                K += w_poly * poly_kernel(X1, X2, degree=3, gamma=1.0/d, coef0=1.0)
            if w_sig > 0:
                K += w_sig * sigmoid_kernel(X1, X2, gamma=1.0/d, coef0=1.0)
            return K

        for alpha in alphas:
            if time.time() > deadline - 20:
                break

            # Cross-validation
            cv_errors = []
            for fold in range(n_folds):
                val_idx = indices[fold * fold_size: (fold + 1) * fold_size]
                train_idx = np.concatenate([indices[:fold * fold_size], indices[(fold + 1) * fold_size:]])

                X_tr = X_norm[train_idx]
                y_tr = y_norm[train_idx]
                X_val = X_norm[val_idx]
                y_val = y_norm[val_idx]

                K_train = compute_kernel(X_tr, X_tr)
                K_val = compute_kernel(X_tr, X_val)

                # Solve (K + alpha*I) w = y
                try:
                    K_reg = K_train + alpha * np.eye(len(X_tr))
                    w = np.linalg.solve(K_reg, y_tr)
                    pred = K_val @ w
                    cv_errors.append(np.mean((pred - y_val) ** 2))
                except np.linalg.LinAlgError:
                    cv_errors.append(np.inf)

            mean_cv = np.mean(cv_errors)

            if mean_cv < best_score:
                best_score = mean_cv

                # Fit on all data
                K_all = compute_kernel(X_norm, X_norm)
                K_reg = K_all + alpha * np.eye(n)
                try:
                    w_all = np.linalg.solve(K_reg, y_norm)
                except np.linalg.LinAlgError:
                    w_all = np.linalg.lstsq(K_reg, y_norm, rcond=None)[0]

                def make_predict(w_all, compute_kernel, X_norm, X_min, X_range, y_mean, y_std):
                    def predict(X):
                        X_n = (X - X_min) / X_range
                        K_test = compute_kernel(X_n, X_norm)
                        pred_norm = K_test @ w_all
                        return pred_norm * y_std + y_mean
                    return predict

                best_predict = make_predict(w_all, compute_kernel, X_norm, X_min, X_range, y_mean, y_std)

    # If no valid predict was found, fall back to a simple RBF KRR
    if best_predict is None:
        ls = 0.5
        alpha = 1e-4

        def compute_kernel(X1, X2):
            return rbf_kernel(X1, X2, ls)

        K_all = compute_kernel(X_norm, X_norm)
        K_reg = K_all + alpha * np.eye(n)
        try:
            w_all = np.linalg.solve(K_reg, y_norm)
        except np.linalg.LinAlgError:
            w_all = np.linalg.lstsq(K_reg, y_norm, rcond=None)[0]

        def predict(X):
            X_n = (X - X_min) / X_range
            K_test = compute_kernel(X_n, X_norm)
            pred_norm = K_test @ w_all
            return pred_norm * y_std + y_mean

        best_predict = predict

    # Final safety: ensure predictions are finite
    def safe_predict(X):
        try:
            result = best_predict(X)
            result = np.asarray(result, dtype=np.float64)
            # Replace any non-finite values with the mean
            if not np.all(np.isfinite(result)):
                result = np.where(np.isfinite(result), result, y_mean)
            return result
        except Exception:
            return np.full(X.shape[0], y_mean)

    return safe_predict
# EVOLVE-BLOCK-END
