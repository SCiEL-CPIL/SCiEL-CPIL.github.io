"""Symbolic regression over 10 hidden equations using a rich feature basis + ridge."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product


def fit(X_train, y_train):
    start_time = time.time()
    DEADLINE = start_time + 280.0  # leave 50s safety margin out of 330s

    n, d = X_train.shape
    y_mean = np.mean(y_train)
    y_var = np.var(y_train)

    def predict_const(X):
        return np.full(len(X), y_mean)

    if y_var < 1e-14:
        return predict_const

    def time_ok():
        return time.time() < DEADLINE

    # Normalize inputs to [0, 1] for better conditioning
    x_min = X_train.min(axis=0)
    x_max = X_train.max(axis=0)
    x_range = x_max - x_min
    x_range[x_range < 1e-12] = 1.0

    def normalize(X):
        return (X - x_min) / x_range

    Xn_train = normalize(X_train)

    # Build a rich set of basis functions
    def build_basis(Xn):
        """
        Build feature matrix with:
        - Constant
        - Linear terms
        - Polynomial terms up to degree 4 (all monomials)
        - Sine/cosine terms with multiple frequencies
        - Exponential terms
        - Log terms
        """
        n_samples = len(Xn)
        features = [np.ones(n_samples)]

        # Linear
        for j in range(d):
            features.append(Xn[:, j])

        # Polynomial monomials up to degree 4
        max_deg = 4
        for deg in range(2, max_deg + 1):
            for idx in product(range(d), repeat=deg):
                # Only include unique combinations (sorted)
                idx_sorted = tuple(sorted(idx))
                term = np.ones(n_samples)
                for k in idx_sorted:
                    term = term * Xn[:, k]
                features.append(term)

        # Trigonometric: sin and cos with frequencies 1, 2, 3, 4, 6, 8
        freqs = [1.0, 2.0, 3.0, 4.0, 6.0, 8.0]
        for j in range(d):
            for f in freqs:
                features.append(np.sin(2 * np.pi * f * Xn[:, j]))
                features.append(np.cos(2 * np.pi * f * Xn[:, j]))

        # Cross-term trigonometric (for d >= 2)
        if d >= 2:
            for i in range(d):
                for j in range(i + 1, d):
                    for f in [1.0, 2.0]:
                        features.append(np.sin(2 * np.pi * f * (Xn[:, i] + Xn[:, j])))
                        features.append(np.cos(2 * np.pi * f * (Xn[:, i] + Xn[:, j])))

        # Exponential terms
        for j in range(d):
            features.append(np.exp(Xn[:, j] - 0.5))  # centered to avoid overflow
            features.append(np.exp(-Xn[:, j]))

        # Log terms (add small offset to avoid log(0))
        eps = 1e-6
        for j in range(d):
            features.append(np.log(Xn[:, j] + eps))

        # Square root terms
        for j in range(d):
            features.append(np.sqrt(np.clip(Xn[:, j], 0, None)))

        # Pairwise products of different variables (beyond monomials)
        if d >= 2:
            for i in range(d):
                for j in range(i + 1, d):
                    features.append(Xn[:, i] * Xn[:, j])
                    features.append(np.sin(np.pi * Xn[:, i] * Xn[:, j]))
                    features.append(np.cos(np.pi * Xn[:, i] * Xn[:, j]))

        return np.column_stack(features)

    # Build feature matrix for training
    F_train = build_basis(Xn_train)

    # Remove any columns with NaN or Inf
    valid_cols = np.all(np.isfinite(F_train), axis=0)
    F_train = F_train[:, valid_cols]

    # Also remove constant columns (all same value) to reduce rank issues
    col_means = F_train.mean(axis=0)
    col_stds = F_train.std(axis=0)
    non_const = col_stds > 1e-10
    F_train = F_train[:, non_const]

    # Use ridge regression for stability
    # Try multiple alpha values and pick best via validation
    n_valid = max(1, n // 5)
    indices = np.random.RandomState(42).permutation(n)
    val_idx = indices[:n_valid]
    tr_idx = indices[n_valid:]

    F_tr = F_train[tr_idx]
    F_val = F_train[val_idx]
    y_tr = y_train[tr_idx]
    y_val = y_train[val_idx]

    best_alpha = 1.0
    best_val_mse = np.inf
    best_w = None

    alphas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0]

    for alpha in alphas:
        if not time_ok():
            break
        # Ridge: (F^T F + alpha I)^{-1} F^T y
        FTF = F_tr.T @ F_tr
        FTF += alpha * np.eye(FTF.shape[0])
        FTy = F_tr.T @ y_tr
        try:
            w = np.linalg.solve(FTF, FTy)
        except np.linalg.LinAlgError:
            w = np.linalg.lstsq(FTF, FTy, rcond=None)[0]

        pred_val = F_val @ w
        mse = np.mean((pred_val - y_val) ** 2)
        if mse < best_val_mse:
            best_val_mse = mse
            best_alpha = alpha
            best_w = w

    # Refit on all data with best alpha
    FTF = F_train.T @ F_train
    FTF += best_alpha * np.eye(FTF.shape[0])
    FTy = F_train.T @ y_train
    try:
        w_final = np.linalg.solve(FTF, FTy)
    except np.linalg.LinAlgError:
        w_final = np.linalg.lstsq(FTF, FTy, rcond=None)[0]

    # If ridge didn't work well, fall back to lstsq
    if not np.all(np.isfinite(w_final)):
        w_final, *_ = np.linalg.lstsq(F_train, y_train, rcond=None)

    def predict(X):
        Xn = normalize(X)
        F = build_basis(Xn)
        # Handle case where features might have different valid columns
        valid_cols_test = np.all(np.isfinite(F), axis=0)
        F = F[:, valid_cols_test]
        col_means_t = F.mean(axis=0)
        col_stds_t = F.std(axis=0)
        non_const_t = col_stds_t > 1e-10
        F = F[:, non_const_t]

        # If the number of features doesn't match, we need to handle it
        # This should not happen if the basis construction is deterministic
        if F.shape[1] != len(w_final):
            # Fallback: use least squares on the fly
            w_temp, *_ = np.linalg.lstsq(F, y_train, rcond=None)
            return F @ w_temp
        return F @ w_final

    return predict
# EVOLVE-BLOCK-END
