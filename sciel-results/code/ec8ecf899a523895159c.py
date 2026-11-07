"""Symbolic regression over hidden equations. Rich basis + forward feature selection."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def fit(X_train, y_train):
    start_time = time.time()
    DEADLINE = 320.0  # safety margin before 330s kill

    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    n, d = X.shape

    # Remove constant columns to avoid redundancy with intercept
    # Also handle potential issues with very large/small values

    def time_left():
        return DEADLINE - time.time()

    # Build a rich basis of features
    def build_basis(X):
        feats = []
        names = []

        # Constant
        feats.append(np.ones(len(X)))
        names.append('1')

        # Linear terms
        for j in range(d):
            feats.append(X[:, j])
            names.append(f'x{j}')

        # Quadratic terms (including cross)
        for i in range(d):
            for j in range(i, d):
                feats.append(X[:, i] * X[:, j])
                names.append(f'x{i}x{j}')

        # Cubic terms
        for i in range(d):
            for j in range(i, d):
                for k in range(j, d):
                    feats.append(X[:, i] * X[:, j] * X[:, k])
                    names.append(f'x{i}x{j}x{k}')

        # Powers: x^2, x^3, x^4 for each input
        for j in range(d):
            feats.append(X[:, j] ** 2)
            names.append(f'x{j}^2')
            feats.append(X[:, j] ** 3)
            names.append(f'x{j}^3')
            feats.append(X[:, j] ** 4)
            names.append(f'x{j}^4')

        # Exponential terms
        for j in range(d):
            # Clip to avoid overflow
            safe = np.clip(X[:, j], -50, 50)
            feats.append(np.exp(safe))
            names.append(f'exp(x{j})')
            feats.append(np.exp(-safe))
            names.append(f'exp(-x{j})')

        # Log terms (only for positive values, clip to avoid -inf)
        for j in range(d):
            safe = np.clip(X[:, j], 1e-10, None)
            feats.append(np.log(safe))
            names.append(f'log(x{j})')

        # Sqrt terms
        for j in range(d):
            safe = np.clip(X[:, j], 0, None)
            feats.append(np.sqrt(safe))
            names.append(f'sqrt(x{j})')

        # Trig terms
        for j in range(d):
            feats.append(np.sin(X[:, j]))
            names.append(f'sin(x{j})')
            feats.append(np.cos(X[:, j]))
            names.append(f'cos(x{j})')
            feats.append(np.tanh(X[:, j]))
            names.append(f'tanh(x{j})')

        # Pairwise products with trig/exp
        for i in range(d):
            for j in range(i + 1, d):
                feats.append(np.sin(X[:, i]) * X[:, j])
                names.append(f'sin(x{i})*x{j}')
                feats.append(np.cos(X[:, i]) * X[:, j])
                names.append(f'cos(x{i})*x{j}')
                feats.append(np.sin(X[:, i]) * np.sin(X[:, j]))
                names.append(f'sin(x{i})*sin(x{j})')
                feats.append(np.cos(X[:, i]) * np.cos(X[:, j]))
                names.append(f'cos(x{i})*cos(x{j})')
                feats.append(X[:, i] * np.exp(-np.clip(X[:, j], -50, 50)))
                names.append(f'x{i}*exp(-x{j})')
                feats.append(X[:, i] * np.exp(np.clip(X[:, j], -50, 50)))
                names.append(f'x{i}*exp(x{j})')
                feats.append(np.sin(X[:, i] * X[:, j]))
                names.append(f'sin(x{i}*x{j})')
                feats.append(np.cos(X[:, i] * X[:, j]))
                names.append(f'cos(x{i}*x{j})')
                feats.append(np.tanh(X[:, i] * X[:, j]))
                names.append(f'tanh(x{i}*x{j})')

        # Ratios (safe division)
        for i in range(d):
            for j in range(d):
                if i != j:
                    denom = np.clip(X[:, j], 1e-10, None)
                    feats.append(X[:, i] / denom)
                    names.append(f'x{i}/x{j}')

        # Combine all features
        B = np.column_stack(feats)
        return B, names

    # Fit with ridge regularization
    def fit_ridge(B, y, alpha):
        # Solve (B^T B + alpha I) w = B^T y
        # Use lstsq for stability
        B_reg = np.column_stack([B, alpha * np.ones((len(B), 1))])
        y_reg = np.append(y, 0.0)
        w, *_ = np.linalg.lstsq(B_reg, y_reg, rcond=None)
        return w

    # Compute MSE
    def mse(pred, y):
        return np.mean((pred - y) ** 2)

    # Build basis
    B, names = build_basis(X)
    n_feats = B.shape[1]

    # Forward feature selection
    # Start with just the intercept
    selected = [0]  # index of intercept
    best_mse = mse(B[:, selected] @ np.linalg.lstsq(B[:, selected], y, rcond=None)[0], y)

    # Keep a record of best model found
    best_w = np.linalg.lstsq(B[:, selected], y, rcond=None)[0]
    best_selected = selected.copy()

    # Add features one by one
    max_features = min(n_feats, 50)  # cap to avoid overfitting and time issues
    alpha = 1e-6  # small regularization

    for step in range(max_features - 1):
        if time.time() > DEADLINE - 10:
            break

        best_add_idx = -1
        best_add_mse = best_mse

        for idx in range(n_feats):
            if idx in selected:
                continue
            if time.time() > DEADLINE - 10:
                break

            trial_selected = selected + [idx]
            B_trial = B[:, trial_selected]
            try:
                w_trial, *_ = np.linalg.lstsq(B_trial, y, rcond=None)
                pred_trial = B_trial @ w_trial
                mse_trial = mse(pred_trial, y)
            except:
                continue

            if mse_trial < best_add_mse:
                best_add_mse = mse_trial
                best_add_idx = idx

        if best_add_idx == -1 or best_add_mse >= best_mse * 0.999:
            # No significant improvement
            break

        selected.append(best_add_idx)
        best_mse = best_add_mse
        best_selected = selected.copy()

        # Refit with regularization
        B_sel = B[:, selected]
        try:
            w, *_ = np.linalg.lstsq(B_sel, y, rcond=None)
            best_w = w
        except:
            break

    # Final prediction function
    B_sel = B[:, best_selected]
    w_final = best_w

    def predict(X_test):
        X_test = np.asarray(X_test, dtype=np.float64)
        if X_test.ndim == 1:
            X_test = X_test.reshape(1, -1)
        B_test, _ = build_basis(X_test)
        B_test_sel = B_test[:, best_selected]
        return B_test_sel @ w_final

    return predict
# EVOLVE-BLOCK-END
