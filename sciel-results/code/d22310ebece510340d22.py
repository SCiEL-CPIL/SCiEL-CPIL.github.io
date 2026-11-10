# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_sq_dists(A, B):
    """Compute squared Euclidean distances between rows of A and B efficiently."""
    A_sq = np.sum(A**2, axis=1, keepdims=True)
    B_sq = np.sum(B**2, axis=1, keepdims=True)
    sq_dists = A_sq + B_sq.T - 2 * (A @ B.T)
    return np.maximum(sq_dists, 0.0)

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 280  # Safety margin before 300s limit

    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)

    n_samples, n_features = X.shape

    # 1. Normalize inputs to [0, 1]
    min_vals = X.min(axis=0, keepdims=True)
    max_vals = X.max(axis=0, keepdims=True)
    range_vals = max_vals - min_vals
    range_vals[range_vals < 1e-12] = 1.0

    X_norm = (X - min_vals) / range_vals

    # 2. Split into train and validation sets for model selection
    # Use 80% train, 20% val. With 200 points, this is 160/40.
    rng = np.random.RandomState(42)
    indices = np.arange(n_samples)
    rng.shuffle(indices)
    val_size = int(0.2 * n_samples)
    val_idx = indices[:val_size]
    tr_idx = indices[val_size:]

    X_tr = X_norm[tr_idx]
    y_tr = y[tr_idx]
    X_val = X_norm[val_idx]
    y_val = y[val_idx]

    n_tr = len(y_tr)
    n_val = len(y_val)

    # Precompute squared distances for train and val sets
    sq_dists_tr_tr = _compute_sq_dists(X_tr, X_tr)
    sq_dists_val_tr = _compute_sq_dists(X_val, X_tr)

    # Hyperparameter Grid
    # RBF: gamma (inverse length scale squared)
    rbf_gammas = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]
    # Polynomial: degree, coefficient
    poly_degrees = [2, 3]
    poly_coeffs = [1.0, 2.0]
    # Periodic: period
    per_periods = [1.0, 2.0, 4.0]

    # Regularization
    alphas = [1e-4, 1e-3, 1e-2, 1e-1]

    best_val_mse = float('inf')
    best_config = None

    # Helper to evaluate a specific kernel configuration
    def evaluate_config(kernel_type, params, alpha):
        if time.time() > deadline - 10:
            return float('inf'), None

        try:
            K_tr = np.zeros((n_tr, n_tr))
            K_val_tr = np.zeros((n_val, n_tr))

            # Build Kernel
            if 'rbf' in kernel_type:
                gamma = params['gamma']
                K_tr += np.exp(-gamma * sq_dists_tr_tr)
                K_val_tr += np.exp(-gamma * sq_dists_val_tr)

            if 'poly' in kernel_type:
                deg = params['degree']
                c = params['coeff']
                # Polynomial kernel: (X @ X.T + c)^deg
                # Note: X_tr is normalized, so X @ X.T is dot product
                K_tr += (X_tr @ X_tr.T + c) ** deg
                K_val_tr += (X_val @ X_tr.T + c) ** deg

            if 'per' in kernel_type:
                period = params['period']
                # Periodic kernel
                diff_tr = X_tr[:, None, :] - X_tr[None, :, :]
                sin_term_tr = np.sin(np.pi * diff_tr / period)
                K_tr += np.exp(-0.5 * np.sum((2.0 * np.pi / period) ** 2 * sin_term_tr ** 2, axis=2))

                diff_val = X_val[:, None, :] - X_tr[None, :, :]
                sin_term_val = np.sin(np.pi * diff_val / period)
                K_val_tr += np.exp(-0.5 * np.sum((2.0 * np.pi / period) ** 2 * sin_term_val ** 2, axis=2))

            # Solve
            A = K_tr + alpha * np.eye(n_tr)
            w = np.linalg.solve(A, y_tr)

            # Predict
            y_val_pred = K_val_tr @ w
            mse_val = np.mean((y_val_pred - y_val)**2)

            return mse_val, (kernel_type, params, alpha)

        except Exception:
            return float('inf'), None

    # Search Grid
    # 1. RBF
    for gamma in rbf_gammas:
        for alpha in alphas:
            mse, cfg = evaluate_config('rbf', {'gamma': gamma}, alpha)
            if mse < best_val_mse:
                best_val_mse = mse
                best_config = cfg

    # 2. Poly
    for deg in poly_degrees:
        for c in poly_coeffs:
            for alpha in alphas:
                mse, cfg = evaluate_config('poly', {'degree': deg, 'coeff': c}, alpha)
                if mse < best_val_mse:
                    best_val_mse = mse
                    best_config = cfg

    # 3. Periodic
    for period in per_periods:
        for alpha in alphas:
            mse, cfg = evaluate_config('per', {'period': period}, alpha)
            if mse < best_val_mse:
                best_val_mse = mse
                best_config = cfg

    # 4. RBF + Poly
    for gamma in rbf_gammas:
        for deg in poly_degrees:
            for c in poly_coeffs:
                for alpha in alphas:
                    if time.time() > deadline - 5:
                        break
                    mse, cfg = evaluate_config('rbf+poly', {'gamma': gamma, 'degree': deg, 'coeff': c}, alpha)
                    if mse < best_val_mse:
                        best_val_mse = mse
                        best_config = cfg

    # 5. RBF + Periodic
    for gamma in rbf_gammas:
        for period in per_periods:
            for alpha in alphas:
                if time.time() > deadline - 5:
                    break
                mse, cfg = evaluate_config('rbf+per', {'gamma': gamma, 'period': period}, alpha)
                if mse < best_val_mse:
                    best_val_mse = mse
                    best_config = cfg

    # 6. Poly + Periodic
    for deg in poly_degrees:
        for c in poly_coeffs:
            for period in per_periods:
                for alpha in alphas:
                    if time.time() > deadline - 5:
                        break
                    mse, cfg = evaluate_config('poly+per', {'degree': deg, 'coeff': c, 'period': period}, alpha)
                    if mse < best_val_mse:
                        best_val_mse = mse
                        best_config = cfg

    # 7. All three (limited search to save time)
    # Only try a subset of hyperparameters for the full combo
    for gamma in [1.0, 4.0, 16.0]:
        for deg in [2]:
            for c in [1.0]:
                for period in [2.0]:
                    for alpha in [1e-3, 1e-2]:
                        if time.time() > deadline - 2:
                            break
                        mse, cfg = evaluate_config('rbf+poly+per', {'gamma': gamma, 'degree': deg, 'coeff': c, 'period': period}, alpha)
                        if mse < best_val_mse:
                            best_val_mse = mse
                            best_config = cfg

    # If no config found (shouldn't happen), fallback to simple RBF
    if best_config is None:
        best_config = ('rbf', {'gamma': 1.0}, 1e-3)

    kernel_type, params, best_alpha = best_config

    # Refit on ALL data with the best config
    sq_dists_full = _compute_sq_dists(X_norm, X_norm)

    K_full = np.zeros((n_samples, n_samples))

    if 'rbf' in kernel_type:
        gamma = params['gamma']
        K_full += np.exp(-gamma * sq_dists_full)

    if 'poly' in kernel_type:
        deg = params['degree']
        c = params['coeff']
        K_full += (X_norm @ X_norm.T + c) ** deg

    if 'per' in kernel_type:
        period = params['period']
        diff_full = X_norm[:, None, :] - X_norm[None, :, :]
        sin_term_full = np.sin(np.pi * diff_full / period)
        K_full += np.exp(-0.5 * np.sum((2.0 * np.pi / period) ** 2 * sin_term_full ** 2, axis=2))

    A_full = K_full + best_alpha * np.eye(n_samples)

    try:
        w_full = np.linalg.solve(A_full, y)
    except np.linalg.LinAlgError:
        # Fallback to lstsq if singular
        w_full, *_ = np.linalg.lstsq(A_full, y, rcond=None)

    # Capture variables for closure
    final_w = w_full
    final_kernel_type = kernel_type
    final_params = params
    final_min_vals = min_vals
    final_range_vals = range_vals
    final_X_norm = X_norm

    def predict(X):
        X = np.asarray(X, dtype=np.float64)
        X_norm_test = (X - final_min_vals) / final_range_vals

        n_test = X_norm_test.shape[0]
        K_test = np.zeros((n_test, n_samples))

        if 'rbf' in final_kernel_type:
            gamma = final_params['gamma']
            sq_dists_test = _compute_sq_dists(X_norm_test, final_X_norm)
            K_test += np.exp(-gamma * sq_dists_test)

        if 'poly' in final_kernel_type:
            deg = final_params['degree']
            c = final_params['coeff']
            K_test += (X_norm_test @ final_X_norm.T + c) ** deg

        if 'per' in final_kernel_type:
            period = final_params['period']
            diff_test = X_norm_test[:, None, :] - final_X_norm[None, :, :]
            sin_term_test = np.sin(np.pi * diff_test / period)
            K_test += np.exp(-0.5 * np.sum((2.0 * np.pi / period) ** 2 * sin_term_test ** 2, axis=2))

        return K_test @ final_w

    return predict
# EVOLVE-BLOCK-END
