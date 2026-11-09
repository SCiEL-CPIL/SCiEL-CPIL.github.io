# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 310.0  # Safety margin before 330s kill

    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)

    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    n, d = X.shape
    y_flat = y.flatten()
    y_mean = float(np.mean(y_flat))
    y_std = float(np.std(y_flat))
    y_var = float(np.var(y_flat))

    if y_var < 1e-12:
        return lambda X: np.full(X.shape[0], y_mean)

    def check_time():
        return time.time() < deadline

    # Normalize inputs to [0, 1]
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    X_range = X_max - X_min
    X_range[X_range < 1e-12] = 1.0

    def normalize(X):
        return (X - X_min) / X_range

    Xn = normalize(X).astype(np.float64)
    yn = (y_flat - y_mean) / (y_std + 1e-12)

    # Random Fourier Features (RFF) kernel approximation
    # K(x, x') ≈ sigma_f^2 * exp(-0.5 * ||x - x'||^2 / l^2)
    # Approximate with: (sigma_f / sqrt(M)) * phi(x)^T phi(x')
    # where phi(x) = [cos(x^T w_1), sin(x^T w_1), ..., cos(x^T w_M), sin(x^T w_M)]
    # w_j ~ N(0, 1/l^2 * I)

    def rff_features(Xn, l, M, seed=42):
        """Compute random Fourier features for normalized input Xn."""
        rng = np.random.RandomState(seed)
        # w_j ~ N(0, 1/l^2) for each dimension
        W = rng.normal(0.0, 1.0 / l, size=(d, M))
        b = rng.uniform(0.0, 2.0 * np.pi, size=M)
        # phi(x) = [cos(x^T w_1 + b_1), sin(x^T w_1 + b_1), ...]
        proj = Xn @ W  # (n, M)
        proj += b[None, :]
        return np.concatenate([np.cos(proj), np.sin(proj)], axis=1)  # (n, 2M)

    def rff_predict(Xn, l, M, seed, yn, ridge=1e-6):
        """Fit ridge regression on RFF features and predict."""
        Phi_train = rff_features(Xn, l, M, seed)  # (n, 2M)
        # Add bias term
        Phi_train = np.column_stack([Phi_train, np.ones(n)])
        # Ridge regression: w = (Phi^T Phi + lambda I)^-1 Phi^T y
        A = Phi_train.T @ Phi_train + ridge * np.eye(Phi_train.shape[1])
        b_vec = Phi_train.T @ yn
        try:
            w = np.linalg.solve(A, b_vec)
        except np.linalg.LinAlgError:
            w, *_ = np.linalg.lstsq(Phi_train, yn, rcond=None)
        return w, Phi_train

    def rff_eval(Xn_test, l, M, seed, w):
        """Predict using RFF features and weights."""
        Phi_test = rff_features(Xn_test, l, M, seed)
        Phi_test = np.column_stack([Phi_test, np.ones(Xn_test.shape[0])])
        return Phi_test @ w

    # Hyperparameter search: length_scale l, number of features M
    # Use cross-validation to select best (l, M)
    n_folds = 5
    fold_size = n // n_folds
    indices = np.arange(n)
    np.random.seed(0)
    np.random.shuffle(indices)
    folds = [indices[i * fold_size:(i + 1) * fold_size] for i in range(n_folds)]

    # Candidate hyperparameters
    l_candidates = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
    M_candidates = [50, 100, 200, 400]
    seeds = [42, 123, 456]

    best_val_mse = np.inf
    best_l = 0.5
    best_M = 100
    best_seed = 42

    # Evaluate combinations via CV
    for l in l_candidates:
        if not check_time():
            break
        for M in M_candidates:
            if not check_time():
                break
            for seed in seeds:
                if not check_time():
                    break
                fold_errors = []
                for k in range(n_folds):
                    val_idx = folds[k]
                    train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != k])
                    try:
                        w, _ = rff_predict(Xn[train_idx], l, M, seed, yn[train_idx])
                        preds = rff_eval(Xn[val_idx], l, M, seed, w)
                        mse = np.mean((preds - yn[val_idx]) ** 2)
                        fold_errors.append(mse)
                    except Exception:
                        fold_errors.append(np.inf)
                avg_mse = np.mean(fold_errors)
                if avg_mse < best_val_mse:
                    best_val_mse = avg_mse
                    best_l = l
                    best_M = M
                    best_seed = seed

    # Fit final model with best hyperparameters on all data
    try:
        w_final, _ = rff_predict(Xn, best_l, best_M, best_seed, yn)
    except Exception:
        # Fallback to polynomial
        w_final = None

    # Also fit a polynomial model as fallback/complement
    def poly_features(Xn, degree=2):
        """Generate polynomial features up to given degree."""
        # All monomials of degree <= degree
        terms = [np.ones(Xn.shape[0])]
        for deg in range(1, degree + 1):
            # Generate all combinations with repetition of size deg
            from itertools import combinations_with_replacement
            for combo in combinations_with_replacement(range(d), deg):
                term = np.ones(Xn.shape[0])
                for j in combo:
                    term *= Xn[:, j]
                terms.append(term)
        return np.column_stack(terms)

    def fit_poly(Xn, yn, degree=2, ridge=1e-4):
        Phi = poly_features(Xn, degree)
        A = Phi.T @ Phi + ridge * np.eye(Phi.shape[1])
        b_vec = Phi.T @ yn
        try:
            w = np.linalg.solve(A, b_vec)
        except np.linalg.LinAlgError:
            w, *_ = np.linalg.lstsq(Phi, yn, rcond=None)
        return w, Phi

    poly_w, poly_Phi = fit_poly(Xn, yn, degree=2)

    # Ensemble: combine RFF and polynomial predictions
    # Use CV to determine mixing weight
    best_mix = 0.5  # weight for RFF
    best_ensemble_mse = np.inf
    for mix in [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]:
        fold_errors = []
        for k in range(n_folds):
            val_idx = folds[k]
            train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != k])
            try:
                if w_final is not None:
                    w_rff_fold, _ = rff_predict(Xn[train_idx], best_l, best_M, best_seed, yn[train_idx])
                    preds_rff = rff_eval(Xn[val_idx], best_l, best_M, best_seed, w_rff_fold)
                else:
                    preds_rff = yn[val_idx]  # fallback
                w_poly_fold, _ = fit_poly(Xn[train_idx], yn[train_idx], degree=2)
                Phi_val = poly_features(Xn[val_idx], degree=2)
                preds_poly = Phi_val @ w_poly_fold
                preds_ens = mix * preds_rff + (1 - mix) * preds_poly
                mse = np.mean((preds_ens - yn[val_idx]) ** 2)
                fold_errors.append(mse)
            except Exception:
                fold_errors.append(np.inf)
        avg_mse = np.mean(fold_errors)
        if avg_mse < best_ensemble_mse:
            best_ensemble_mse = avg_mse
            best_mix = mix

    # Final prediction function
    def predict(X):
        Xn_test = normalize(X)
        preds_list = []
        if w_final is not None:
            try:
                preds_rff = rff_eval(Xn_test, best_l, best_M, best_seed, w_final)
                preds_list.append(preds_rff)
            except Exception:
                pass
        # Polynomial prediction
        try:
            Phi_test = poly_features(Xn_test, degree=2)
            preds_poly = Phi_test @ poly_w
            preds_list.append(preds_poly)
        except Exception:
            pass
        if not preds_list:
            # Ultimate fallback: mean
            return np.full(Xn_test.shape[0], y_mean)
        # Weighted ensemble
        if len(preds_list) == 2:
            preds = best_mix * preds_list[0] + (1 - best_mix) * preds_list[1]
        else:
            preds = preds_list[0]
        preds = preds * y_std + y_mean
        # Ensure finite
        preds = np.where(np.isfinite(preds), preds, y_mean)
        return preds

    # Validate
    try:
        test_preds = predict(X)
        if np.any(np.isnan(test_preds)) or np.any(np.isinf(test_preds)):
            raise ValueError("Non-finite predictions")
    except Exception:
        # Fallback to simple linear
        X_aug = np.column_stack([Xn, np.ones(n)])
        w_lin, *_ = np.linalg.lstsq(X_aug, yn, rcond=None)
        def predict_fallback(X):
            Xn_test = normalize(X)
            X_aug_test = np.column_stack([Xn_test, np.ones(Xn_test.shape[0])])
            return (X_aug_test @ w_lin) * y_std + y_mean
        return predict_fallback

    return predict

# EVOLVE-BLOCK-END
