"""Symbolic regression over 4 hidden 1D equations using adaptive basis regression."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.interpolate import interp1d


def _chebyshev_basis(x, n_terms):
    """Compute Chebyshev polynomials T_0..T_{n_terms-1} at points x.
    x should be in [-1, 1]."""
    n = len(x)
    terms = np.zeros((n, n_terms))
    if n_terms <= 0:
        return terms
    terms[:, 0] = 1.0
    if n_terms > 1:
        terms[:, 1] = x
    for k in range(2, n_terms):
        terms[:, k] = 2.0 * x * terms[:, k - 1] - terms[:, k - 2]
    return terms


def _fourier_basis(x, n_sin, n_cos):
    """Compute Fourier basis: 1, cos(k*pi*t), sin(k*pi*t) for t in x (normalized to [-1,1])."""
    n = len(x)
    cols = [np.ones(n)]
    for k in range(1, n_cos + 1):
        cols.append(np.cos(k * np.pi * x))
    for k in range(1, n_sin + 1):
        cols.append(np.sin(k * np.pi * x))
    return np.column_stack(cols)


def _poly_basis(x, degree):
    """Polynomial basis up to given degree."""
    n = len(x)
    cols = [np.ones(n)]
    for k in range(1, degree + 1):
        cols.append(x ** k)
    return np.column_stack(cols)


def _fit_regularized(X, y, lam):
    """Solve (X^T X + lam*I) w = X^T y with ridge regularization."""
    n_features = X.shape[1]
    A = X.T @ X + lam * np.eye(n_features)
    b = X.T @ y
    try:
        w = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        w = np.linalg.lstsq(X, y, rcond=None)[0]
    return w


def _predict_with_basis(basis_func, X_norm, w, lam):
    """Predict using basis function and weights."""
    B = basis_func(X_norm)
    return B @ w


def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 280.0  # safety margin before 330s limit

    n_samples = X_train.shape[0]
    d = X_train.shape[1]

    # All equations are 1D based on task description
    x_train = X_train[:, 0] if d >= 1 else np.ones(n_samples)

    # Normalize to [-1, 1] based on training range
    x_min = np.min(x_train)
    x_max = np.max(x_train)
    x_range = x_max - x_min
    if x_range < 1e-12:
        x_range = 1.0
    x_norm_train = 2.0 * (x_train - x_min) / x_range - 1.0

    # Sort by x for potential use
    sort_idx = np.argsort(x_norm_train)
    x_sorted = x_norm_train[sort_idx]
    y_sorted = y_train[sort_idx]

    # Candidate models: (name, basis_func, n_terms_params, description)
    # We'll try different basis functions and select via cross-validation

    def make_model(basis_type, params):
        """Create a basis function factory given type and parameters."""
        if basis_type == 'chebyshev':
            n_terms = params
            def bf(x):
                return _chebyshev_basis(x, n_terms)
        elif basis_type == 'fourier':
            n_sin = params
            n_cos = params
            def bf(x):
                return _fourier_basis(x, n_sin, n_cos)
        elif basis_type == 'poly':
            degree = params
            def bf(x):
                return _poly_basis(x, degree)
        else:
            def bf(x):
                return np.ones((len(x), 1))
        return bf

    # Define candidate models
    candidates = []

    # Chebyshev with various numbers of terms
    for n_terms in [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25]:
        candidates.append(('chebyshev', n_terms))

    # Fourier with various frequencies
    for n_freq in [2, 3, 4, 5, 6, 8, 10, 12]:
        candidates.append(('fourier', n_freq))

    # Polynomial with various degrees
    for deg in [2, 3, 4, 5, 6, 7, 8]:
        candidates.append(('poly', deg))

    # Ridge regularization strengths
    lambdas = [1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 0.1, 1.0]

    # Cross-validation: 5-fold
    n_folds = 5
    fold_size = n_samples // n_folds
    indices = np.arange(n_samples)

    best_score = np.inf
    best_model = None
    best_basis_func = None
    best_w = None
    best_lam = None

    for basis_type, params in candidates:
        if time.time() > deadline - 30.0:
            break

        basis_func = make_model(basis_type, params)
        B_train = basis_func(x_norm_train)
        n_features = B_train.shape[1]

        for lam in lambdas:
            if time.time() > deadline - 15.0:
                break

            # Quick check: if features > samples, skip unless heavily regularized
            if n_features > n_samples and lam < 1e-3:
                continue

            # Cross-validation
            cv_errors = []
            for fold in range(n_folds):
                val_start = fold * fold_size
                val_end = val_start + fold_size if fold < n_folds - 1 else n_samples
                val_idx = indices[val_start:val_end]
                train_idx = np.concatenate([indices[:val_start], indices[val_end:]])

                B_tr = B_train[train_idx]
                y_tr = y_train[train_idx]
                B_val = B_train[val_idx]
                y_val = y_train[val_idx]

                w_cv = _fit_regularized(B_tr, y_tr, lam)
                pred_val = B_val @ w_cv
                mse = np.mean((pred_val - y_val) ** 2)
                cv_errors.append(mse)

            avg_cv_mse = np.mean(cv_errors)

            if avg_cv_mse < best_score:
                best_score = avg_cv_mse
                best_model = (basis_type, params)
                best_basis_func = basis_func
                best_lam = lam

                # Fit on full training set
                w_full = _fit_regularized(B_train, y_train, lam)
                best_w = w_full

    # If no model was found (shouldn't happen), fallback to simple linear
    if best_w is None:
        x_lin = np.column_stack([np.ones(n_samples), x_norm_train])
        best_w = np.linalg.lstsq(x_lin, y_train, rcond=None)[0]
        best_basis_func = lambda x: np.column_stack([np.ones(len(x)), x])
        best_lam = 0.0

    # Final check: if we have time, try to refine with more terms or different regularization
    if time.time() < deadline - 10.0:
        # Try a few more Chebyshev terms around the best
        if best_model and best_model[0] == 'chebyshev':
            best_n = best_model[1]
            for n_try in [best_n - 2, best_n + 2, best_n + 4]:
                if n_try < 2 or n_try > 40:
                    continue
                bf_try = make_model('chebyshev', n_try)
                B_try = bf_try(x_norm_train)
                for lam_try in [best_lam * 0.5, best_lam * 2.0]:
                    if lam_try <= 0:
                        continue
                    cv_errors = []
                    for fold in range(n_folds):
                        val_start = fold * fold_size
                        val_end = val_start + fold_size if fold < n_folds - 1 else n_samples
                        val_idx = indices[val_start:val_end]
                        train_idx = np.concatenate([indices[:val_start], indices[val_end:]])
                        B_tr = B_try[train_idx]
                        y_tr = y_train[train_idx]
                        B_val = B_try[val_idx]
                        y_val = y_train[val_idx]
                        w_cv = _fit_regularized(B_tr, y_tr, lam_try)
                        pred_val = B_val @ w_cv
                        mse = np.mean((pred_val - y_val) ** 2)
                        cv_errors.append(mse)
                    avg_cv_mse = np.mean(cv_errors)
                    if avg_cv_mse < best_score:
                        best_score = avg_cv_mse
                        best_basis_func = bf_try
                        best_lam = lam_try
                        best_w = _fit_regularized(B_try, y_train, lam_try)

    # Create prediction function
    def predict(X):
        if X.ndim == 1:
            X = X.reshape(1, -1)
        x_pred = X[:, 0] if X.shape[1] >= 1 else np.ones(X.shape[0])
        # Normalize using the same mapping as training
        x_pred_norm = 2.0 * (x_pred - x_min) / x_range - 1.0
        B_pred = best_basis_func(x_pred_norm)
        preds = B_pred @ best_w
        return preds

    return predict
# EVOLVE-BLOCK-END
