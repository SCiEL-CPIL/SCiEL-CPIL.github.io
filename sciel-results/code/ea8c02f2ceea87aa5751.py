# EVOLVE-BLOCK-START
import numpy as np
import time
import warnings
from scipy.linalg import solve
from scipy.special import chebyshev_t, chebyshev_u

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 280.0  # safety margin for 330s kill

    n, d = X_train.shape
    y_mean = np.mean(y_train)
    y_var = np.var(y_train)
    if y_var < 1e-12:
        return lambda X: np.full(len(X), y_mean)

    def check_time():
        return time.time() < deadline

    # Feature normalization for numerical stability
    # Map inputs to [-1, 1] which is the natural domain for Chebyshev polynomials
    X_min = X_train.min(axis=0)
    X_max = X_train.max(axis=0)
    X_range = X_max - X_min
    X_range[X_range < 1e-10] = 1.0
    
    # Map to [-1, 1]
    Xn_train = 2.0 * (X_train - X_min) / X_range - 1.0
    Xn_train = np.clip(Xn_train, -1.0, 1.0)

    y_std = np.std(y_train)
    if y_std < 1e-10:
        y_std = 1.0
    yn = (y_train - y_mean) / y_std

    def build_basis(Xn, d, max_cheb_deg=6, max_fourier_freq=5):
        """Build basis matrix with Chebyshev and Fourier terms."""
        terms = [np.ones(len(Xn))]
        
        # Univariate terms for each dimension
        for j in range(d):
            xj = Xn[:, j]
            
            # Chebyshev polynomials of the first kind
            for k in range(1, max_cheb_deg + 1):
                terms.append(chebyshev_t(k)(xj))
            
            # Fourier series (sin and cos)
            for k in range(1, max_fourier_freq + 1):
                terms.append(np.sin(k * np.pi * xj))
                terms.append(np.cos(k * np.pi * xj))
            
            # Smooth non-linearities
            terms.append(np.tanh(xj))
            terms.append(np.exp(xj))
            terms.append(np.exp(-xj))
            terms.append(np.sqrt(np.abs(xj)) * np.sign(xj))

        # Pairwise interactions
        if d >= 2:
            for i in range(d):
                for j in range(i + 1, d):
                    xi = Xn[:, i]
                    xj = Xn[:, j]
                    
                    # Product of base terms
                    terms.append(xi * xj)
                    
                    # Product of Chebyshev terms (low degree)
                    terms.append(chebyshev_t(1)(xi) * chebyshev_t(1)(xj))
                    terms.append(chebyshev_t(2)(xi) * chebyshev_t(1)(xj))
                    terms.append(chebyshev_t(1)(xi) * chebyshev_t(2)(xj))
                    
                    # Interaction in Fourier space
                    terms.append(np.sin(np.pi * (xi + xj)))
                    terms.append(np.cos(np.pi * (xi + xj)))

        # Triple interactions for d == 3
        if d == 3:
            x0, x1, x2 = Xn[:, 0], Xn[:, 1], Xn[:, 2]
            terms.append(x0 * x1 * x2)
            terms.append(chebyshev_t(1)(x0) * chebyshev_t(1)(x1) * chebyshev_t(1)(x2))

        F = np.column_stack(terms)
        # Ensure finite values
        F = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)
        return F

    # Build the basis
    F_train = build_basis(Xn_train, d)
    n_terms = F_train.shape[1]

    best_predict = None
    best_mse = np.inf

    def update_best(model):
        nonlocal best_predict, best_mse
        if model is None:
            return
        try:
            preds = model(X_train)
            preds = np.asarray(preds, dtype=float)
            if preds.shape != y_train.shape:
                return
            if np.any(np.isnan(preds)) or np.any(np.isinf(preds)):
                return
            mse = np.mean((preds - y_train) ** 2)
            if np.isnan(mse) or np.isinf(mse):
                return
            if mse < best_mse:
                best_mse = mse
                best_predict = model
        except Exception:
            pass

    # Strategy 1: Ridge Regression on the fixed basis
    # Try a range of regularization strengths
    for lam in [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0]:
        if not check_time():
            break
        try:
            # Add small diagonal to ensure invertibility
            A = F_train.T @ F_train + lam * np.eye(n_terms)
            b = F_train.T @ yn
            
            # Solve for weights
            w = solve(A, b)
            
            # Check for non-finite weights
            if np.any(np.isnan(w)) or np.any(np.isinf(w)):
                continue

            def predict_ridge(X, w, y_mean, y_std, X_min, X_range, d):
                # Normalize X to [-1, 1]
                Xn = 2.0 * (X - X_min) / X_range - 1.0
                Xn = np.clip(Xn, -1.0, 1.0)
                F_test = build_basis(Xn, d)
                return y_mean + y_std * (F_test @ w)
            
            update_best(lambda X: predict_ridge(X, w, y_mean, y_std, X_min, X_range, d))
        except Exception:
            pass

    # Fallback
    if best_predict is None:
        return lambda X: np.full(len(X), y_mean)

    return best_predict

# EVOLVE-BLOCK-END
