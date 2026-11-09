"""Symbolic regression via adaptive basis search and polynomial/Fourier fitting."""
import time
import numpy as np
from scipy.optimize import minimize_scalar

# EVOLVE-BLOCK-START
import numpy as np
import time

def fit(X_train, y_train):
    """
    Fit a regression model to X_train, y_train and return a predict callable.
    Uses an adaptive search over diverse basis functions (polynomial, Fourier,
    rational-like, etc.) and picks the best by training MSE.
    """
    start_time = time.time()
    deadline = start_time + 280.0  # 50s safety margin
    
    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    
    n, d = X.shape
    
    def time_left():
        return deadline - time.time()
    
    def safe_predict(X_test, w, basis_fn):
        """Safe prediction wrapper that returns finite values."""
        try:
            F = basis_fn(X_test)
            pred = F @ w
            # Replace non-finite with 0
            pred = np.where(np.isfinite(pred), pred, 0.0)
            return pred
        except Exception:
            return np.zeros(len(X_test))
    
    def mse(pred, y_true):
        return np.mean((pred - y_true) ** 2)
    
    # --- Basis function generators ---
    
    def poly_basis(X, degree):
        """Polynomial basis up to given degree (for 1D input)."""
        x = X[:, 0]
        cols = [np.ones_like(x)]
        for k in range(1, degree + 1):
            cols.append(x ** k)
        return np.column_stack(cols)
    
    def fourier_basis(X, n_terms):
        """Fourier basis with harmonics."""
        x = X[:, 0]
        # Normalize x to [-1, 1] roughly
        x_min, x_max = x.min(), x.max()
        x_range = x_max - x_min + 1e-10
        xn = (x - (x_min + x_max) / 2) / (x_range / 2)
        cols = [np.ones_like(xn)]
        for k in range(1, n_terms + 1):
            cols.append(np.sin(k * np.pi * xn))
            cols.append(np.cos(k * np.pi * xn))
        return np.column_stack(cols)
    
    def mixed_basis(X, poly_deg, fourier_n):
        """Combined polynomial and Fourier basis."""
        F_poly = poly_basis(X, poly_deg)
        F_four = fourier_basis(X, fourier_n)
        return np.column_stack([F_poly, F_four])
    
    def exp_basis(X, n_exp):
        """Exponential basis: exp(k*x) for small k."""
        x = X[:, 0]
        cols = [np.ones_like(x)]
        for k in range(1, n_exp + 1):
            cols.append(np.exp(k * 0.5 * x))
            cols.append(np.exp(-k * 0.5 * x))
        return np.column_stack(cols)
    
    def rational_basis(X, n_num, n_den):
        """Rational-like basis using 1/(1+x^2) type terms."""
        x = X[:, 0]
        cols = [np.ones_like(x)]
        cols.append(x)
        for k in range(1, n_num + 1):
            cols.append(x ** k)
        for k in range(1, n_den + 1):
            cols.append(1.0 / (1.0 + x ** (2 * k)))
        return np.column_stack(cols)
    
    def tanh_basis(X, n_terms):
        """Tanh-based basis."""
        x = X[:, 0]
        cols = [np.ones_like(x)]
        for k in range(1, n_terms + 1):
            cols.append(np.tanh(k * 0.5 * x))
        return np.column_stack(cols)
    
    # --- Candidate models to try ---
    
    candidates = []
    
    # Polynomial degrees 1-8
    for deg in range(1, 9):
        candidates.append(('poly', deg, None))
    
    # Fourier with various numbers of terms
    for n in range(2, 12):
        candidates.append(('fourier', n, None))
    
    # Mixed polynomial + Fourier
    for pd in range(1, 5):
        for fn in range(1, 8):
            candidates.append(('mixed', (pd, fn), None))
    
    # Exponential basis
    for ne in range(1, 6):
        candidates.append(('exp', ne, None))
    
    # Rational-like basis
    for nn in range(1, 5):
        for nd in range(1, 4):
            candidates.append(('rational', (nn, nd), None))
    
    # Tanh basis
    for nt in range(1, 8):
        candidates.append(('tanh', nt, None))
    
    best_model = None
    best_mse = np.inf
    
    for name, params, _ in candidates:
        if time_left() < 5.0:
            break
        
        try:
            if name == 'poly':
                F = poly_basis(X, params)
            elif name == 'fourier':
                F = fourier_basis(X, params)
            elif name == 'mixed':
                F = mixed_basis(X, params[0], params[1])
            elif name == 'exp':
                F = exp_basis(X, params)
            elif name == 'rational':
                F = rational_basis(X, params[0], params[1])
            elif name == 'tanh':
                F = tanh_basis(X, params)
            else:
                continue
            
            # Check for ill-conditioning
            if F.shape[1] > n:
                # Use ridge regression for over-parameterized models
                alpha = 1e-6
                A = F.T @ F + alpha * np.eye(F.shape[1])
                b = F.T @ y
                w = np.linalg.solve(A, b)
            else:
                w, residuals, rank, sv = np.linalg.lstsq(F, y, rcond=None)
            
            pred = F @ w
            if not np.all(np.isfinite(pred)):
                continue
            
            cur_mse = mse(pred, y)
            
            if cur_mse < best_mse:
                best_mse = cur_mse
                # Store the basis type and params to reconstruct later
                if name == 'poly':
                    best_model = ('poly', params)
                elif name == 'fourier':
                    best_model = ('fourier', params)
                elif name == 'mixed':
                    best_model = ('mixed', params)
                elif name == 'exp':
                    best_model = ('exp', params)
                elif name == 'rational':
                    best_model = ('rational', params)
                elif name == 'tanh':
                    best_model = ('tanh', params)
                
        except Exception:
            continue
    
    # If no model found, fall back to degree-2 polynomial
    if best_model is None:
        best_model = ('poly', 2)
    
    # Refine: try a few more specific models around the best
    if time_left() > 10.0:
        name, params = best_model
        if name == 'poly':
            for deg in range(max(1, params - 2), min(15, params + 3)):
                if deg == params:
                    continue
                try:
                    F = poly_basis(X, deg)
                    if F.shape[1] > n:
                        alpha = 1e-6
                        A = F.T @ F + alpha * np.eye(F.shape[1])
                        b = F.T @ y
                        w = np.linalg.solve(A, b)
                    else:
                        w, *_ = np.linalg.lstsq(F, y, rcond=None)
                    pred = F @ w
                    if np.all(np.isfinite(pred)):
                        cur_mse = mse(pred, y)
                        if cur_mse < best_mse:
                            best_mse = cur_mse
                            best_model = ('poly', deg)
                except Exception:
                    pass
        elif name == 'fourier':
            for n_terms in range(max(1, params - 3), min(20, params + 4)):
                if n_terms == params:
                    continue
                try:
                    F = fourier_basis(X, n_terms)
                    if F.shape[1] > n:
                        alpha = 1e-6
                        A = F.T @ F + alpha * np.eye(F.shape[1])
                        b = F.T @ y
                        w = np.linalg.solve(A, b)
                    else:
                        w, *_ = np.linalg.lstsq(F, y, rcond=None)
                    pred = F @ w
                    if np.all(np.isfinite(pred)):
                        cur_mse = mse(pred, y)
                        if cur_mse < best_mse:
                            best_mse = cur_mse
                            best_model = ('fourier', n_terms)
                except Exception:
                    pass
        elif name == 'mixed':
            for pd in range(max(1, params[0] - 1), min(6, params[0] + 2)):
                for fn in range(max(1, params[1] - 1), min(10, params[1] + 2)):
                    if (pd, fn) == params:
                        continue
                    try:
                        F = mixed_basis(X, pd, fn)
                        if F.shape[1] > n:
                            alpha = 1e-6
                            A = F.T @ F + alpha * np.eye(F.shape[1])
                            b = F.T @ y
                            w = np.linalg.solve(A, b)
                        else:
                            w, *_ = np.linalg.lstsq(F, y, rcond=None)
                        pred = F @ w
                        if np.all(np.isfinite(pred)):
                            cur_mse = mse(pred, y)
                            if cur_mse < best_mse:
                                best_mse = cur_mse
                                best_model = ('mixed', (pd, fn))
                    except Exception:
                        pass
        elif name == 'exp':
            for ne in range(max(1, params - 2), min(10, params + 3)):
                if ne == params:
                    continue
                try:
                    F = exp_basis(X, ne)
                    if F.shape[1] > n:
                        alpha = 1e-6
                        A = F.T @ F + alpha * np.eye(F.shape[1])
                        b = F.T @ y
                        w = np.linalg.solve(A, b)
                    else:
                        w, *_ = np.linalg.lstsq(F, y, rcond=None)
                    pred = F @ w
                    if np.all(np.isfinite(pred)):
                        cur_mse = mse(pred, y)
                        if cur_mse < best_mse:
                            best_mse = cur_mse
                            best_model = ('exp', ne)
                except Exception:
                    pass
        elif name == 'rational':
            for nn in range(max(1, params[0] - 1), min(7, params[0] + 2)):
                for nd in range(max(1, params[1] - 1), min(5, params[1] + 2)):
                    if (nn, nd) == params:
                        continue
                    try:
                        F = rational_basis(X, nn, nd)
                        if F.shape[1] > n:
                            alpha = 1e-6
                            A = F.T @ F + alpha * np.eye(F.shape[1])
                            b = F.T @ y
                            w = np.linalg.solve(A, b)
                        else:
                            w, *_ = np.linalg.lstsq(F, y, rcond=None)
                        pred = F @ w
                        if np.all(np.isfinite(pred)):
                            cur_mse = mse(pred, y)
                            if cur_mse < best_mse:
                                best_mse = cur_mse
                                best_model = ('rational', (nn, nd))
                    except Exception:
                        pass
        elif name == 'tanh':
            for nt in range(max(1, params - 2), min(12, params + 3)):
                if nt == params:
                    continue
                try:
                    F = tanh_basis(X, nt)
                    if F.shape[1] > n:
                        alpha = 1e-6
                        A = F.T @ F + alpha * np.eye(F.shape[1])
                        b = F.T @ y
                        w = np.linalg.solve(A, b)
                    else:
                        w, *_ = np.linalg.lstsq(F, y, rcond=None)
                    pred = F @ w
                    if np.all(np.isfinite(pred)):
                        cur_mse = mse(pred, y)
                        if cur_mse < best_mse:
                            best_mse = cur_mse
                            best_model = ('tanh', nt)
                except Exception:
                    pass
    
    # Final fit with the best model
    name, params = best_model
    if name == 'poly':
        F_train = poly_basis(X, params)
    elif name == 'fourier':
        F_train = fourier_basis(X, params)
    elif name == 'mixed':
        F_train = mixed_basis(X, params[0], params[1])
    elif name == 'exp':
        F_train = exp_basis(X, params)
    elif name == 'rational':
        F_train = rational_basis(X, params[0], params[1])
    elif name == 'tanh':
        F_train = tanh_basis(X, params)
    else:
        F_train = poly_basis(X, 2)
    
    if F_train.shape[1] > n:
        alpha = 1e-6
        A = F_train.T @ F_train + alpha * np.eye(F_train.shape[1])
        b = F_train.T @ y
        w_final = np.linalg.solve(A, b)
    else:
        w_final, *_ = np.linalg.lstsq(F_train, y, rcond=None)
    
    # Define the predict function
    def predict(X_test):
        X_test = np.asarray(X_test, dtype=np.float64)
        if name == 'poly':
            F_test = poly_basis(X_test, params)
        elif name == 'fourier':
            F_test = fourier_basis(X_test, params)
        elif name == 'mixed':
            F_test = mixed_basis(X_test, params[0], params[1])
        elif name == 'exp':
            F_test = exp_basis(X_test, params)
        elif name == 'rational':
            F_test = rational_basis(X_test, params[0], params[1])
        elif name == 'tanh':
            F_test = tanh_basis(X_test, params)
        else:
            F_test = poly_basis(X_test, 2)
        
        try:
            pred = F_test @ w_final
            pred = np.where(np.isfinite(pred), pred, 0.0)
            return pred
        except Exception:
            return np.zeros(len(X_test))
    
    return predict
# EVOLVE-BLOCK-END
