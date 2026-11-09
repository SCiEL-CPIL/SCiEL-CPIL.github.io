"""Symbolic regression with adaptive forward selection and iterative basis expansion."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import warnings

warnings.filterwarnings('ignore')


def _safe_predict(predict_fn, X):
    """Ensure predictions are finite."""
    X = np.asarray(X, dtype=float)
    try:
        pred = predict_fn(X)
    except Exception:
        return np.zeros(len(np.asarray(X).reshape(-1)))
    pred = np.asarray(pred, dtype=float)
    pred = np.where(np.isfinite(pred), pred, 0.0)
    return pred


def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 285  # Safety margin for 330s limit

    X = np.asarray(X_train, dtype=float).reshape(-1, 1)
    y = np.asarray(y_train, dtype=float).flatten()
    n = len(y)

    if n < 2:
        mean_y = np.mean(y) if n > 0 else 0.0
        return lambda X: _safe_predict(lambda x: np.full(len(np.asarray(x).reshape(-1)), mean_y), X)

    x_vals = X[:, 0].copy()
    y_vals = y.copy()

    # Constant check
    if np.std(y_vals) < 1e-12:
        mean_y = np.mean(y_vals)
        return lambda X: _safe_predict(lambda x: np.full(len(np.asarray(x).reshape(-1)), mean_y), X)

    y_var = np.var(y_vals)
    if y_var < 1e-15:
        mean_y = np.mean(y_vals)
        return lambda X: _safe_predict(lambda x: np.full(len(np.asarray(x).reshape(-1)), mean_y), X)

    x_min, x_max = np.min(x_vals), np.max(x_vals)
    x_range = x_max - x_min
    if x_range < 1e-10:
        x_range = 1.0
    x_center = (x_min + x_max) / 2.0

    # 5-fold CV setup
    rng = np.random.RandomState(42)
    idx_all = np.arange(n)
    rng.shuffle(idx_all)
    folds = np.array_split(idx_all, 5)

    def cv_mse(basis_matrix, y_target):
        """Compute CV normalized MSE for a given basis matrix."""
        if time.time() > deadline - 3:
            return np.inf
        try:
            se = 0.0
            for tr, va in zip(range(5), folds):
                train_idx = np.concatenate([folds[i] for i in range(5) if i != tr])
                val_idx = folds[tr]
                A_tr = basis_matrix[train_idx]
                b_tr = y_target[train_idx]
                A_va = basis_matrix[val_idx]
                b_va = y_target[val_idx]
                # Ridge solve
                n_cols = A_tr.shape[1]
                try:
                    w = np.linalg.solve(A_tr.T @ A_tr + 1e-6 * np.eye(n_cols), A_tr.T @ b_tr)
                except np.linalg.LinAlgError:
                    w = np.linalg.lstsq(A_tr, b_tr, rcond=None)[0]
                pred = A_va @ w
                if not np.all(np.isfinite(pred)):
                    return np.inf
                se += np.sum((pred - b_va) ** 2)
            return se / (n * y_var)
        except Exception:
            return np.inf

    # Generate candidate basis functions
    def make_basis_funcs():
        """Generate a pool of candidate basis functions."""
        funcs = []
        # Constant
        funcs.append(('const', lambda x: np.ones_like(x, dtype=float)))
        # Linear
        funcs.append(('x', lambda x: x.copy()))
        # Polynomials (centered for better conditioning)
        for d in range(2, 10):
            funcs.append((f'c^{d}', lambda x, d=d: (x - x_center) ** d))
        # Trig
        for f in [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]:
            funcs.append((f'cos{f}', lambda x, f=f: np.cos(f * x)))
            funcs.append((f'sin{f}', lambda x, f=f: np.sin(f * x)))
        # Exp
        for b in [-2.0, -1.0, -0.5, -0.25, 0.25, 0.5, 1.0, 2.0]:
            funcs.append((f'exp{b}', lambda x, b=b: np.exp(b * x)))
        # RBFs at various centers and widths
        n_centers = 12
        centers = np.linspace(x_min, x_max, n_centers + 2)[1:-1]
        widths = [0.1 * x_range, 0.2 * x_range, 0.3 * x_range, 0.5 * x_range]
        for c in centers:
            for w in widths:
                funcs.append((f'rbf_{c:.3f}_{w:.3f}', lambda x, c=c, w=w: np.exp(-((x - c) ** 2) / (2 * w ** 2))))
        # Rational
        funcs.append(('1/(x+1)', lambda x: 1.0 / (x + 1.0 + 1e-10)))
        funcs.append(('1/(x-1)', lambda x: 1.0 / (x - 1.0 + 1e-10)))
        funcs.append(('1/(x^2+1)', lambda x: 1.0 / (x ** 2 + 1.0)))
        funcs.append(('1/(x^2+0.1)', lambda x: 1.0 / (x ** 2 + 0.1)))
        # Sqrt/abs
        funcs.append(('abs', lambda x: np.abs(x)))
        funcs.append(('x| x|', lambda x: x * np.abs(x)))
        funcs.append(('sqrt(x^2+1)', lambda x: np.sqrt(x ** 2 + 1.0)))
        if x_min > -0.1:
            funcs.append(('sqrt(x+0.1)', lambda x: np.sqrt(np.maximum(x + 0.1, 1e-10))))
        # Tanh
        for f in [0.25, 0.5, 1.0, 2.0, 4.0]:
            funcs.append((f'tanh{f}', lambda x, f=f: np.tanh(f * x)))
        # Log
        funcs.append(('log(x^2+1)', lambda x: np.log(x ** 2 + 1.0)))
        if x_min > -0.1:
            funcs.append(('log(x+0.1)', lambda x: np.log(np.maximum(x + 0.1, 1e-10))))
        return funcs

    candidate_pool = make_basis_funcs()
    
    # Pre-compute all basis values on training data
    n_cand = len(candidate_pool)
    B_all = np.zeros((n, n_cand))
    valid_cand = []
    for j, (name, func) in enumerate(candidate_pool):
        try:
            col = func(x_vals)
            col = np.asarray(col, dtype=float)
            if not np.all(np.isfinite(col)):
                col = np.where(np.isfinite(col), col, 0.0)
            if np.var(col) > 1e-10:
                B_all[:, j] = col
                valid_cand.append(j)
        except Exception:
            pass
    
    valid_cand = np.array(valid_cand)
    B_valid = B_all[:, valid_cand]
    n_valid = len(valid_cand)
    
    if n_valid == 0:
        mean_y = np.mean(y_vals)
        return lambda X: _safe_predict(lambda x: np.full(len(np.asarray(x).reshape(-1)), mean_y), X)

    # Forward selection with CV
    selected = [0]  # Start with constant
    best_cv = cv_mse(B_valid[:, selected], y_vals)
    
    # Try adding each candidate one at a time
    max_iter = min(25, n_valid)  # Limit number of selections
    for it in range(max_iter):
        if time.time() > deadline - 10:
            break
        best_add_idx = -1
        best_add_cv = best_cv
        for j in range(n_valid):
            if j in selected:
                continue
            trial_selected = selected + [j]
            cv = cv_mse(B_valid[:, trial_selected], y_vals)
            if cv < best_add_cv:
                best_add_cv = cv
                best_add_idx = j
        if best_add_idx >= 0:
            selected.append(best_add_idx)
            best_cv = best_add_cv
            # Early stop if CV is very good
            if best_cv < 1e-6:
                break
        else:
            break
    
    # Final model: train on all data with selected basis
    A_final = B_valid[:, selected]
    n_cols = A_final.shape[1]
    try:
        w_final = np.linalg.solve(A_final.T @ A_final + 1e-6 * np.eye(n_cols), A_final.T @ y_vals)
    except np.linalg.LinAlgError:
        w_final = np.linalg.lstsq(A_final, y_vals, rcond=None)[0]
    
    if not np.all(np.isfinite(w_final)):
        w_final = np.zeros(n_cols)
        mean_y = np.mean(y_vals)
        w_final[0] = mean_y  # Constant term

    # Also try a pure polynomial fit as alternative
    poly_best = None
    poly_best_cv = np.inf
    for deg in range(1, min(12, n)):
        if time.time() > deadline - 5:
            break
        try:
            # Build polynomial basis
            A_poly = np.zeros((n, deg + 1))
            for d in range(deg + 1):
                A_poly[:, d] = (x_vals - x_center) ** d
            cv = cv_mse(A_poly, y_vals)
            if cv < poly_best_cv:
                poly_best_cv = cv
                # Train final poly
                try:
                    w_poly = np.linalg.solve(A_poly.T @ A_poly + 1e-6 * np.eye(deg + 1), A_poly.T @ y_vals)
                except np.linalg.LinAlgError:
                    w_poly = np.linalg.lstsq(A_poly, y_vals, rcond=None)[0]
                if np.all(np.isfinite(w_poly)):
                    poly_best = (deg, w_poly, A_poly)
        except Exception:
            continue
    
    # Choose between forward-selected model and polynomial
    use_poly = (poly_best is not None and poly_best_cv < best_cv * 0.95)
    
    def predict(X_pred):
        X_pred = np.asarray(X_pred, dtype=float).reshape(-1, 1)[:, 0]
        n_pred = len(X_pred)
        try:
            if use_poly and poly_best is not None:
                deg, w_poly, _ = poly_best
                A_pred = np.zeros((n_pred, deg + 1))
                for d in range(deg + 1):
                    A_pred[:, d] = (X_pred - x_center) ** d
                pred = A_pred @ w_poly
            else:
                A_pred = np.zeros((n_pred, len(selected)))
                for k, idx in enumerate(selected):
                    name, func = candidate_pool[idx]
                    try:
                        col = func(X_pred)
                        col = np.asarray(col, dtype=float)
                        if not np.all(np.isfinite(col)):
                            col = np.where(np.isfinite(col), col, 0.0)
                        A_pred[:, k] = col
                    except Exception:
                        A_pred[:, k] = 0.0
                pred = A_pred @ w_final
            pred = np.asarray(pred, dtype=float)
            pred = np.where(np.isfinite(pred), pred, 0.0)
            return pred
        except Exception:
            return np.full(n_pred, np.mean(y_vals))

    return predict
# EVOLVE-BLOCK-END
