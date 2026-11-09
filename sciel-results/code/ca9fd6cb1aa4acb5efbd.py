"""Symbolic regression via greedy feature selection with cross-validation."""
import time
import numpy as np
from scipy.optimize import curve_fit

# EVOLVE-BLOCK-START
import numpy as np


def fit(X_train, y_train):
    """
    Fit a regression model using greedy feature selection with CV.
    Returns a predict callable.
    """
    start_time = time.time()
    deadline = start_time + 280  # safety margin before 330s

    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    n, d = X.shape

    # If d > 1, we still handle it, but likely d=1
    # Build candidate basis functions
    def build_basis(X):
        """Return a list of (name, vector) candidate features."""
        feats = []
        x = X[:, 0] if X.shape[1] >= 1 else X.ravel()
        # For multi-dim, we'd need to handle differently, but assume 1D
        if X.shape[1] > 1:
            # Just use polynomial in all dims for now
            for i in range(X.shape[1]):
                for p in range(1, 6):
                    feats.append((f"x{i}^{p}", X[:, i] ** p))
            for i in range(X.shape[1]):
                for j in range(i, X.shape[1]):
                    feats.append((f"x{i}*x{j}", X[:, i] * X[:, j]))
        else:
            x = X[:, 0]
            # Polynomials
            for p in range(1, 8):
                feats.append((f"x^{p}", x ** p))
            # Trig
            for k in [1, 2, 3]:
                feats.append((f"sin({k}x)", np.sin(k * x)))
                feats.append((f"cos({k}x)", np.cos(k * x)))
            # Exp
            for a in [-1, -0.5, 0.5, 1]:
                with np.errstate(over='ignore', invalid='ignore'):
                    v = np.exp(a * x)
                    v = np.where(np.isfinite(v), v, 0.0)
                feats.append((f"exp({a}x)", v))
            # Log (only where x > 0)
            mask = x > 0
            if mask.sum() > 10:
                with np.errstate(divide='ignore', invalid='ignore'):
                    v = np.zeros_like(x)
                    v[mask] = np.log(x[mask])
                    v = np.where(np.isfinite(v), v, 0.0)
                feats.append(("log(x)", v))
            # Reciprocal
            with np.errstate(divide='ignore', invalid='ignore'):
                v = np.where(np.abs(x) > 1e-10, 1.0 / x, 0.0)
                v = np.where(np.isfinite(v), v, 0.0)
            feats.append(("1/x", v))
            # Sqrt
            mask = x >= 0
            if mask.sum() > 10:
                v = np.zeros_like(x)
                v[mask] = np.sqrt(x[mask])
                feats.append(("sqrt(x)", v))
            # Tanh
            feats.append(("tanh(x)", np.tanh(x)))
            # x*exp(-x^2) type
            feats.append(("x*exp(-x^2)", x * np.exp(-x ** 2)))
            feats.append(("x^2*exp(-x^2)", x ** 2 * np.exp(-x ** 2)))
        return feats

    basis = build_basis(X)
    
    # Add constant
    all_feats = [("1", np.ones(n))] + basis
    
    # Remove non-finite or constant features
    def is_good_feat(v):
        if not np.all(np.isfinite(v)):
            return False
        if np.std(v) < 1e-12:
            return False
        return True
    
    good_feats = [(name, v) for name, v in all_feats if is_good_feat(v)]
    
    # Cross-validation function
    def cv_score(feature_indices, n_folds=5):
        """Return mean CV MSE for given feature indices."""
        if len(feature_indices) == 0:
            return np.inf
        Xf = np.column_stack([good_feats[i][1] for i in feature_indices])
        # Check for rank deficiency
        if Xf.shape[1] > n:
            return np.inf
        try:
            # Simple k-fold
            indices = np.arange(n)
            np.random.seed(42)
            np.random.shuffle(indices)
            folds = np.array_split(indices, n_folds)
            mses = []
            for f in range(n_folds):
                val_idx = folds[f]
                tr_idx = np.concatenate([folds[i] for i in range(n_folds) if i != f])
                Xtr = Xf[tr_idx]
                ytr = y[tr_idx]
                Xval = Xf[val_idx]
                yval = y[val_idx]
                # Regularized least squares
                lam = 1e-6 * Xtr.shape[1]
                try:
                    A = Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1])
                    b = Xtr.T @ ytr
                    w = np.linalg.solve(A, b)
                except np.linalg.LinAlgError:
                    w, *_ = np.linalg.lstsq(Xtr, ytr, rcond=None)
                pred = Xval @ w
                mse = np.mean((pred - yval) ** 2)
                mses.append(mse)
            return np.mean(mses)
        except Exception:
            return np.inf

    # Greedy forward selection
    selected = []
    remaining = list(range(len(good_feats)))
    best_cv = np.inf
    
    # Always start with constant
    selected.append(0)
    remaining.remove(0)
    best_cv = cv_score(selected)
    
    max_features = min(15, len(good_feats))
    
    while len(selected) < max_features:
        if time.time() > deadline:
            break
        best_idx = None
        best_cv_new = best_cv
        for idx in remaining:
            if time.time() > deadline:
                break
            trial = selected + [idx]
            cv = cv_score(trial)
            if cv < best_cv_new - 1e-9:
                best_cv_new = cv
                best_idx = idx
        if best_idx is None:
            break
        selected.append(best_idx)
        remaining.remove(best_idx)
        best_cv = best_cv_new
    
    # Final fit with selected features
    Xf = np.column_stack([good_feats[i][1] for i in selected])
    lam = 1e-6 * Xf.shape[1]
    try:
        A = Xf.T @ Xf + lam * np.eye(Xf.shape[1])
        b = Xf.T @ y
        w = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        w, *_ = np.linalg.lstsq(Xf, y, rcond=None)
    
    # Also try a few fixed models as fallbacks
    fallback_models = []
    
    # Degree 2 polynomial
    if X.shape[1] == 1:
        x = X[:, 0]
        poly2 = np.column_stack([np.ones(n), x, x**2])
        try:
            w2, *_ = np.linalg.lstsq(poly2, y, rcond=None)
            fallback_models.append(("poly2", lambda X: np.column_stack([np.ones(len(X)), X[:,0], X[:,0]**2]) @ w2))
        except:
            pass
        # Degree 3
        poly3 = np.column_stack([np.ones(n), x, x**2, x**3])
        try:
            w3, *_ = np.linalg.lstsq(poly3, y, rcond=None)
            fallback_models.append(("poly3", lambda X: np.column_stack([np.ones(len(X)), X[:,0], X[:,0]**2, X[:,0]**3]) @ w3))
        except:
            pass
    
    # Evaluate all candidates on training data and pick best
    def train_mse(pred_func):
        try:
            pred = pred_func(X)
            if not np.all(np.isfinite(pred)):
                return np.inf
            return np.mean((pred - y) ** 2)
        except:
            return np.inf
    
    # The greedy model
    def greedy_predict(X_new):
        Xf_new = np.column_stack([good_feats[i][1] for i in selected])
        return Xf_new @ w
    
    candidates = [("greedy", greedy_predict)]
    for name, pred_func in fallback_models:
        candidates.append((name, pred_func))
    
    best_name = "greedy"
    best_score = train_mse(greedy_predict)
    for name, pred_func in fallback_models:
        s = train_mse(pred_func)
        if s < best_score:
            best_score = s
            best_name = name
    
    # But prefer CV-based selection: use CV score for greedy, train MSE for fallbacks
    # Actually, let's use CV score for all if possible
    # For simplicity, pick the one with best training MSE but ensure it's not overfitting
    # Use a simple rule: if greedy has <= 5 features, use it; else compare
    
    # Final decision: use the model with best CV if available, else best train MSE
    # For fallbacks, we don't have easy CV, so use train MSE as proxy
    # But greedy has CV score
    if best_name == "greedy":
        final_predict = greedy_predict
    else:
        final_predict = [p for n, p in candidates if n == best_name][0]
    
    # Ensure predict returns finite values
    def safe_predict(X_new):
        X_new = np.asarray(X_new, dtype=np.float64)
        try:
            pred = final_predict(X_new)
            pred = np.asarray(pred, dtype=np.float64).ravel()
            pred = np.where(np.isfinite(pred), pred, 0.0)
            return pred
        except Exception:
            return np.zeros(len(X_new))
    
    return safe_predict
# EVOLVE-BLOCK-END
