"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.interpolate import Rbf


def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 280  # Safety margin before 300s limit
    
    X = X_train.astype(np.float64)
    y = y_train.astype(np.float64)
    n, d = X.shape
    
    # For 1D case, use the single input dimension
    if d == 1:
        x = X[:, 0].copy()
    else:
        x = X  # fallback for multi-dim (shouldn't happen per task spec)
    
    # Sort by x for better RBF behavior
    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]
    
    best_predict = None
    best_score = np.inf
    
    # Strategy 1: RBF with various bandwidths
    # Try a range of epsilon values (inverse bandwidth)
    x_range = x_sorted.max() - x_sorted.min()
    if x_range < 1e-10:
        x_range = 1.0
    
    candidate_epsilons = []
    # Fine grid around optimal range
    for exp in np.arange(-2, 4, 0.25):
        eps = 10**exp * (x_range / 20)
        if eps > 0:
            candidate_epsilons.append(eps)
    # Also add some fixed reasonable values
    for factor in [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]:
        eps = factor * x_range / 10
        if eps > 0:
            candidate_epsilons.append(eps)
    
    candidate_epsilons = sorted(set(candidate_epsilons))
    
    # Use a subset if too many
    if len(candidate_epsilons) > 50:
        indices = np.linspace(0, len(candidate_epsilons)-1, 50).astype(int)
        candidate_epsilons = [candidate_epsilons[i] for i in indices]
    
    # Evaluate RBF candidates using k-fold cross-validation on training data
    n_folds = 5
    fold_size = n // n_folds
    folds = np.array_split(np.arange(n), n_folds)
    
    results = []
    for eps in candidate_epsilons:
        cv_errors = []
        for k in range(n_folds):
            if time.time() > deadline - 30:
                break
            val_idx = folds[k]
            train_idx = np.concatenate([folds[i] for i in range(n_folds) if i != k])
            
            x_train_cv = x_sorted[train_idx]
            y_train_cv = y_sorted[train_idx]
            x_val_cv = x_sorted[val_idx]
            y_val_cv = y_sorted[val_idx]
            
            try:
                # Use RBF interpolation with regularization
                # scipy Rbf with function='gaussian' and epsilon=1/sqrt(2)*bandwidth
                rbf = Rbf(x_train_cv, y_train_cv, function='gaussian', epsilon=eps, smooth=0.1)
                pred = rbf(x_val_cv)
                if np.all(np.isfinite(pred)):
                    mse = np.mean((pred - y_val_cv)**2)
                    cv_errors.append(mse)
            except Exception:
                continue
        
        if cv_errors:
            mean_cv_err = np.mean(cv_errors)
            results.append((mean_cv_err, eps))
        
        if time.time() > deadline - 20:
            break
    
    # Sort by CV error
    results.sort(key=lambda x: x[0])
    
    # Try top few RBF models and pick best
    for cv_err, eps in results[:5]:
        if time.time() > deadline - 10:
            break
        try:
            rbf = Rbf(x_sorted, y_sorted, function='gaussian', epsilon=eps, smooth=0.1)
            # Verify on full training set
            pred_train = rbf(x_sorted)
            if np.all(np.isfinite(pred_train)):
                train_mse = np.mean((pred_train - y_sorted)**2)
                var_y = np.var(y_sorted)
                if var_y > 1e-15:
                    norm_mse = train_mse / var_y
                    if norm_mse < best_score:
                        best_score = norm_mse
                        best_predict = lambda X, r=rbf: r(X[:, 0]) if X.shape[1] == 1 else r(X.ravel())
        except Exception:
            continue
    
    # Strategy 2: Polynomial regression as fallback
    if time.time() < deadline - 5:
        for degree in [2, 3, 4, 5, 6, 7, 8]:
            if time.time() > deadline - 5:
                break
            try:
                # Use numpy polyfit
                coeffs = np.polyfit(x, y, degree)
                def poly_predict(X, c=coeffs):
                    x_test = X[:, 0] if X.shape[1] == 1 else X.ravel()
                    return np.polyval(c, x_test)
                
                pred_train = poly_predict(X)
                if np.all(np.isfinite(pred_train)):
                    mse = np.mean((pred_train - y)**2)
                    var_y = np.var(y)
                    if var_y > 1e-15:
                        norm_mse = mse / var_y
                        if norm_mse < best_score:
                            best_score = norm_mse
                            best_predict = poly_predict
            except Exception:
                continue
    
    # Final fallback: constant predictor (mean)
    if best_predict is None:
        mean_y = np.mean(y)
        best_predict = lambda X: np.full(X.shape[0], mean_y)
    
    return best_predict
# EVOLVE-BLOCK-END
