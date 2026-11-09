"""Symbolic regression over hidden equations using GP + local refinement."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')


def _rbf_kernel(X1, X2, length_scale, variance):
    """RBF kernel between two sets of points."""
    # X1: (n, d), X2: (m, d)
    d1 = np.sum(X1 ** 2, axis=1, keepdims=True)
    d2 = np.sum(X2 ** 2, axis=1, keepdims=True).T
    sq_dists = d1 + d2 - 2 * X1 @ X2.T
    sq_dists = np.maximum(sq_dists, 0)
    return variance * np.exp(-sq_dists / (2 * length_scale ** 2))


def _fit_gp(X_train, y_train, n_restarts=5):
    """Fit a Gaussian Process with RBF kernel using MLE."""
    n, d = X_train.shape
    y_mean = np.mean(y_train)
    y_std = np.std(y_train)
    if y_std < 1e-10:
        y_std = 1.0
    y_centered = (y_train - y_mean) / y_std
    
    # Normalize inputs to [0, 1] for better kernel scaling
    X_min = X_train.min(axis=0, keepdims=True)
    X_max = X_train.max(axis=0, keepdims=True)
    X_range = X_max - X_min
    X_range[X_range < 1e-10] = 1.0
    X_norm = (X_train - X_min) / X_range
    
    def log_marginal_likelihood(params):
        log_ls, log_var, log_noise = params
        ls = np.exp(log_ls)
        var = np.exp(log_var)
        noise = np.exp(log_noise)
        
        K = _rbf_kernel(X_norm, X_norm, ls, var)
        K += noise * np.eye(n)
        
        try:
            L = np.linalg.cholesky(K)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_centered))
            lml = -0.5 * y_centered @ alpha - np.sum(np.log(np.diag(L))) - 0.5 * n * np.log(2 * np.pi)
            return -lml
        except np.linalg.LinAlgError:
            return 1e10
    
    best_params = None
    best_score = np.inf
    
    for _ in range(n_restarts):
        # Random initialization in log-space
        log_ls = np.random.uniform(-2, 1)
        log_var = np.random.uniform(-2, 2)
        log_noise = np.random.uniform(-8, -2)
        x0 = np.array([log_ls, log_var, log_noise])
        
        res = minimize(log_marginal_likelihood, x0, method='Nelder-Mead', 
                       options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-4})
        
        if res.fun < best_score:
            best_score = res.fun
            best_params = res.x
    
    if best_params is None:
        best_params = np.array([0, 0, -5])
    
    log_ls, log_var, log_noise = best_params
    ls = np.exp(log_ls)
    var = np.exp(log_var)
    noise = np.exp(log_noise)
    
    K = _rbf_kernel(X_norm, X_norm, ls, var)
    K += noise * np.eye(n)
    
    try:
        L = np.linalg.cholesky(K)
    except np.linalg.LinAlgError:
        K += 1e-6 * np.eye(n)
        L = np.linalg.cholesky(K)
    
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_centered))
    
    def predict_gp(X):
        X_min_ = X_train.min(axis=0, keepdims=True)
        X_max_ = X_train.max(axis=0, keepdims=True)
        X_range_ = X_max_ - X_min_
        X_range_[X_range_ < 1e-10] = 1.0
        X_norm_ = (X - X_min_) / X_range_
        
        K_s = _rbf_kernel(X_norm_, X_norm, ls, var)
        preds = K_s @ alpha * y_std + y_mean
        return preds
    
    return predict_gp, (ls, var, noise, y_mean, y_std, X_min, X_range)


def _fit_polynomial(X_train, y_train, degree=4):
    """Fit a polynomial regression with feature selection."""
    n, d = X_train.shape
    
    # Generate polynomial features
    def gen_features(X, deg):
        features = [np.ones(len(X))]
        # Linear terms
        for j in range(d):
            features.append(X[:, j])
        # Interaction and power terms up to degree
        for total_deg in range(2, deg + 1):
            for i in range(d):
                for j in range(i, d):
                    if total_deg == 2:
                        if i == j:
                            features.append(X[:, i] ** 2)
                        else:
                            features.append(X[:, i] * X[:, j])
                    else:
                        # Higher order: simple powers and products
                        # Keep it manageable
                        if i == j:
                            features.append(X[:, i] ** total_deg)
                        elif total_deg == 3:
                            features.append(X[:, i] * X[:, j] * X[:, max(i, j)])
                        # Skip very high order for stability
        return np.column_stack(features)
    
    # Use degree 3 to keep feature count reasonable
    F = gen_features(X_train, 3)
    
    # Ridge regression for stability
    alpha_ridge = 1e-3
    w = np.linalg.lstsq(F + alpha_ridge * np.eye(F.shape[1]), y_train, rcond=None)[0]
    
    def predict_poly(X):
        F_test = gen_features(X, 3)
        return F_test @ w
    
    return predict_poly


def _fit_local_linear(X_train, y_train, k=10):
    """Local linear regression."""
    n, d = X_train.shape
    
    def predict_ll(X):
        preds = np.zeros(len(X))
        for i in range(len(X)):
            dists = np.sum((X_train - X[i:i+1]) ** 2, axis=1)
            idx = np.argsort(dists)[:k]
            X_local = X_train[idx]
            y_local = y_train[idx]
            
            # Weighted linear regression
            weights = 1.0 / (dists[idx] + 1e-10)
            X_aug = np.column_stack([np.ones(k), X_local])
            W = np.diag(weights)
            XtW = X_aug.T @ W
            XtWX = XtW @ X_aug
            XtWy = XtW @ y_local
            try:
                params = np.linalg.solve(XtWX, XtWy)
            except np.linalg.LinAlgError:
                params = np.linalg.lstsq(XtWX, XtWy, rcond=None)[0]
            X_aug_test = np.column_stack([np.ones(1), X[i:i+1]])
            preds[i] = X_aug_test @ params
        return preds
    
    return predict_ll


def _ensemble_predict(predictors, weights=None):
    """Combine multiple predictors."""
    if weights is None:
        weights = [1.0 / len(predictors)] * len(predictors)
    
    def predict_ensemble(X):
        preds = np.zeros(len(X))
        for pred, w in zip(predictors, weights):
            preds += w * pred(X)
        return preds
    
    return predict_ensemble


def fit(X_train, y_train):
    """
    Fit multiple models and return the best one based on cross-validation.
    """
    n, d = X_train.shape
    
    # Cross-validation to select best model
    n_folds = 5
    fold_size = n // n_folds
    indices = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(indices)
    
    cv_scores = {'gp': [], 'poly': [], 'll': []}
    
    for fold in range(n_folds):
        val_idx = indices[fold * fold_size:(fold + 1) * fold_size]
        train_idx = np.concatenate([indices[:fold * fold_size], indices[(fold + 1) * fold_size:]])
        
        X_tr, y_tr = X_train[train_idx], y_train[train_idx]
        X_val, y_val = X_train[val_idx], y_train[val_idx]
        
        # GP
        try:
            pred_gp, _ = _fit_gp(X_tr, y_tr, n_restarts=3)
            mse_gp = np.mean((pred_gp(X_val) - y_val) ** 2)
            cv_scores['gp'].append(mse_gp)
        except:
            cv_scores['gp'].append(np.inf)
        
        # Polynomial
        try:
            pred_poly = _fit_polynomial(X_tr, y_tr, degree=3)
            mse_poly = np.mean((pred_poly(X_val) - y_val) ** 2)
            cv_scores['poly'].append(mse_poly)
        except:
            cv_scores['poly'].append(np.inf)
        
        # Local linear (only if n is small enough)
        if n <= 100:
            try:
                pred_ll = _fit_local_linear(X_tr, y_tr, k=min(10, len(X_tr) // 2))
                mse_ll = np.mean((pred_ll(X_val) - y_val) ** 2)
                cv_scores['ll'].append(mse_ll)
            except:
                cv_scores['ll'].append(np.inf)
        else:
            cv_scores['ll'].append(np.inf)
    
    mean_scores = {k: np.mean(v) for k, v in cv_scores.items()}
    
    # Select best model
    best_model = min(mean_scores, key=mean_scores.get)
    
    # Fit final model on all data
    if best_model == 'gp':
        try:
            predict, _ = _fit_gp(X_train, y_train, n_restarts=5)
            return predict
        except:
            pass
    
    if best_model == 'poly':
        try:
            return _fit_polynomial(X_train, y_train, degree=3)
        except:
            pass
    
    # Fallback: polynomial
    return _fit_polynomial(X_train, y_train, degree=3)
# EVOLVE-BLOCK-END
