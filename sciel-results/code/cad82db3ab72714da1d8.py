"""Symbolic regression via multi-model oracle selection with LOO-CV."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from itertools import product

def fit(X_train, y_train):
    start_time = time.time()
    DEADLINE = 325.0  # Safety margin for 330s limit
    
    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    
    n, d = X.shape
    
    # Normalize inputs to [0, 1]
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    X_range = X_max - X_min
    X_range[X_range == 0] = 1.0
    X_norm = (X - X_min) / X_range
    
    # Normalize target for numerical stability
    y_mean = np.mean(y)
    y_std = np.std(y)
    if y_std == 0:
        y_std = 1.0
    y_norm = (y - y_mean) / y_std

    models = []

    # --- Model Family 1: Polynomial Kernel Ridge Regression ---
    poly_configs = [
        (3, 1.0), (5, 1.0), (3, 2.0), (5, 2.0), (7, 1.0)
    ]
    
    for deg, coef in poly_configs:
        for alpha in [1e-4, 1e-3, 1e-2, 1e-1]:
            def make_poly_fitter(deg_val, coef_val, alpha_val):
                def fitter(X_tr, y_tr, X_val):
                    K_tr = (X_tr @ X_tr.T + coef_val) ** deg_val
                    w = np.linalg.solve(K_tr + alpha_val * np.eye(len(y_tr)), y_tr)
                    K_val = (X_val @ X_tr.T + coef_val) ** deg_val
                    return K_val @ w
                return fitter
            
            name = f'poly_d{deg}_c{coef}_a{alpha}'
            models.append((name, make_poly_fitter(deg, coef, alpha)))

    # --- Model Family 2: RBF Kernel Ridge Regression ---
    rbf_ls_vals = [0.1, 0.3, 0.5, 1.0]
    rbf_alpha_vals = [1e-4, 1e-3, 1e-2]
    
    for ls in rbf_ls_vals:
        for alpha in rbf_alpha_vals:
            def make_rbf_fitter(ls_val, alpha_val):
                def fitter(X_tr, y_tr, X_val):
                    D_tr = cdist(X_tr, X_tr, metric='euclidean')
                    K_tr = np.exp(-0.5 * (D_tr**2) / (ls_val**2))
                    w = np.linalg.solve(K_tr + alpha_val * np.eye(len(y_tr)), y_tr)
                    D_val = cdist(X_val, X_tr, metric='euclidean')
                    K_val = np.exp(-0.5 * (D_val**2) / (ls_val**2))
                    return K_val @ w
                return fitter
            
            name = f'rbf_ls{ls}_a{alpha}'
            models.append((name, make_rbf_fitter(ls, alpha)))

    # --- Model Family 3: Gaussian Process (Matern 5/2) ---
    def matern52_kernel(D, ls, var):
        r = D
        return var * (1.0 + np.sqrt(3.0) * r / ls) * np.exp(-np.sqrt(3.0) * r / ls)

    def fit_gp(X_tr, y_tr, X_val):
        # Optimize hyperparameters on training data
        n_tr = len(y_tr)
        D_tr = cdist(X_tr, X_tr, metric='euclidean')
        
        def nll(params):
            ls = np.exp(params[0])
            var = np.exp(params[1])
            noise = np.exp(params[2])
            K = matern52_kernel(D_tr, ls, var) + noise * np.eye(n_tr)
            try:
                L = np.linalg.cholesky(K)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_tr))
                log_det = 2.0 * np.sum(np.log(np.diag(L)))
                ll = -0.5 * (y_tr @ alpha + log_det + n_tr * np.log(2 * np.pi))
                return -ll
            except:
                return 1e10

        x0 = np.array([0.0, 0.0, -10.0])
        try:
            res = minimize(nll, x0, method='L-BFGS-B', options={'maxiter': 20, 'maxfun': 20})
            ls = np.exp(res.x[0])
            var = np.exp(res.x[1])
            noise = np.exp(res.x[2])
            
            K = matern52_kernel(D_tr, ls, var) + noise * np.eye(n_tr)
            L = np.linalg.cholesky(K)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_tr))
            
            D_val = cdist(X_val, X_tr, metric='euclidean')
            K_star = matern52_kernel(D_val, ls, var)
            mean = K_star @ alpha
            return mean
        except:
            return np.full(len(y_tr), np.mean(y_tr))

    models.append(('gp', fit_gp))

    # --- Selection via 5-Fold CV ---
    indices = np.arange(n)
    np.random.shuffle(indices)
    folds = np.array_split(indices, 5)
    
    cv_scores = {}
    
    for name, fitter in models:
        scores = []
        valid = True
        for i in range(5):
            val_idx = folds[i]
            trn_idx = np.concatenate([folds[j] for j in range(5) if j != i])
            
            X_tr = X_norm[trn_idx]
            y_tr = y_norm[trn_idx]
            X_val = X_norm[val_idx]
            y_val = y_norm[val_idx]
            
            try:
                pred = fitter(X_tr, y_tr, X_val)
                # Ensure finite
                if not np.all(np.isfinite(pred)):
                    valid = False
                    break
                mse = np.mean((pred - y_val)**2)
                scores.append(mse)
            except Exception:
                valid = False
                break
            
            if time.time() > DEADLINE - 10:
                valid = False
                break
        
        if valid and scores:
            cv_scores[name] = np.mean(scores)
        else:
            cv_scores[name] = np.inf

    # Find best model
    if not cv_scores:
        # Fallback if no models could be evaluated
        def predict(X_test):
            return np.full(len(X_test), y_mean)
        return predict

    best_name = min(cv_scores, key=cv_scores.get)
    best_fitter = [f for n, f in models if n == best_name][0]
    
    # Retrain best model on all data
    # We need a version of the fitter that takes X_train and y_train and returns a predict function
    # The current fitters take X_tr, y_tr, X_val and return predictions.
    # We need to wrap the best fitter to create a closure that holds the trained weights/params.
    
    # Since the fitters above are stateless (they retrain every time), 
    # for the final prediction, we will call the fitter with X_train, y_train, and the test X.
    # This is inefficient (retraining on every predict call) but correct for the interface.
    # However, the interface requires `fit` to return a `predict` callable.
    # If we retrain on every predict call, it might be too slow if predict is called many times?
    # The task says "200 separate test points". It's likely called once or a few times.
    # But to be safe, we should train once in `fit` and return a closure.
    
    # Let's refactor: Train the best model once on all data.
    
    if best_name.startswith('poly'):
        # Parse params from name? No, let's just re-implement the specific best fitter logic here.
        # This is fragile. Better approach: 
        # The fitters in `models` are generic. 
        # We can just use the fitter function directly in the predict closure, passing X_train, y_train.
        
        def predict(X_test):
            X_test = np.asarray(X_test, dtype=np.float64)
            X_test_norm = (X_test - X_min) / X_range
            # Call the fitter with training data and test data
            return best_fitter(X_norm, y_norm, X_test_norm) * y_std + y_mean

    elif best_name.startswith('rbf'):
        def predict(X_test):
            X_test = np.asarray(X_test, dtype=np.float64)
            X_test_norm = (X_test - X_min) / X_range
            return best_fitter(X_norm, y_norm, X_test_norm) * y_std + y_mean

    elif best_name == 'gp':
        def predict(X_test):
            X_test = np.asarray(X_test, dtype=np.float64)
            X_test_norm = (X_test - X_min) / X_range
            return best_fitter(X_norm, y_norm, X_test_norm) * y_std + y_mean

    else:
        # Fallback to mean
        def predict(X_test):
            return np.full(len(X_test), y_mean)

    # Optimization: The fitters above retrain on every call. 
    # For GP, this is expensive (minimization). 
    # We should train the GP once here.
    
    if best_name == 'gp':
        # Train GP once
        n_tr = n
        D_tr = cdist(X_norm, X_norm, metric='euclidean')
        
        def nll(params):
            ls = np.exp(params[0])
            var = np.exp(params[1])
            noise = np.exp(params[2])
            K = matern52_kernel(D_tr, ls, var) + noise * np.eye(n_tr)
            try:
                L = np.linalg.cholesky(K)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_norm))
                log_det = 2.0 * np.sum(np.log(np.diag(L)))
                ll = -0.5 * (y_norm @ alpha + log_det + n_tr * np.log(2 * np.pi))
                return -ll
            except:
                return 1e10

        x0 = np.array([0.0, 0.0, -10.0])
        try:
            res = minimize(nll, x0, method='L-BFGS-B', options={'maxiter': 50, 'maxfun': 50})
            ls = np.exp(res.x[0])
            var = np.exp(res.x[1])
            noise = np.exp(res.x[2])
            
            K = matern52_kernel(D_tr, ls, var) + noise * np.eye(n_tr)
            L = np.linalg.cholesky(K)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_norm))
            
            def predict_gp(X_test):
                X_test = np.asarray(X_test, dtype=np.float64)
                X_test_norm = (X_test - X_min) / X_range
                D_val = cdist(X_test_norm, X_norm, metric='euclidean')
                K_star = matern52_kernel(D_val, ls, var)
                mean = K_star @ alpha
                return mean * y_std + y_mean
            return predict_gp
        except:
            pass

    # For Poly and RBF, we can also train once to avoid re-computation in predict.
    # But the fitters in `models` are stateless. 
    # Let's just use the stateless fitter in the closure for Poly/RBF. 
    # It's O(N^2) per predict call. N=200. 200^2 = 40k ops. Very fast.
    
    def predict_generic(X_test):
        X_test = np.asarray(X_test, dtype=np.float64)
        X_test_norm = (X_test - X_min) / X_range
        return best_fitter(X_norm, y_norm, X_test_norm) * y_std + y_mean

    return predict_generic

# EVOLVE-BLOCK-END
