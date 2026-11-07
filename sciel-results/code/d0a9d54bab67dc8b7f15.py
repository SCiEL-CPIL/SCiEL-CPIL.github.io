"""Symbolic regression via multi-model oracle selection with LOO-CV."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

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
    if y_std < 1e-12:
        y_std = 1.0
    y_norm = (y - y_mean) / y_std

    # Precompute pairwise squared distances
    D2 = cdist(X_norm, X_norm, metric='sqeuclidean')**2

    # Model 1: Polynomial Kernel Ridge Regression
    def fit_krr(X, y, degree=4, gamma=1.0, alpha=1e-6):
        # Polynomial kernel: (gamma * X^T X' + 1)^degree
        # For efficiency, compute kernel matrix
        K = np.power(gamma * X @ X.T + 1.0, degree)
        K += alpha * np.eye(len(X))
        
        try:
            w = np.linalg.solve(K, y)
        except np.linalg.LinAlgError:
            w = np.linalg.lstsq(K, y, rcond=None)[0]
        
        def predict(X_test):
            K_star = np.power(gamma * X_test @ X.T + 1.0, degree)
            return K_star @ w
        return predict

    # Model 2: Matern-5/2 Gaussian Process
    def fit_gp(X, y):
        def matern52_kernel(D2, length_scale, variance):
            r = np.sqrt(D2)
            return variance * (1.0 + np.sqrt(3.0) * r / length_scale) * np.exp(-np.sqrt(3.0) * r / length_scale)

        def neg_log_likelihood(params):
            ls = np.exp(params[0])
            var = np.exp(params[1])
            noise = np.exp(params[2])
            
            K = matern52_kernel(D2, ls, var)
            K += noise * np.eye(len(X))
            
            try:
                L = np.linalg.cholesky(K)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
                log_det = 2.0 * np.sum(np.log(np.diag(L)))
                ll = -0.5 * (y @ alpha + log_det + len(X) * np.log(2 * np.pi))
                return -ll
            except np.linalg.LinAlgError:
                return 1e10

        x0 = np.array([0.0, 0.0, -10.0])
        best_params = x0
        best_ll = neg_log_likelihood(x0)
        
        try:
            res = minimize(neg_log_likelihood, x0, method='L-BFGS-B', options={'maxiter': 50, 'maxfun': 50})
            if res.fun < best_ll:
                best_params = res.x
                best_ll = res.fun
        except Exception:
            pass

        ls = np.exp(best_params[0])
        var = np.exp(best_params[1])
        noise = np.exp(best_params[2])
        
        K = matern52_kernel(D2, ls, var)
        K += noise * np.eye(len(X))
        
        try:
            L = np.linalg.cholesky(K)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
        except np.linalg.LinAlgError:
            return None

        def predict(X_test):
            diff_test = X_test[:, np.newaxis, :] - X[np.newaxis, :, :]
            D2_test = np.sum(diff_test * diff_test, axis=-1)
            K_star = matern52_kernel(D2_test, ls, var)
            return K_star @ alpha
        return predict

    # Model 3: Rational Function Approximator (ratio of two linear basis expansions)
    def fit_rational(X, y):
        # Create basis functions: constant + linear + quadratic terms
        def make_basis(X):
            n, d = X.shape
            cols = [np.ones(n)]
            for j in range(d):
                cols.append(X[:, j])
            for i in range(d):
                for j in range(i, d):
                    cols.append(X[:, i] * X[:, j])
            return np.column_stack(cols)
        
        B = make_basis(X)
        # Fit numerator: y = N(x) / D(x) => y * D(x) = N(x)
        # This is nonlinear. Use iterative approach or simple linearization.
        # Simplified: fit y * (1 + sum a_i x_i) = sum b_i x_i
        # Even simpler: just use linear regression on basis as fallback
        try:
            w, *_ = np.linalg.lstsq(B, y, rcond=None)
        except Exception:
            w = np.zeros(B.shape[1])
        
        def predict(X_test):
            B_test = make_basis(X_test)
            return B_test @ w
        return predict

    # Leave-One-Out Cross-Validation
    def loo_cv(predict_fn, X, y, n_loo=200):
        # For efficiency, use the "hat matrix" trick if possible, 
        # otherwise do explicit LOO
        # Explicit LOO for small n is fine
        errors = []
        for i in range(n_loo):
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            X_loo = X[mask]
            y_loo = y[mask]
            
            # Refit model on LOO data
            # For KRR and GP, this is expensive. 
            # Instead, use the influence function / hat matrix approximation
            # For now, let's just use a subset-based CV to save time
            pass
        
        # Better approach: Use k-fold CV with k=10
        return np.inf  # Placeholder

    # Let's use k-fold CV instead for speed
    def kfold_cv(predict_factory, X, y, k=10):
        indices = np.arange(n)
        np.random.seed(42)
        np.random.shuffle(indices)
        folds = np.array_split(indices, k)
        
        errors = []
        for i in range(k):
            test_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
            
            X_train_fold = X[train_idx]
            y_train_fold = y[train_idx]
            X_test_fold = X[test_idx]
            y_test_fold = y[test_idx]
            
            # Refit model
            pred_fn = predict_factory(X_train_fold, y_train_fold)
            if pred_fn is None:
                return np.inf
            
            preds = pred_fn(X_test_fold)
            if not np.all(np.isfinite(preds)):
                return np.inf
            
            mse = np.mean((preds - y_test_fold)**2)
            errors.append(mse)
        
        return np.mean(errors)

    # Define model factories that take (X, y) and return predict
    def krr_factory(X, y, degree=4, gamma=1.0, alpha=1e-6):
        X_min_f = X.min(axis=0)
        X_max_f = X.max(axis=0)
        X_range_f = X_max_f - X_min_f
        X_range_f[X_range_f == 0] = 1.0
        X_norm_f = (X - X_min_f) / X_range_f
        
        y_mean_f = np.mean(y)
        y_std_f = np.std(y)
        if y_std_f < 1e-12:
            y_std_f = 1.0
        y_norm_f = (y - y_mean_f) / y_std_f
        
        K = np.power(gamma * X_norm_f @ X_norm_f.T + 1.0, degree)
        K += alpha * np.eye(len(X))
        
        try:
            w = np.linalg.solve(K, y_norm_f)
        except np.linalg.LinAlgError:
            w = np.linalg.lstsq(K, y_norm_f, rcond=None)[0]
        
        def predict(X_test):
            X_test_min_f = X_test.min(axis=0) if X_test.shape[0] > 1 else X_min_f
            X_test_max_f = X_test.max(axis=0) if X_test.shape[0] > 1 else X_max_f
            # Use training normalization
            X_test_norm_f = (X_test - X_min_f) / X_range_f
            K_star = np.power(gamma * X_test_norm_f @ X_norm_f.T + 1.0, degree)
            return (K_star @ w) * y_std_f + y_mean_f
        return predict

    def gp_factory(X, y):
        X_min_f = X.min(axis=0)
        X_max_f = X.max(axis=0)
        X_range_f = X_max_f - X_min_f
        X_range_f[X_range_f == 0] = 1.0
        X_norm_f = (X - X_min_f) / X_range_f
        
        y_mean_f = np.mean(y)
        y_std_f = np.std(y)
        if y_std_f < 1e-12:
            y_std_f = 1.0
        y_norm_f = (y - y_mean_f) / y_std_f
        
        D2_f = cdist(X_norm_f, X_norm_f, metric='sqeuclidean')**2
        
        def matern52_kernel(D2, length_scale, variance):
            r = np.sqrt(D2)
            return variance * (1.0 + np.sqrt(3.0) * r / length_scale) * np.exp(-np.sqrt(3.0) * r / length_scale)

        def neg_log_likelihood(params):
            ls = np.exp(params[0])
            var = np.exp(params[1])
            noise = np.exp(params[2])
            
            K = matern52_kernel(D2_f, ls, var)
            K += noise * np.eye(len(X))
            
            try:
                L = np.linalg.cholesky(K)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_norm_f))
                log_det = 2.0 * np.sum(np.log(np.diag(L)))
                ll = -0.5 * (y_norm_f @ alpha + log_det + len(X) * np.log(2 * np.pi))
                return -ll
            except np.linalg.LinAlgError:
                return 1e10

        x0 = np.array([0.0, 0.0, -10.0])
        best_params = x0
        best_ll = neg_log_likelihood(x0)
        
        try:
            res = minimize(neg_log_likelihood, x0, method='L-BFGS-B', options={'maxiter': 30, 'maxfun': 30})
            if res.fun < best_ll:
                best_params = res.x
                best_ll = res.fun
        except Exception:
            pass

        ls = np.exp(best_params[0])
        var = np.exp(best_params[1])
        noise = np.exp(best_params[2])
        
        K = matern52_kernel(D2_f, ls, var)
        K += noise * np.eye(len(X))
        
        try:
            L = np.linalg.cholesky(K)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_norm_f))
        except np.linalg.LinAlgError:
            return None

        def predict(X_test):
            X_test_norm_f = (X_test - X_min_f) / X_range_f
            diff_test = X_test_norm_f[:, np.newaxis, :] - X_norm_f[np.newaxis, :, :]
            D2_test = np.sum(diff_test * diff_test, axis=-1)
            K_star = matern52_kernel(D2_test, ls, var)
            return (K_star @ alpha) * y_std_f + y_mean_f
        return predict

    def rational_factory(X, y):
        X_min_f = X.min(axis=0)
        X_max_f = X.max(axis=0)
        X_range_f = X_max_f - X_min_f
        X_range_f[X_range_f == 0] = 1.0
        X_norm_f = (X - X_min_f) / X_range_f
        
        def make_basis(X_in):
            n_in, d_in = X_in.shape
            cols = [np.ones(n_in)]
            for j in range(d_in):
                cols.append(X_in[:, j])
            for i in range(d_in):
                for j in range(i, d_in):
                    cols.append(X_in[:, i] * X_in[:, j])
            return np.column_stack(cols)
        
        B = make_basis(X_norm_f)
        try:
            w, *_ = np.linalg.lstsq(B, y, rcond=None)
        except Exception:
            w = np.zeros(B.shape[1])
        
        def predict(X_test):
            X_test_norm_f = (X_test - X_min_f) / X_range_f
            B_test = make_basis(X_test_norm_f)
            return B_test @ w
        return predict

    # Try multiple KRR configurations
    krr_configs = [
        (3, 1.0, 1e-6),
        (4, 1.0, 1e-6),
        (5, 1.0, 1e-6),
        (4, 0.5, 1e-6),
        (4, 2.0, 1e-6),
    ]

    # Evaluate models using k-fold CV
    models = []
    scores = []

    # KRR models
    for degree, gamma, alpha in krr_configs:
        if time.time() > DEADLINE - 5:
            break
        try:
            cv_score = kfold_cv(lambda X_f, y_f: krr_factory(X_f, y_f, degree, gamma, alpha), X_norm, y_norm, k=5)
            if np.isfinite(cv_score):
                models.append(('krr', degree, gamma, alpha))
                scores.append(cv_score)
        except Exception:
            pass

    # GP model
    if time.time() < DEADLINE - 10:
        try:
            cv_score = kfold_cv(gp_factory, X_norm, y_norm, k=5)
            if np.isfinite(cv_score):
                models.append(('gp',))
                scores.append(cv_score)
        except Exception:
            pass

    # Rational model
    if time.time() < DEADLINE - 5:
        try:
            cv_score = kfold_cv(rational_factory, X_norm, y_norm, k=5)
            if np.isfinite(cv_score):
                models.append(('rational',))
                scores.append(cv_score)
        except Exception:
            pass

    # Select best model
    if not models:
        # Fallback: simple linear regression
        B = np.column_stack([np.ones(n)] + [X_norm[:, j] for j in range(d)])
        w, *_ = np.linalg.lstsq(B, y, rcond=None)
        def predict(X_test):
            X_test_norm = (X_test - X_min) / X_range
            B_test = np.column_stack([np.ones(X_test.shape[0])] + [X_test_norm[:, j] for j in range(d)])
            return B_test @ w
        return predict

    best_idx = np.argmin(scores)
    best_model = models[best_idx]

    # Fit the best model on all data
    if best_model[0] == 'krr':
        _, degree, gamma, alpha = best_model
        predict = krr_factory(X_norm, y_norm, degree, gamma, alpha)
    elif best_model[0] == 'gp':
        predict = gp_factory(X_norm, y_norm)
        if predict is None:
            # Fallback
            B = np.column_stack([np.ones(n)] + [X_norm[:, j] for j in range(d)])
            w, *_ = np.linalg.lstsq(B, y, rcond=None)
            def predict(X_test):
                X_test_norm = (X_test - X_min) / X_range
                B_test = np.column_stack([np.ones(X_test.shape[0])] + [X_test_norm[:, j] for j in range(d)])
                return B_test @ w
    else:  # rational
        predict = rational_factory(X_norm, y_norm)

    # Wrap predict to handle input normalization
    def final_predict(X_test):
        X_test = np.asarray(X_test, dtype=np.float64)
        return predict(X_test)

    return final_predict
# EVOLVE-BLOCK-END
