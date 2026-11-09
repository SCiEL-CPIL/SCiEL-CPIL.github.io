"""Symbolic regression via Sparse Gaussian Process (FITC) with composite kernels."""
import numpy as np
import time
from scipy.optimize import minimize

def _rbf_kernel(X1, X2, ls):
    """RBF kernel with independent length scales."""
    if np.isscalar(ls):
        ls = np.full(X1.shape[1], ls)
    X1_s = X1 / ls
    X2_s = X2 / ls
    sq_dists = np.sum(X1_s**2, axis=1)[:, None] + np.sum(X2_s**2, axis=1)[None, :] - 2 * (X1_s @ X2_s.T)
    np.maximum(sq_dists, 0, out=sq_dists)
    return np.exp(-0.5 * sq_dists)

def _poly_kernel(X1, X2, degree, coef0):
    """Polynomial kernel: (coef0 + <x1, x2>)^degree."""
    return (coef0 + X1 @ X2.T) ** degree

def _periodic_kernel(X1, X2, period):
    """Periodic kernel based on Euclidean distance."""
    dists_sq = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * X1 @ X2.T
    np.maximum(dists_sq, 0, out=dists_sq)
    dists = np.sqrt(dists_sq)
    return np.exp(-2.0 * np.sin(np.pi * dists / period)**2)

def _compute_composite_kernel(X1, X2, rbf_ls, rbf_var, poly_deg, poly_var, per_var, per_period):
    """Compute composite kernel: rbf_var * RBF + poly_var * Poly + per_var * Periodic."""
    K = rbf_var * _rbf_kernel(X1, X2, rbf_ls)
    if poly_var > 1e-10:
        K = K + poly_var * _poly_kernel(X1, X2, poly_deg, 1.0)
    if per_var > 1e-10:
        K = K + per_var * _periodic_kernel(X1, X2, per_period)
    return K

def fit(X_train, y_train):
    t0 = time.time()
    deadline = t0 + 290.0
    
    n, d = X_train.shape
    
    # Standardize inputs
    X_mean = X_train.mean(axis=0)
    X_std = X_train.std(axis=0)
    X_std[X_std < 1e-8] = 1.0
    X_norm = (X_train - X_mean) / X_std
    
    # Standardize outputs
    y_mean = y_train.mean()
    y_std_val = float(y_train.std()) if y_train.std() > 1e-8 else 1.0
    y_norm = (y_train - y_mean) / y_std_val

    # Select inducing points using k-means-like strategy or random sampling
    # For FITC, we need M < N inducing points.
    # We'll use M = min(50, n) for efficiency.
    M = min(50, n)
    
    # Simple k-means implementation for inducing point selection
    def kmeans(X, k, rng, n_iter=10):
        # Initialize centers randomly
        centers = X[rng.choice(n, size=k, replace=False)].copy()
        for _ in range(n_iter):
            # Assign points to nearest center
            dists = np.sum((X[:, None, :] - centers[None, :, :])**2, axis=2)
            labels = np.argmin(dists, axis=1)
            # Update centers
            for j in range(k):
                mask = labels == j
                if np.sum(mask) > 0:
                    centers[j] = X[mask].mean(axis=0)
                else:
                    # Reinitialize if empty cluster
                    centers[j] = X[rng.choice(n)]
        return centers

    rng = np.random.RandomState(42)
    Z = kmeans(X_norm, M, rng)

    # Precompute kernels involving inducing points
    # K_ZZ: M x M
    # K_ZX: M x N
    # We need to recompute these when hyperparameters change.
    
    def loocv_fitc_score(rbf_ls, rbf_var, poly_deg, poly_var, per_var, per_period, alpha, noise_var):
        """
        Compute LOOCV score for FITC approximation.
        alpha: regularization for K_ZZ (sigma^2)
        noise_var: noise variance (sigma_n^2)
        """
        K_ZZ = _compute_composite_kernel(Z, Z, rbf_ls, rbf_var, poly_deg, poly_var, per_var, per_period)
        K_ZX = _compute_composite_kernel(Z, X_norm, rbf_ls, rbf_var, poly_deg, poly_var, per_var, per_period)
        
        # FITC matrices:
        # Q = K_ZZ + alpha * I_M
        # S = Q^-1 K_ZX
        # V = K_XZ S^T - K_XX (approximation of covariance)
        
        # For LOOCV in FITC, the prediction variance and mean updates are more complex.
        # A simpler approach for LOOCV in sparse GP is to use the "leave-one-out" formula 
        # derived from the marginal likelihood or approximate it.
        # However, a common approximation for LOOCV in sparse GP is to treat it as 
        # standard KRR on the inducing points if we assume the data lies on the manifold.
        # But that ignores the noise.
        
        # Let's use a standard 5-fold CV instead of LOOCV for FITC, as LOOCV formulas 
        # for FITC are non-trivial and error-prone.
        # Or, we can optimize the marginal log-likelihood (MLL) of FITC.
        
        # MLL of FITC:
        # log p(y|X, Z, theta) = -0.5 * y^T (V + noise*I)^-1 y - 0.5 log|V + noise*I| - 0.5 n log(2pi) + const
        # where V = K_ZX (K_ZZ + alpha I)^-1 K_XZ
        
        # To avoid inverting N x N matrix, we use Woodbury or similar tricks.
        # (V + noise I)^-1 = I - (1/noise) K_XZ (Q + (1/noise) K_ZX K_XZ)^-1 K_ZX
        # This is still O(N M^2) or O(M^3 + N M^2).
        
        # Let's stick to 5-fold CV for simplicity and robustness.
        
        n_folds = 5
        indices = np.arange(n)
        rng_cv = np.random.RandomState(123)
        rng_cv.shuffle(indices)
        folds = [indices[i::n_folds] for i in range(n_folds)]
        
        cv_mses = []
        
        for i in range(n_folds):
            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != i])
            
            # We need to recompute kernels for the training subset?
            # No, FITC uses all inducing points. The training set changes.
            # K_ZZ is fixed.
            # K_ZX_train = K_ZX[:, train_idx]
            # y_train = y_norm[train_idx]
            
            K_ZX_train = K_ZX[:, train_idx]
            y_tr = y_norm[train_idx]
            y_val = y_norm[val_idx]
            
            # Q = K_ZZ + alpha I
            Q = K_ZZ + alpha * np.eye(M)
            
            # Solve Q c = K_ZX_train^T y_tr  => c = Q^-1 K_ZX_train^T y_tr
            try:
                L_Q = np.linalg.cholesky(Q)
                # c = L_Q^-T L_Q^-1 K_ZX_train^T y_tr
                tmp = np.linalg.solve(L_Q, K_ZX_train.T @ y_tr)
                c = np.linalg.solve(L_Q.T, tmp)
            except np.linalg.LinAlgError:
                try:
                    c = np.linalg.solve(Q, K_ZX_train.T @ y_tr)
                except np.linalg.LinAlgError:
                    return np.inf
            
            # Prediction for val:
            # y_hat = K_X_val Z c
            # K_X_val Z = K_ZX[val_idx, :].T
            K_val_Z = K_ZX[val_idx, :].T
            y_hat = K_val_Z @ c
            
            # Add noise variance? In FITC, the prediction variance includes noise.
            # For point prediction, we use the mean.
            mse = np.mean((y_hat - y_val)**2)
            cv_mses.append(mse)
            
        return np.mean(cv_mses)

    best_score = np.inf
    best_params = None
    
    # Hyperparameter ranges
    # rbf_ls: 0.1 to 10
    # rbf_var: 0.1 to 100
    # poly_deg: 1, 2, 3
    # poly_var: 0.01 to 10
    # per_var: 0.01 to 10
    # per_period: 1 to 30
    # alpha (sigma^2): 1e-4 to 10
    # noise_var (sigma_n^2): 1e-4 to 10 (though not used in point pred mean, it's in MLL)
    
    n_random_evals = 150
    for i in range(n_random_evals):
        if time.time() > deadline - 30:
            break
            
        ls = 10 ** rng.uniform(-1, 1, size=d)
        rbf_var = 10 ** rng.uniform(-1, 2)
        poly_deg = int(rng.choice([1, 2, 3]))
        poly_var = 10 ** rng.uniform(-2, 1)
        per_var = 10 ** rng.uniform(-2, 1)
        per_period = 10 ** rng.uniform(0, 1.5)
        alpha = 10 ** rng.uniform(-4, 2)
        noise_var = 10 ** rng.uniform(-4, 2)
        
        try:
            score = loocv_fitc_score(ls, rbf_var, poly_deg, poly_var, per_var, per_period, alpha, noise_var)
            if score < best_score:
                best_score = score
                best_params = (ls, rbf_var, poly_deg, poly_var, per_var, per_period, alpha, noise_var)
        except Exception:
            continue
    
    if best_params is None:
        best_params = (np.ones(d), 1.0, 2, 1.0, 1.0, 5.0, 1.0, 1.0)
    
    # Local Refinement
    best_ls, best_rbf_var, best_poly_deg, best_poly_var, best_per_var, best_per_period, best_alpha, best_noise = best_params
    
    def objective_continuous(x):
        ls = np.exp(x[:d])
        rbf_var = np.exp(x[d])
        poly_var = np.exp(x[d+1])
        per_var = np.exp(x[d+2])
        per_period = np.exp(x[d+3])
        alpha = np.exp(x[d+4])
        noise_var = np.exp(x[d+5])
        
        ls = np.clip(ls, 0.01, 100.0)
        rbf_var = np.clip(rbf_var, 0.01, 1000.0)
        poly_var = np.clip(poly_var, 0.001, 100.0)
        per_var = np.clip(per_var, 0.001, 100.0)
        per_period = np.clip(per_period, 0.5, 100.0)
        alpha = np.clip(alpha, 1e-6, 1000.0)
        noise_var = np.clip(noise_var, 1e-6, 1000.0)
        
        try:
            score = loocv_fitc_score(ls, rbf_var, best_poly_deg, poly_var, per_var, per_period, alpha, noise_var)
            return score
        except Exception:
            return np.inf
    
    x0 = np.concatenate([
        np.log(best_ls),
        [np.log(best_rbf_var)],
        [np.log(best_poly_var)],
        [np.log(best_per_var)],
        [np.log(best_per_period)],
        [np.log(best_alpha)],
        [np.log(best_noise)]
    ])
    
    bounds = [(-4, 4)] * d + [(-4, 4), (-4, 4), (-4, 4), (-4, 4), (-4, 4), (-4, 4)]
    
    candidate_degrees = sorted(set([best_poly_deg - 1, best_poly_deg, best_poly_deg + 1, 1, 2, 3]))
    candidate_degrees = [deg for deg in candidate_degrees if 1 <= deg <= 5]
    
    best_global_params = best_params
    best_global_score = best_score
    
    for deg in candidate_degrees:
        if time.time() > deadline - 10:
            break
            
        # Update objective to use this degree
        def obj_with_deg(x, deg=deg):
            ls = np.exp(x[:d])
            rbf_var = np.exp(x[d])
            poly_var = np.exp(x[d+1])
            per_var = np.exp(x[d+2])
            per_period = np.exp(x[d+3])
            alpha = np.exp(x[d+4])
            noise_var = np.exp(x[d+5])
            
            ls = np.clip(ls, 0.01, 100.0)
            rbf_var = np.clip(rbf_var, 0.01, 1000.0)
            poly_var = np.clip(poly_var, 0.001, 100.0)
            per_var = np.clip(per_var, 0.001, 100.0)
            per_period = np.clip(per_period, 0.5, 100.0)
            alpha = np.clip(alpha, 1e-6, 1000.0)
            noise_var = np.clip(noise_var, 1e-6, 1000.0)
            
            try:
                score = loocv_fitc_score(ls, rbf_var, deg, poly_var, per_var, per_period, alpha, noise_var)
                return score
            except Exception:
                return np.inf
        
        try:
            res = minimize(obj_with_deg, x0, method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 50, 'ftol': 1e-6, 'gtol': 1e-4})
            
            if res.fun < best_global_score:
                ls = np.exp(res.x[:d])
                rbf_var = np.exp(res.x[d])
                poly_var = np.exp(res.x[d+1])
                per_var = np.exp(res.x[d+2])
                per_period = np.exp(res.x[d+3])
                alpha = np.exp(res.x[d+4])
                noise_var = np.exp(res.x[d+5])
                
                ls = np.clip(ls, 0.01, 100.0)
                rbf_var = np.clip(rbf_var, 0.01, 1000.0)
                poly_var = np.clip(poly_var, 0.001, 100.0)
                per_var = np.clip(per_var, 0.001, 100.0)
                per_period = np.clip(per_period, 0.5, 100.0)
                alpha = np.clip(alpha, 1e-6, 1000.0)
                noise_var = np.clip(noise_var, 1e-6, 1000.0)
                
                best_global_score = res.fun
                best_global_params = (ls, rbf_var, deg, poly_var, per_var, per_period, alpha, noise_var)
                x0 = res.x
        except Exception:
            pass
    
    # Final fit
    rbf_ls, rbf_var, poly_deg, poly_var, per_var, per_period, alpha, noise_var = best_global_params
    
    K_ZZ = _compute_composite_kernel(Z, Z, rbf_ls, rbf_var, poly_deg, poly_var, per_var, per_period)
    K_ZX = _compute_composite_kernel(Z, X_norm, rbf_ls, rbf_var, poly_deg, poly_var, per_var, per_period)
    
    Q = K_ZZ + alpha * np.eye(M)
    
    try:
        L_Q = np.linalg.cholesky(Q)
        # c = Q^-1 K_ZX^T y
        tmp = np.linalg.solve(L_Q, K_ZX.T @ y_norm)
        c = np.linalg.solve(L_Q.T, tmp)
    except np.linalg.LinAlgError:
        try:
            c = np.linalg.solve(Q, K_ZX.T @ y_norm)
        except np.linalg.LinAlgError:
            # Fallback to standard KRR if FITC fails
            K_final = _compute_composite_kernel(X_norm, X_norm, rbf_ls, rbf_var, poly_deg, poly_var, per_var, per_period)
            A = K_final + alpha * np.eye(n)
            try:
                c_full = np.linalg.solve(A, y_norm)
                def predict_fallback(X):
                    X = np.asarray(X, dtype=float)
                    if X.ndim == 1:
                        X = X.reshape(1, -1)
                    X_test_norm = (X - X_mean) / X_std
                    K_test = _compute_composite_kernel(X_test_norm, X_norm, rbf_ls, rbf_var, poly_deg, poly_var, per_var, per_period)
                    preds = K_test @ c_full
                    preds = preds * y_std_val + y_mean
                    preds = np.where(np.isfinite(preds), preds, 0.0)
                    return preds
                return predict_fallback
            except:
                c = np.zeros(M)

    def predict(X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        X_test_norm = (X - X_mean) / X_std
        
        # K_XZ: N_test x M
        K_XZ = _compute_composite_kernel(X_test_norm, Z, rbf_ls, rbf_var, poly_deg, poly_var, per_var, per_period)
        
        # y_hat = K_XZ c
        preds = K_XZ @ c
        preds = preds * y_std_val + y_mean
        preds = np.where(np.isfinite(preds), preds, 0.0)
        return preds

    return predict
