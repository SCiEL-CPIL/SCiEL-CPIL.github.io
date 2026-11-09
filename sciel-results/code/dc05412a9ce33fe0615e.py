"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _rbf_kernel(X1, X2, length_scale):
    """Compute RBF kernel matrix between X1 and X2."""
    # X1: (n1, d), X2: (n2, d)
    sq1 = np.sum(X1 ** 2, axis=1, keepdims=True)
    sq2 = np.sum(X2 ** 2, axis=1, keepdims=True).T
    sqd = sq1 + sq2 - 2.0 * (X1 @ X2.T)
    np.maximum(sqd, 0.0, out=sqd)
    return np.exp(-sqd / (2.0 * length_scale ** 2))


def _poly_kernel(X1, X2, degree=3, gamma=1.0, coef0=1.0):
    """Compute polynomial kernel matrix."""
    return (gamma * (X1 @ X2.T) + coef0) ** degree


def _composite_kernel(X1, X2, length_scale, poly_degree, poly_gamma, poly_coef0, w_rbf, w_poly):
    """Compute composite kernel: RBF + polynomial."""
    K_rbf = _rbf_kernel(X1, X2, length_scale)
    K_poly = _poly_kernel(X1, X2, poly_degree, poly_gamma, poly_coef0)
    return w_rbf * K_rbf + w_poly * K_poly


def _solve_ridge(K, y, alpha):
    """Solve kernel ridge regression: (K + alpha*I)^{-1} y."""
    n = K.shape[0]
    A = K + alpha * np.eye(n)
    try:
        # Use Cholesky for stability
        L = np.linalg.cholesky(A)
        # Solve L L^T w = y
        tmp = np.linalg.solve(L, y)
        w = np.linalg.solve(L.T, tmp)
        return w
    except np.linalg.LinAlgError:
        # Fallback to lstsq
        w, *_ = np.linalg.lstsq(A, y, rcond=None)
        return w


def fit(X_train, y_train):
    t0 = time.time()
    deadline = t0 + 290.0  # Leave margin for 330s total across all equations
    
    n, d = X_train.shape
    
    # Hyperparameter grid
    length_scales = [0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0]
    alphas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0]
    poly_degrees = [2, 3]
    poly_gammas = [0.5, 1.0, 2.0]
    poly_coef0s = [1.0, 2.0]
    w_rbf_list = [0.5, 1.0]
    w_poly_list = [0.5, 1.0]
    
    # Cross-validation setup: 5-fold
    n_folds = 5
    indices = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(indices)
    fold_indices = np.array_split(indices, n_folds)
    
    best_score = np.inf
    best_params = None
    
    # To speed up, we'll do a coarser search first, then refine
    # Phase 1: Coarse search over length_scale and alpha
    for ls in length_scales:
        if time.time() > deadline - 30:
            break
        for alpha in alphas:
            # Compute kernel once
            K_full = _rbf_kernel(X_train, X_train, ls)
            
            cv_scores = []
            for i in range(n_folds):
                if time.time() > deadline - 30:
                    break
                val_idx = fold_indices[i]
                train_idx = np.concatenate([fold_indices[j] for j in range(n_folds) if j != i])
                
                K_train = K_full[np.ix_(train_idx, train_idx)]
                y_train_cv = y_train[train_idx]
                y_val_cv = y_train[val_idx]
                
                w = _solve_ridge(K_train, y_train_cv, alpha)
                K_val = K_full[np.ix_(val_idx, train_idx)]
                pred_val = K_val @ w
                
                mse_val = np.mean((pred_val - y_val_cv) ** 2)
                var_y = np.var(y_val_cv)
                if var_y > 1e-12:
                    nmse_val = mse_val / var_y
                else:
                    nmse_val = mse_val
                cv_scores.append(nmse_val)
            
            if len(cv_scores) == n_folds:
                mean_score = np.mean(cv_scores)
                if mean_score < best_score:
                    best_score = mean_score
                    best_params = (ls, alpha, None)
    
    # Phase 2: If time permits, try composite kernels
    remaining_time = deadline - time.time()
    if remaining_time > 60:
        for ls in length_scales[1:5]:  # Subset of length scales
            if time.time() > deadline - 30:
                break
            for alpha in alphas[1:6]:  # Subset of alphas
                if time.time() > deadline - 30:
                    break
                for poly_deg in poly_degrees:
                    if time.time() > deadline - 30:
                        break
                    for poly_g in poly_gammas:
                        if time.time() > deadline - 30:
                            break
                        for poly_c0 in poly_coef0s:
                            if time.time() > deadline - 30:
                                break
                            for w_rbf in w_rbf_list:
                                if time.time() > deadline - 30:
                                    break
                                for w_poly in w_poly_list:
                                    if time.time() > deadline - 30:
                                        break
                                    
                                    K_full = _composite_kernel(
                                        X_train, X_train, ls, poly_deg, poly_g, poly_c0, w_rbf, w_poly
                                    )
                                    
                                    cv_scores = []
                                    for i in range(n_folds):
                                        if time.time() > deadline - 30:
                                            break
                                        val_idx = fold_indices[i]
                                        train_idx = np.concatenate([fold_indices[j] for j in range(n_folds) if j != i])
                                        
                                        K_train = K_full[np.ix_(train_idx, train_idx)]
                                        y_train_cv = y_train[train_idx]
                                        y_val_cv = y_train[val_idx]
                                        
                                        w = _solve_ridge(K_train, y_train_cv, alpha)
                                        K_val = K_full[np.ix_(val_idx, train_idx)]
                                        pred_val = K_val @ w
                                        
                                        mse_val = np.mean((pred_val - y_val_cv) ** 2)
                                        var_y = np.var(y_val_cv)
                                        if var_y > 1e-12:
                                            nmse_val = mse_val / var_y
                                        else:
                                            nmse_val = mse_val
                                        cv_scores.append(nmse_val)
                                    
                                    if len(cv_scores) == n_folds:
                                        mean_score = np.mean(cv_scores)
                                        if mean_score < best_score:
                                            best_score = mean_score
                                            best_params = (ls, alpha, (poly_deg, poly_g, poly_c0, w_rbf, w_poly))
    
    # Final fit with best parameters
    if best_params is None:
        # Fallback: simple RBF with default params
        ls, alpha = 1.0, 1e-3
        K_full = _rbf_kernel(X_train, X_train, ls)
        w = _solve_ridge(K_full, y_train, alpha)
    else:
        ls, alpha, poly_params = best_params
        if poly_params is None:
            K_full = _rbf_kernel(X_train, X_train, ls)
        else:
            poly_deg, poly_g, poly_c0, w_rbf, w_poly = poly_params
            K_full = _composite_kernel(
                X_train, X_train, ls, poly_deg, poly_g, poly_c0, w_rbf, w_poly
            )
        w = _solve_ridge(K_full, y_train, alpha)
    
    # Store training data and weights for prediction
    X_train_copy = X_train.copy()
    w_copy = w.copy()
    
    def predict(X):
        # Compute kernel between test points and training points
        if best_params is not None and best_params[2] is not None:
            poly_deg, poly_g, poly_c0, w_rbf, w_poly = best_params[2]
            K_test = _composite_kernel(X, X_train_copy, best_params[0], poly_deg, poly_g, poly_c0, w_rbf, w_poly)
        else:
            K_test = _rbf_kernel(X, X_train_copy, best_params[0])
        
        return K_test @ w_copy
    
    return predict
# EVOLVE-BLOCK-END
