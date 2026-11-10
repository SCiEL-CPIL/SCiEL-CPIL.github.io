import numpy as np
import time

def construct(training_x, training_subgrid_tendency):
    t_start = time.time()
    deadline = t_start + 135.0  # Safety margin before 150s kill

    x = np.asarray(training_x, dtype=np.float64)
    y = np.asarray(training_subgrid_tendency, dtype=np.float64)

    # Remove any NaN/Inf
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    n = len(x)
    if n < 2:
        return np.array([np.mean(y) if n > 0 else 0.0, 0, 0, 0, 0, 0], dtype=np.float64)

    y_var = np.var(y)
    if y_var < 1e-15:
        y_var = 1e-15

    # Standardize x for better conditioning of the RBF centers and bandwidth
    mu_x = np.mean(x)
    sigma_x = np.std(x)
    if sigma_x < 1e-12:
        sigma_x = 1.0
    x_norm = (x - mu_x) / sigma_x

    # RBF Model: f(x) = sum_j w_j * exp(-0.5 * ((x - c_j) / sigma_rbf)^2)
    # Centers c_j are the normalized training points x_norm.
    
    # Precompute pairwise squared distances for the training set to build the Gram matrix
    # D[i, j] = (x_norm[i] - x_norm[j])^2
    # To save memory and time for large n, we might need to be careful, but n is likely manageable.
    # Let's check n. If n is huge, we might subsample centers, but let's assume standard size first.
    # For n ~ 1000, D is 1M floats, which is fine.
    
    # Compute D efficiently
    x_sq = np.sum(x_norm**2, axis=1)
    D = x_sq[:, np.newaxis] + x_sq[np.newaxis, :] - 2.0 * np.dot(x_norm, x_norm.T)
    # Clip negative values due to floating point errors
    np.maximum(D, 0, out=D)

    best_score = np.inf
    best_sigma = 1.0
    best_alpha = 1e-6

    # Grid search for bandwidth sigma_rbf and regularization alpha
    # sigma_rbf controls the width of the Gaussian kernels
    # alpha controls the regularization of the weights (Tikhonov)
    
    # Heuristic range for sigma based on typical spacing
    # Median distance between points is a good scale
    # We can approximate this or just use a logspace relative to the domain width
    # Domain width in normalized space is roughly 2*sigma_x/sigma_x = 2? No, x_norm has std 1.
    # So typical distances are O(1).
    
    sigma_candidates = np.logspace(-1.5, 1.0, 15)  # 0.03 to 10
    alpha_candidates = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]

    # Cross-validation function
    def _cv_mse_rbf(sigma_rbf, alpha):
        # 5-fold CV
        n_samples = len(x_norm)
        k_folds = 5
        if n_samples < 50:
            k_folds = 2
            
        idx = np.random.RandomState(42).permutation(n_samples)
        fold_size = n_samples // k_folds
        mse_sum = 0.0
        
        for k in range(k_folds):
            lo = k * fold_size
            hi = lo + fold_size
            if k == k_folds - 1:
                hi = n_samples
            
            val_idx = idx[lo:hi]
            train_idx = np.concatenate([idx[:lo], idx[hi:]])
            
            x_train = x_norm[train_idx]
            y_train = y[train_idx]
            x_val = x_norm[val_idx]
            y_val = y[val_idx]
            
            # Build Gram matrix for training set
            x_train_sq = np.sum(x_train**2, axis=1)
            D_train = x_train_sq[:, np.newaxis] + x_train_sq[np.newaxis, :] - 2.0 * np.dot(x_train, x_train.T)
            np.maximum(D_train, 0, out=D_train)
            
            # Kernel matrix K = exp(-0.5 * D_train / sigma_rbf^2)
            K = np.exp(-0.5 * D_train / (sigma_rbf * sigma_rbf))
            
            # Regularized system: (K + alpha * I) w = y_train
            # Add a small ridge to ensure invertibility if K is singular
            reg = alpha * np.eye(len(x_train))
            A = K + reg
            b = y_train
            
            try:
                w = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                try:
                    w = np.linalg.lstsq(A, b, rcond=None)[0]
                except Exception:
                    return np.inf

            # Predict on validation set
            # K_val_train[i, j] = exp(-0.5 * (x_val[i] - x_train[j])^2 / sigma_rbf^2)
            x_val_sq = np.sum(x_val**2, axis=1)
            D_val_train = x_val_sq[:, np.newaxis] + x_train_sq[np.newaxis, :] - 2.0 * np.dot(x_val, x_train.T)
            np.maximum(D_val_train, 0, out=D_val_train)
            
            K_val = np.exp(-0.5 * D_val_train / (sigma_rbf * sigma_rbf))
            
            pred = K_val @ w
            mse = np.mean((pred - y_val)**2)
            mse_sum += mse
            
        return mse_sum / k_folds

    # Grid Search
    for sigma in sigma_candidates:
        if time.time() > deadline:
            break
        for alpha in alpha_candidates:
            if time.time() > deadline:
                break
            try:
                cv_mse = _cv_mse_rbf(sigma, alpha)
                score = cv_mse / y_var
                if score < best_score:
                    best_score = score
                    best_sigma = sigma
                    best_alpha = alpha
            except Exception:
                continue

    # Refine if time permits
    if time.time() < deadline - 10.0:
        sigma_refined = np.logspace(np.log10(max(0.01, best_sigma/2)), np.log10(min(20, best_sigma*2)), 10)
        alpha_refined = [best_alpha*0.5, best_alpha, best_alpha*2]
        
        for sigma in sigma_refined:
            if time.time() > deadline:
                break
            for alpha in alpha_refined:
                if time.time() > deadline:
                    break
                try:
                    cv_mse = _cv_mse_rbf(sigma, alpha)
                    score = cv_mse / y_var
                    if score < best_score:
                        best_score = score
                        best_sigma = sigma
                        best_alpha = alpha
                except Exception:
                    continue

    # Final Fit on all data
    # Use the precomputed D and x_norm
    K = np.exp(-0.5 * D / (best_sigma * best_sigma))
    reg = best_alpha * np.eye(n)
    A = K + reg
    b = y
    
    try:
        w = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        try:
            w = np.linalg.lstsq(A, b, rcond=None)[0]
        except Exception:
            w = np.zeros(n)

    # Now we have an RBF model: f(x) = sum_j w_j * exp(-0.5 * ((x - x_norm[j]) / best_sigma)^2)
    # We need to approximate this with a degree-5 polynomial in the ORIGINAL x basis.
    # f(x) = sum_{i=0}^5 c_i x^i
    
    # We can do this by evaluating the RBF model on a set of quadrature points or a grid
    # and then fitting a polynomial to those values.
    # Alternatively, we can analytically expand the Gaussian? No, that's infinite series.
    # Fitting a polynomial to the RBF output on a representative grid is robust.
    
    # Choose a grid in the original x domain.
    # The training data range is a good guide.
    x_min = np.min(x)
    x_max = np.max(x)
    
    # Extend range slightly to capture tails if needed, but keep within reasonable bounds
    # The polynomial is global, so it will extrapolate. We want it to match the RBF 
    # well within the support of the data.
    x_grid = np.linspace(x_min, x_max, 100)
    
    # Evaluate RBF on x_grid
    # x_grid_norm = (x_grid - mu_x) / sigma_x
    x_grid_norm = (x_grid - mu_x) / sigma_x
    
    # D_grid_centers[i, j] = (x_grid_norm[i] - x_norm[j])^2
    x_grid_sq = x_grid_norm**2
    D_grid = x_grid_sq[:, np.newaxis] + np.sum(x_norm**2, axis=1)[np.newaxis, :] - 2.0 * np.dot(x_grid_norm, x_norm.T)
    np.maximum(D_grid, 0, out=D_grid)
    
    K_grid = np.exp(-0.5 * D_grid / (best_sigma * best_sigma))
    y_rbf = K_grid @ w
    
    # Fit degree 5 polynomial to (x_grid, y_rbf)
    # Use least squares
    V = np.vander(x_grid, 6, increasing=True)
    try:
        c_poly, residuals, rank, sv = np.linalg.lstsq(V, y_rbf, rcond=None)
    except Exception:
        c_poly = np.zeros(6)
        
    # Ensure finite
    if not np.all(np.isfinite(c_poly)):
        c_poly = np.zeros(6)
        c_poly[0] = np.mean(y)

    return c_poly

