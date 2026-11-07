"""Symbolic regression via Adaptive Kernel Ridge Regression (Ensemble) with deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import operator
from scipy.optimize import minimize

def fit(X_train, y_train):
    start_time = time.time()
    # 300s limit. We reserve 30s for safety/overhead, leaving 270s for computation.
    deadline = start_time + 270.0
    
    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64).flatten()
    n_samples, n_features = X.shape
    
    # Normalize inputs to [0, 1] for numerical stability and kernel scaling
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    X_range = X_max - X_min
    X_range[X_range == 0] = 1.0
    X_norm = (X - X_min) / X_range
    
    # Standardize y for numerical stability in kernel computations
    y_mean = y.mean()
    y_std = y.std()
    if y_std == 0:
        y_std = 1.0
    y_norm = (y - y_mean) / y_std

    # Strategy: Kernel Ridge Regression with Adaptive Hyperparameters
    # We use a composite kernel: RBF + Polynomial + Sine
    # This allows capturing smooth, polynomial-like, and periodic structures.
    
    def rbf_kernel(X1, X2, length_scale, sigma):
        # Squared Euclidean distance
        sq_dists = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * X1 @ X2.T
        sq_dists = np.maximum(sq_dists, 0)
        return sigma * np.exp(-sq_dists / (2 * length_scale**2))
    
    def poly_kernel(X1, X2, degree, gamma, coef0):
        return (gamma * (X1 @ X2.T) + coef0) ** degree
    
    def sine_kernel(X1, X2, period):
        # Sine kernel: sin(2*pi*x1*x2/period)
        # Note: Standard sine kernel is often sin(2*pi * <x1,x2> / period)
        # For univariate or low-dim, this can capture oscillations.
        # However, for general regression, RBF is often more robust.
        # We will primarily rely on RBF and Poly, adding Sine only if it helps.
        # To keep it simple and robust, let's stick to RBF + Poly for now, 
        # but make the weights adaptive.
        return np.sin(2 * np.pi * (X1 @ X2.T) / period)

    # Composite Kernel: K = w_rbf * K_rbf + w_poly * K_poly + w_sine * K_sine
    # Hyperparameters: length_scale (RBF), degree (Poly), gamma (Poly), coef0 (Poly), 
    # weights w_rbf, w_poly, w_sine, and regularization alpha.
    
    # Since data is noiseless, we want low regularization (small alpha) to fit training data well,
    # but not zero to avoid numerical instability if the kernel matrix is ill-conditioned.
    
    def compute_kernel(X1, X2, params):
        # params: [length_scale, gamma, coef0, w_rbf, w_poly, w_sine, alpha]
        ls = params[0]
        gamma = params[1]
        coef0 = params[2]
        w_rbf = params[3]
        w_poly = params[4]
        w_sine = params[5]
        alpha = params[6]
        
        # Ensure positive values for scale parameters
        ls = max(ls, 1e-4)
        gamma = max(gamma, 1e-4)
        coef0 = max(coef0, 0.0)
        w_rbf = max(w_rbf, 1e-6)
        w_poly = max(w_poly, 1e-6)
        w_sine = max(w_sine, 1e-6)
        alpha = max(alpha, 1e-8)
        
        K_rbf = rbf_kernel(X1, X2, ls, 1.0)
        K_poly = poly_kernel(X1, X2, 3, gamma, coef0) # Fixed degree 3 for simplicity, gamma/coef0 adaptive
        # Skip sine for now to reduce complexity, RBF+Poly is strong
        # K_sine = sine_kernel(X1, X2, 1.0)
        
        K = w_rbf * K_rbf + w_poly * K_poly
        # If we included sine: K += w_sine * K_sine
        
        return K, alpha

    # We will perform a grid search over a few hyperparameter combinations.
    # Given 200 samples, O(N^3) solve is ~8 million ops, very fast.
    # We can afford to try ~50-100 configurations.
    
    # Candidate hyperparameters
    length_scales = [0.1, 0.2, 0.5, 1.0]
    gammas = [0.1, 1.0, 10.0]
    coef0s = [0.0, 1.0]
    alphas = [1e-6, 1e-8, 1e-10] # Low regularization for noiseless data
    
    # Weights: We can fix w_rbf=1, w_poly=1 and just scale alpha, 
    # or try a few weight combinations.
    # Let's try a few weight ratios.
    weight_combos = [
        (1.0, 1.0), # RBF : Poly
        (1.0, 0.1), # RBF dominant
        (0.1, 1.0), # Poly dominant
    ]
    
    best_alpha = None
    best_K_inv_y = None
    best_train_mse = np.inf
    best_params = None
    
    # Precompute X_norm @ X_norm.T for poly kernel speedup
    X_norm_T = X_norm.T
    X_norm_dot = X_norm @ X_norm_T
    
    # Loop through candidates
    for ls in length_scales:
        if time.time() > deadline - 5:
            break
        for gamma in gammas:
            if time.time() > deadline - 5:
                break
            for coef0 in coef0s:
                if time.time() > deadline - 5:
                    break
                for w_rbf, w_poly in weight_combos:
                    if time.time() > deadline - 5:
                        break
                    for alpha in alphas:
                        if time.time() > deadline - 5:
                            break
                        
                        try:
                            # Compute Kernel
                            K_rbf = rbf_kernel(X_norm, X_norm, ls, 1.0)
                            K_poly = (gamma * X_norm_dot + coef0) ** 3
                            K = w_rbf * K_rbf + w_poly * K_poly
                            
                            # Add regularization
                            K_reg = K + alpha * np.eye(n_samples)
                            
                            # Solve K_reg @ alpha_vec = y_norm
                            # Using Cholesky for stability
                            try:
                                L = np.linalg.cholesky(K_reg)
                                # Solve L L^T x = y
                                # L y_temp = y
                                y_temp = np.linalg.solve(L, y_norm)
                                # L^T x = y_temp
                                alpha_vec = np.linalg.solve(L.T, y_temp)
                            except np.linalg.LinAlgError:
                                # Fallback to lstsq if Cholesky fails
                                alpha_vec = np.linalg.lstsq(K_reg, y_norm, rcond=None)[0]
                            
                            # Calculate training MSE
                            pred_train = K @ alpha_vec
                            mse = np.mean((pred_train - y_norm)**2)
                            
                            if mse < best_train_mse:
                                best_train_mse = mse
                                best_alpha = alpha_vec
                                best_params = (ls, gamma, coef0, w_rbf, w_poly, alpha)
                                # Store K for prediction? No, we need to compute K_test @ K_train_inv_y
                                # We can store the inverse or the solution vector.
                                # For prediction: f(x_new) = K(x_new, X_train) @ alpha_vec
                                # So we just need alpha_vec and the hyperparams to compute K(x_new, X_train).
                                # But wait, alpha_vec depends on K_train.
                                # So we store alpha_vec and the hyperparams used to generate K_train.
                                # We do NOT need to store K_train itself, just the params to recompute K_test_train.
                                
                        except Exception as e:
                            continue

    if best_alpha is None:
        # Fallback: Simple linear regression or constant
        # If all failed, use mean prediction
        def predict(X_test):
            return np.full(X_test.shape[0], y_mean)
        return predict

    # Extract best hyperparameters for prediction
    best_ls, best_gamma, best_coef0, best_w_rbf, best_w_poly, best_alpha_val = best_params
    
    def predict(X_test):
        X_test = np.asarray(X_test, dtype=np.float64)
        # Normalize test data using train stats
        X_test_norm = (X_test - X_min) / X_range
        
        # Compute K_test_train
        # K_rbf_test_train
        sq_dists = np.sum(X_test_norm**2, axis=1)[:, None] + np.sum(X_norm**2, axis=1)[None, :] - 2 * X_test_norm @ X_norm.T
        sq_dists = np.maximum(sq_dists, 0)
        K_rbf_tt = best_w_rbf * np.exp(-sq_dists / (2 * best_ls**2))
        
        # K_poly_test_train
        K_poly_tt = best_w_poly * (best_gamma * (X_test_norm @ X_norm.T) + best_coef0) ** 3
        
        K_tt = K_rbf_tt + K_poly_tt
        
        # Prediction
        pred_norm = K_tt @ best_alpha
        pred = pred_norm * y_std + y_mean
        
        return pred

    return predict
# EVOLVE-BLOCK-END
