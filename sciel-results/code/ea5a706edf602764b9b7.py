import numpy as np
import time

def fit(X_train, y_train):
    """
    Fits a Kernel Ridge Regression model with a composite kernel 
    (Linear + Constant + RBF) to the training data.
    Uses Leave-One-Out Cross-Validation to select hyperparameters.
    """
    t0 = time.time()
    DEADLINE = 325.0  # Safety margin before 330s kill
    
    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    n, d = X.shape
    
    # Standardize features for better kernel scaling
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd < 1e-12] = 1.0
    X_std = (X - mu) / sd
    
    # Define hyperparameter grid
    # Length scales for RBF. 
    # Median distance heuristic is often a good start.
    # We'll use a fixed grid based on standardized space.
    # Typical length scales in standardized space: 0.1 to 5.0
    length_scales = np.array([0.1, 0.3, 0.5, 1.0, 2.0, 5.0])
    # Regularization parameters
    # Lambda controls the trade-off between fit and smoothness.
    # For n=200, typical range 1e-4 to 1e2
    lambdas = np.array([1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0])
    
    best_score = np.inf
    best_params = (1.0, 1.0) # default length_scale, lambda
    
    # Pre-compute squared distances for RBF kernel? 
    # No, we need to recompute for each length scale.
    
    # Function to compute Gram matrix for given length scale
    def compute_kernel(X_a, X_b, l):
        # RBF part: exp(-||x-y||^2 / (2*l^2))
        # Linear part: x.T y
        # Constant part: 1
        
        # Efficient distance calculation
        # ||x-y||^2 = ||x||^2 + ||y||^2 - 2 x.y
        sq_norms_a = np.sum(X_a**2, axis=1)[:, np.newaxis]
        sq_norms_b = np.sum(X_b**2, axis=1)[np.newaxis, :]
        sq_dists = sq_norms_a + sq_norms_b - 2 * (X_a @ X_b.T)
        # Clip small negative values due to floating point errors
        sq_dists = np.maximum(sq_dists, 0)
        
        rbf_part = np.exp(-sq_dists / (2 * l**2))
        linear_part = X_a @ X_b.T
        const_part = np.ones_like(rbf_part)
        
        return rbf_part + linear_part + const_part

    # LOOCV Loop
    for l in length_scales:
        if time.time() > DEADLINE:
            break
        K_train = compute_kernel(X_std, X_std, l)
        for lam in lambdas:
            if time.time() > DEADLINE:
                break
            
            # Solve for alpha: (K + lam*I) alpha = y
            # For LOOCV, we need the diagonal of the inverse matrix.
            # inv_K = np.linalg.inv(K_train + lam * np.eye(n))
            # Using solve is more stable than inv, but we need diagonal of inverse.
            # We can solve (K + lam*I) X = I to get inverse, or use Cholesky.
            # Given n=200, direct inverse is fine.
            
            A = K_train + lam * np.eye(n)
            try:
                # Calculate inverse
                A_inv = np.linalg.inv(A)
            except np.linalg.LinAlgError:
                continue
            
            # LOOCV predictions: 
            # y_hat_i = y_i - 1 / A_inv[i,i]
            diag_inv = np.diag(A_inv)
            # Avoid division by zero
            diag_inv[diag_inv < 1e-12] = 1e-12
            
            y_hat = y - 1.0 / diag_inv
            
            # Compute LOOCV MSE
            mse = np.mean((y_hat - y)**2)
            
            if mse < best_score:
                best_score = mse
                best_params = (l, lam)
                
    l_opt, lam_opt = best_params
    
    # Final Fit with best parameters on all training data
    K_final = compute_kernel(X_std, X_std, l_opt)
    A_final = K_final + lam_opt * np.eye(n)
    
    try:
        alpha = np.linalg.solve(A_final, y)
    except np.linalg.LinAlgError:
        # Fallback to least squares if singular (unlikely with regularization)
        alpha, *_ = np.linalg.lstsq(A_final, y, rcond=None)
        
    def predict(X_test):
        X_test = np.asarray(X_test, dtype=np.float64)
        X_test_std = (X_test - mu) / sd
        K_test = compute_kernel(X_test_std, X_std, l_opt)
        return K_test @ alpha

    return predict
