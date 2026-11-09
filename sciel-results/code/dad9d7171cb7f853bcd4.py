"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def fit(X_train, y_train):
    """
    Fits a regression model using RBF Kernel Ridge Regression.
    This approach is more flexible than polynomial regression for arbitrary hidden equations.
    """
    start_time = time.time()
    deadline = start_time + 250  # Safety margin before 300s limit
    
    X = np.asarray(X_train, dtype=float)
    y = np.asarray(y_train, dtype=float)
    
    n_samples, n_features = X.shape
    
    # 1. Normalize inputs to [0, 1] to make RBF scale invariant and stable
    min_vals = X.min(axis=0, keepdims=True)
    max_vals = X.max(axis=0, keepdims=True)
    range_vals = max_vals - min_vals
    # Avoid division by zero
    range_vals[range_vals == 0] = 1.0
    
    X_norm = (X - min_vals) / range_vals
    
    # 2. Compute RBF Kernel Matrix
    # K(x, x') = exp(-gamma * ||x - x'||^2)
    # We need to choose gamma. A common heuristic is 1 / (n_features * var(X)) or similar.
    # Since X is normalized to [0,1], variance is roughly 1/12 per feature.
    # Let's try a few gammas and pick the best one via simple validation or just a robust default.
    # Given the time constraint and "noiseless" data, we want high fidelity.
    # Small gamma -> wide kernel -> smoother. Large gamma -> narrow -> overfitting.
    # With 200 points, we can afford a reasonably narrow kernel.
    
    # Calculate squared distances
    # ||x - y||^2 = ||x||^2 + ||y||^2 - 2xy
    X_norm_sq = np.sum(X_norm ** 2, axis=1)
    sq_dists = X_norm_sq[:, None] + X_norm_sq[None, :] - 2 * (X_norm @ X_norm.T)
    # Clip small negative values due to floating point errors
    sq_dists = np.maximum(sq_dists, 0)
    
    # Try a small set of gammas to find one that fits well (low training error but not overfitting)
    # Since we don't have a validation set, we rely on the fact that the equations are likely smooth.
    # We will select gamma based on a simple heuristic: 1 / (n_features * median_dist)
    # Or just try a few values and pick the one with lowest training error (since data is noiseless, 
    # lower training error is generally better until it hits numerical instability).
    
    gammas = [1.0, 5.0, 10.0, 20.0, 50.0, 100.0]
    
    best_alpha = None
    best_train_mse = float('inf')
    best_gamma = 1.0
    best_K_inv = None
    
    # Regularization parameter
    # For noiseless data, we want strong interpolation but stable inverse.
    # Alpha controls regularization. Smaller alpha -> closer to interpolation.
    # Let's use a small alpha like 1e-6 or 1e-8.
    
    for gamma in gammas:
        if time.time() > deadline - 10:
            break
            
        K = np.exp(-gamma * sq_dists)
        
        # Add regularization
        # (K + alpha*I) alpha = y
        # We solve for coefficients
        try:
            # Use solve for speed and stability
            # K is 200x200, very fast
            K_reg = K + 1e-8 * np.eye(n_samples)
            
            # Solve K_reg @ alpha = y
            alpha = np.linalg.solve(K_reg, y)
            
            # Check training error
            y_pred = K @ alpha
            mse = np.mean((y_pred - y) ** 2)
            
            # We prefer lower MSE. Since data is noiseless, we want MSE ~ 0.
            # If multiple gammas give ~0 MSE, we might prefer the one with smaller norm of alpha (smoother)?
            # Or just the first one that fits well.
            if mse < best_train_mse:
                best_train_mse = mse
                best_alpha = alpha
                best_gamma = gamma
                best_K_inv = K_reg # Store for potential use, though we only need K for prediction
                
        except np.linalg.LinAlgError:
            continue

    if best_alpha is None:
        # Fallback to polynomial if kernel fails
        def features(X):
            cols = [np.ones(len(X))] + [X[:, j] for j in range(X.shape[1])] + [X[:, i] * X[:, j] for i in range(X.shape[1]) for j in range(i, X.shape[1])]
            return np.column_stack(cols)
        w, *_ = np.linalg.lstsq(features(X_train), y_train, rcond=None)
        return lambda X: features(X) @ w

    # Capture variables for closure
    final_alpha = best_alpha
    final_gamma = best_gamma
    min_vals_closure = min_vals
    range_vals_closure = range_vals
    X_train_norm = X_norm
    
    def predict(X):
        X = np.asarray(X, dtype=float)
        X_norm_test = (X - min_vals_closure) / range_vals_closure
        
        # Compute kernel between test points and training points
        # K_test(x, x_i) = exp(-gamma * ||x - x_i||^2)
        X_test_sq = np.sum(X_norm_test ** 2, axis=1)
        X_train_sq = np.sum(X_train_norm ** 2, axis=1)
        sq_dists_test = X_test_sq[:, None] + X_train_sq[None, :] - 2 * (X_norm_test @ X_train_norm.T)
        sq_dists_test = np.maximum(sq_dists_test, 0)
        
        K_test = np.exp(-final_gamma * sq_dists_test)
        
        return K_test @ final_alpha

    return predict
# EVOLVE-BLOCK-END
