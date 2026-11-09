"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def fit(X_train, y_train):
    """
    Fits a regression model using Sparse Gaussian Process (Nystrom Approximation)
    with automatic bandwidth selection via internal validation.
    
    Strategy:
    1. Normalize inputs.
    2. Select a subset of points (inducing points) for sparse approximation.
    3. Search over a log-spaced range of RBF bandwidths (length-scales).
    4. For each bandwidth, fit the sparse model and evaluate on a held-out validation set.
    5. Select the bandwidth with the lowest validation MSE.
    6. Refit on all data with the best bandwidth and return a predict function.
    """
    start_time = time.time()
    deadline = start_time + 280  # Safety margin before 300s limit
    
    X = np.asarray(X_train, dtype=float)
    y = np.asarray(y_train, dtype=float)
    
    n_samples, n_features = X.shape
    
    # 1. Normalize inputs to [0, 1]
    min_vals = X.min(axis=0, keepdims=True)
    max_vals = X.max(axis=0, keepdims=True)
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1.0
    
    X_norm = (X - min_vals) / range_vals
    
    # 2. Split into train and validation sets for model selection
    # Use 80% train, 20% val for selection. With 200 points, this is 160/40.
    # Shuffle to ensure random split
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    val_size = int(0.2 * n_samples)
    val_idx = indices[:val_size]
    tr_idx = indices[val_size:]
    
    X_tr = X_norm[tr_idx]
    y_tr = y[tr_idx]
    X_val = X_norm[val_idx]
    y_val = y[val_idx]
    
    n_tr = len(y_tr)
    
    # 3. Define a set of bandwidths (length-scales) to try.
    # gamma = 1 / (2 * l^2). 
    # For normalized data in [0,1]^d, typical distances are O(1).
    # l (length scale) should be comparable to the feature scale.
    # Let's try l in [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
    # This corresponds to gamma = 1/(2*l^2)
    # l=0.1 -> gamma=50
    # l=0.2 -> gamma=12.5
    # l=0.5 -> gamma=2.0
    # l=1.0 -> gamma=0.5
    # l=2.0 -> gamma=0.125
    # l=5.0 -> gamma=0.02
    lengths = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
    gammas = [1.0 / (2.0 * l**2) for l in lengths]
    
    # 4. Sparse approximation: Use a subset of training points as inducing points.
    # To keep it fast and stable, we use a fraction of the training points.
    # If n_tr is large, we might subsample. Here n_tr ~ 160, so we can use all or a subset.
    # Using all points for the "sparse" set makes it equivalent to full GP, but 
    # the validation loop is the key structural change.
    # Let's use a subset of size m = min(n_tr, 50) to ensure speed and regularity.
    # Actually, with n=200, full inversion is fast. The main benefit here is the *selection*.
    # Let's stick to full kernel on the training subset for accuracy, as N=160 is small.
    
    best_val_mse = float('inf')
    best_gamma = 1.0
    
    # Pre-compute squared distances for training set to speed up kernel computation?
    # K(x, x') = exp(-gamma * ||x-x'||^2)
    # We need to compute K for (X_tr, X_tr) and (X_val, X_tr) for each gamma.
    
    # Optimization: Compute squared distances once.
    # sq_dists_tr = ||X_tr[:,i] - X_tr[:,j]||^2
    # sq_dists_val_tr = ||X_val[:,i] - X_tr[:,j]||^2
    
    # Helper to compute sq dists
    def compute_sq_dists(A, B):
        A_sq = np.sum(A**2, axis=1, keepdims=True)
        B_sq = np.sum(B**2, axis=1, keepdims=True)
        sq_dists = A_sq + B_sq.T - 2 * (A @ B.T)
        return np.maximum(sq_dists, 0)
    
    sq_dists_tr_tr = compute_sq_dists(X_tr, X_tr)
    sq_dists_val_tr = compute_sq_dists(X_val, X_tr)
    
    # Regularization for stability
    reg = 1e-8
    
    for gamma in gammas:
        if time.time() > deadline - 20:
            break
            
        try:
            # Compute Kernel on Training Subset
            K_tr = np.exp(-gamma * sq_dists_tr_tr)
            K_tr += reg * np.eye(n_tr)
            
            # Solve for coefficients on training subset
            alpha = np.linalg.solve(K_tr, y_tr)
            
            # Predict on Validation Set
            K_val_tr = np.exp(-gamma * sq_dists_val_tr)
            y_val_pred = K_val_tr @ alpha
            
            val_mse = np.mean((y_val_pred - y_val)**2)
            
            if val_mse < best_val_mse:
                best_val_mse = val_mse
                best_gamma = gamma
                
        except np.linalg.LinAlgError:
            continue
    
    # 5. Refit on ALL data with the best gamma
    # Now use the full X_norm and y
    n_full = len(y)
    
    # Recompute sq dists for full set
    sq_dists_full = compute_sq_dists(X_norm, X_norm)
    
    K_full = np.exp(-best_gamma * sq_dists_full)
    K_full += reg * np.eye(n_full)
    
    try:
        alpha_full = np.linalg.solve(K_full, y)
    except np.linalg.LinAlgError:
        # Fallback if full solve fails (unlikely with reg)
        # Use the subset alpha and predict via subset kernel
        # This is a degraded mode, but better than crashing
        alpha_full = alpha # from the last successful iteration in loop, might be mismatched size if loop broke early
        # Actually, if we broke early or failed, we should handle this.
        # Let's just use the last valid alpha from the loop if available, otherwise fallback to poly
        # To be safe, if this fails, we return a simple polynomial
        def features(X):
            cols = [np.ones(len(X))] + [X[:, j] for j in range(X.shape[1])] + [X[:, i] * X[:, j] for i in range(X.shape[1]) for j in range(i, X.shape[1])]
            return np.column_stack(cols)
        w, *_ = np.linalg.lstsq(features(X_train), y_train, rcond=None)
        return lambda X: features(X) @ w

    # Capture variables for closure
    final_alpha = alpha_full
    final_gamma = best_gamma
    min_vals_closure = min_vals
    range_vals_closure = range_vals
    X_train_norm = X_norm

    def predict(X):
        X = np.asarray(X, dtype=float)
        X_norm_test = (X - min_vals_closure) / range_vals_closure
        
        # Compute kernel between test points and training points
        # K_test(x, x_i) = exp(-gamma * ||x - x_i||^2)
        X_test_sq = np.sum(X_norm_test ** 2, axis=1, keepdims=True)
        X_train_sq = np.sum(X_train_norm ** 2, axis=1, keepdims=True)
        sq_dists_test = X_test_sq + X_train_sq.T - 2 * (X_norm_test @ X_train_norm.T)
        sq_dists_test = np.maximum(sq_dists_test, 0)
        
        K_test = np.exp(-final_gamma * sq_dists_test)
        
        return K_test @ final_alpha

    return predict
# EVOLVE-BLOCK-END
