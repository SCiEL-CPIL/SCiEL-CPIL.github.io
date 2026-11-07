import numpy as np
import time

def fit(X_train, y_train):
    start_time = time.time()
    timeout = 30.0  # Per equation budget, leaving time for 10 equations + overhead
    
    n_samples, n_features = X_train.shape
    
    # Normalize inputs to [0, 1] for stable basis generation
    X_min = X_train.min(axis=0)
    X_max = X_train.max(axis=0)
    # Avoid division by zero
    X_range = X_max - X_min
    X_range[X_range == 0] = 1.0
    
    def normalize(X):
        return (X - X_min) / X_range

    X_norm_train = normalize(X_train)
    
    # Generate Basis Functions
    # We create a list of functions that take X_norm (N, d) and return (N,)
    basis_funcs = []
    basis_funcs.append(lambda X: np.ones(X.shape[0])) # Constant
    
    # Linear
    for j in range(n_features):
        basis_funcs.append(lambda X, j=j: X[:, j])
        
    # Quadratic and Interactions
    for i in range(n_features):
        for j in range(i, n_features):
            basis_funcs.append(lambda X, i=i, j=j: X[:, i] * X[:, j])
            
    # Cubic
    for j in range(n_features):
        basis_funcs.append(lambda X, j=j: X[:, j] ** 3)
        
    # Trig (scaled by pi to have one period in [0,1])
    for j in range(n_features):
        basis_funcs.append(lambda X, j=j: np.sin(np.pi * X[:, j]))
        basis_funcs.append(lambda X, j=j: np.cos(np.pi * X[:, j]))
        
    # Exp
    for j in range(n_features):
        basis_funcs.append(lambda X, j=j: np.exp(-X[:, j]))
        
    # Log
    for j in range(n_features):
        basis_funcs.append(lambda X, j=j: np.log1p(X[:, j])) # log(1+x)

    # Precompute basis matrix for training
    # Note: This might be large if n_features is high, but d is small (1-3)
    # Max basis size: 1 + d + d(d+1)/2 + d + 2d + d + d = 1 + d + d^2+d/2 + d + 4d = ~ 6.5d^2 + 7.5d
    # For d=3: 1+3+6+3+6+3+3 = 25 features. Very small.
    
    Phi_train = np.column_stack([f(X_norm_train) for f in basis_funcs])
    
    # Handle any NaN/Inf from basis generation
    Phi_train = np.nan_to_num(Phi_train, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Use OMP to select features
    # We'll select up to min(len(basis_funcs), 15) features
    max_features = min(len(basis_funcs), 15)
    
    selected_indices = []
    residual = y_train.copy()
    projected_y = np.zeros(n_samples)
    
    # Precompute Gram matrix of basis? Or just use correlations.
    # For small basis, re-fitting is fast.
    
    for _ in range(max_features):
        if time.time() - start_time > timeout:
            break
            
        # Find feature with max correlation to residual
        # corr = dot(Phi[:, j], residual) / (norm(Phi[:, j]) * norm(residual))
        # Since norm(residual) is constant, we just maximize dot(Phi[:, j], residual)
        
        if len(selected_indices) == 0:
            # First feature: max correlation with y
            # Actually, standard OMP starts with max correlation to y.
            # But we are tracking residual. Initially residual = y.
            # So we pick feature with max dot(Phi[:, j], residual)
            
            scores = Phi_train.T @ residual
            # Normalize by feature norm to be fair? 
            # Standard OMP uses correlation.
            norms = np.linalg.norm(Phi_train, axis=0)
            norms[norms == 0] = 1e-9
            scores = scores / norms
            # Also divide by norm of residual? It's constant, so doesn't affect argmax.
            
            best_idx = np.argmax(np.abs(scores))
        else:
            # Regress residual on currently selected features
            # This is equivalent to projecting y onto the space of selected features
            # and taking the residual
            # We can just re-fit the whole thing on selected features
            # But for OMP, we update the projection.
            # Simpler: Just pick the feature that reduces the residual the most?
            # That's forward selection.
            # Let's stick to a simple Forward Selection with Ridge fit at the end.
            # It's robust and easy to implement correctly.
            
            pass # We will use forward selection logic below instead of strict OMP update for simplicity and robustness

    # Let's implement Forward Selection with a stopping criterion
    selected_indices = []
    best_mse = np.inf
    
    # Sort features by individual correlation with y? Optional but helps.
    # For now, just iterate all.
    
    for step in range(max_features):
        if time.time() - start_time > timeout:
            break
            
        best_idx = -1
        best_score = -np.inf
        
        # We evaluate adding each candidate feature to the current set
        # To speed up, we can precompute the projection of y onto the current set
        # and the Gram matrix of the current set.
        
        if len(selected_indices) == 0:
            # Pick feature with max |corr| with y
            norms = np.linalg.norm(Phi_train, axis=0)
            norms[norms == 0] = 1e-9
            scores = (Phi_train.T @ y_train) / norms
            best_idx = np.argmax(np.abs(scores))
        else:
            # Compute residual after fitting current selected features
            # Use Ridge for stability
            X_sel = Phi_train[:, selected_indices]
            # Simple LSQ for speed, Ridge later
            w, *_ = np.linalg.lstsq(X_sel, y_train, rcond=None)
            pred = X_sel @ w
            residual = y_train - pred
            
            # Find best new feature
            # Maximize |dot(Phi[:, j], residual)| / norm(Phi[:, j])
            norms = np.linalg.norm(Phi_train, axis=0)
            norms[norms == 0] = 1e-9
            scores = (Phi_train.T @ residual) / norms
            # Exclude already selected
            for j in selected_indices:
                scores[j] = -np.inf
            best_idx = np.argmax(scores)
            
            # Check improvement
            # Add best_idx to a temp set and check MSE
            temp_set = selected_indices + [best_idx]
            X_temp = Phi_train[:, temp_set]
            w_temp, *_ = np.linalg.lstsq(X_temp, y_train, rcond=None)
            pred_temp = X_temp @ w_temp
            mse_temp = np.mean((pred_temp - y_train) ** 2)
            
            if mse_temp < best_mse * 0.99: # 1% improvement
                selected_indices = temp_set
                best_mse = mse_temp
            else:
                break # No significant improvement
            
    # If no features selected, select the best single one
    if not selected_indices:
        norms = np.linalg.norm(Phi_train, axis=0)
        norms[norms == 0] = 1e-9
        scores = (Phi_train.T @ y_train) / norms
        selected_indices = [np.argmax(np.abs(scores))]
        
    # Final Fit with Ridge Regression on selected features
    X_sel_train = Phi_train[:, selected_indices]
    alpha = 1e-5
    # Ridge: (X^T X + alpha I)^-1 X^T y
    try:
        A = X_sel_train.T @ X_sel_train + alpha * np.eye(len(selected_indices))
        b = X_sel_train.T @ y_train
        w_final = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        w_final, *_ = np.linalg.lstsq(X_sel_train, y_train, rcond=None)
        
    # Define predict
    def predict(X):
        X_norm = normalize(X)
        # Generate basis for test set
        Phi_test = np.column_stack([f(X_norm) for f in basis_funcs])
        Phi_test = np.nan_to_num(Phi_test, nan=0.0, posinf=1e6, neginf=-1e6)
        X_sel_test = Phi_test[:, selected_indices]
        return X_sel_test @ w_final

    return predict
