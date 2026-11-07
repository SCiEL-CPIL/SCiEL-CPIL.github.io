import numpy as np
import time
import sys

def fit(X_train, y_train):
    """
    Fits a regression model using Forward Feature Selection on a rich basis.
    """
    start_time = time.time()
    deadline = start_time + 290  # Safety margin for 330s total limit
    
    X = X_train
    y = y_train
    n_samples, n_features = X.shape
    
    # Store info needed for basis generation
    x_min = np.min(X, axis=0)
    x_max = np.max(X, axis=0)
    x_mid = (x_min + x_max) / 2.0
    x_spread = np.maximum(1e-6, (x_max - x_min) / 3.0)
    
    # 1. Generate Candidate Feature Library
    basis = [np.ones(n_samples)]  # Bias term
    
    # Polynomial features up to degree 4
    for d in range(1, 5):
        for i in range(n_features):
            basis.append(X[:, i] ** d)
        
        if d == 2:
            for i in range(n_features):
                for j in range(i, n_features):
                    if i != j:
                        basis.append(X[:, i] * X[:, j])
        elif d == 3:
            for i in range(n_features):
                for j in range(i, n_features):
                    for k in range(j, n_features):
                        if len({i, j, k}) == 3:
                            basis.append(X[:, i] * X[:, j] * X[:, k])
    
    # Non-linear features
    for i in range(n_features):
        xi = X[:, i]
        # Trig
        basis.append(np.sin(xi))
        basis.append(np.cos(xi))
        # Log (safe as inputs >= 1)
        basis.append(np.log(xi))
        # Inverse
        basis.append(1.0 / xi)
        # Square root
        basis.append(np.sqrt(xi))
        # tanh
        basis.append(np.tanh(xi))
        # exp (clamped to avoid overflow)
        basis.append(np.exp(np.clip(xi - x_max[i], -50, 50)))
        # Rational: xi / (xi + c) for a few c values
        for c in [0.5, 1.0, 2.0]:
            basis.append(xi / (xi + c))
        
        # Gaussian RBFs centered at min, mid, max of the feature
        centers = [x_min[i], x_mid[i], x_max[i]]
        for c in centers:
            basis.append(np.exp(-((xi - c) ** 2) / (2 * x_spread[i] ** 2)))
    
    # Cross-feature ratio features
    for i in range(n_features):
        for j in range(n_features):
            if i != j:
                basis.append(X[:, i] / (X[:, i] + X[:, j] + 1e-6))
    
    # Check for NaNs or Infs
    for idx, b in enumerate(basis):
        if not np.all(np.isfinite(b)):
            b[np.isnan(b)] = 0
            b[np.isinf(b)] = 0
            basis[idx] = b
             
    Feature_Matrix = np.column_stack(basis)
    n_candidates = Feature_Matrix.shape[1]
    
    # 2. Forward Selection with efficient Gram matrix updates
    selected_indices = [0]  # Start with bias
    current_mse = np.inf
    max_features = 10  # Limit complexity
    min_improvement = 1e-8
    
    # Pre-compute Gram matrix and cross-correlations
    Gram = Feature_Matrix.T @ Feature_Matrix  # n_candidates x n_candidates
    cross = Feature_Matrix.T @ y  # n_candidates x 1
    
    def get_ridge_coefficients_from_gram(Gram_sub, cross_sub, alpha=1.0):
        n_sub = Gram_sub.shape[0]
        reg = np.eye(n_sub) * alpha
        reg[0, 0] = 0  # Don't regularize bias
        try:
            w = np.linalg.solve(Gram_sub + reg, cross_sub)
        except np.linalg.LinAlgError:
            w = np.linalg.lstsq(Feature_Matrix[:, selected_indices], y, rcond=None)[0]
        return w
    
    def compute_mse_from_pred(y, pred):
        return np.mean((y - pred) ** 2)
    
    # Initial fit with just bias
    Gram_sub = Gram[np.ix_(selected_indices, selected_indices)]
    cross_sub = cross[selected_indices]
    w_current = get_ridge_coefficients_from_gram(Gram_sub, cross_sub, alpha=1.0)
    pred_current = Feature_Matrix[:, selected_indices] @ w_current
    current_mse = compute_mse_from_pred(y, pred_current)
    
    for step in range(max_features):
        if time.time() > deadline - 10:
            break
            
        best_mse = current_mse
        best_idx = -1
        best_w = w_current
        best_pred = pred_current
        
        # Evaluate adding each candidate feature
        for i in range(n_candidates):
            if i in selected_indices:
                continue
                
            # Time check inside loop
            if time.time() > deadline - 5:
                break
                
            new_indices = selected_indices + [i]
            
            # Build new Gram submatrix efficiently
            new_Gram_sub = Gram[np.ix_(new_indices, new_indices)]
            new_cross_sub = cross[new_indices]
            
            w_new = get_ridge_coefficients_from_gram(new_Gram_sub, new_cross_sub, alpha=1.0)
            pred_new = Feature_Matrix[:, new_indices] @ w_new
            mse_new = compute_mse_from_pred(y, pred_new)
            
            if mse_new < best_mse - min_improvement:
                best_mse = mse_new
                best_idx = i
                best_w = w_new
                best_pred = pred_new
        
        if best_idx != -1:
            selected_indices.append(best_idx)
            w_current = best_w
            pred_current = best_pred
            current_mse = best_mse
        else:
            break
            
    # 3. Final Fit
    # Use a slightly stronger regularization for the final model to prevent overfitting
    X_final = Feature_Matrix[:, selected_indices]
    final_alpha = 2.0
    
    def get_ridge_coefficients_direct(X_sel, y, alpha):
        XtX = X_sel.T @ X_sel
        Xty = X_sel.T @ y
        reg = np.eye(XtX.shape[0]) * alpha
        reg[0, 0] = 0
        try:
            w = np.linalg.solve(XtX + reg, Xty)
        except np.linalg.LinAlgError:
            w = np.linalg.lstsq(X_sel, y, rcond=None)[0]
        return w
    
    w_final = get_ridge_coefficients_direct(X_final, y, final_alpha)
    
    # Store selected indices and weights
    sel_idx = np.array(selected_indices)
    
    def predict(X_test):
        n_test = X_test.shape[0]
        X_t = X_test
        b_list = [np.ones(n_test)]
        
        # Polynomial features up to degree 4
        for d in range(1, 5):
            for i in range(X_t.shape[1]):
                b_list.append(X_t[:, i] ** d)
            if d == 2:
                for i in range(X_t.shape[1]):
                    for j in range(i, X_t.shape[1]):
                        if i != j:
                            b_list.append(X_t[:, i] * X_t[:, j])
            elif d == 3:
                for i in range(X_t.shape[1]):
                    for j in range(i, X_t.shape[1]):
                        for k in range(j, X_t.shape[1]):
                            if len({i, j, k}) == 3:
                                b_list.append(X_t[:, i] * X_t[:, j] * X_t[:, k])
                                
        for i in range(X_t.shape[1]):
            xi = X_t[:, i]
            b_list.append(np.sin(xi))
            b_list.append(np.cos(xi))
            b_list.append(np.log(xi))
            b_list.append(1.0 / xi)
            b_list.append(np.sqrt(xi))
            b_list.append(np.tanh(xi))
            b_list.append(np.exp(np.clip(xi - x_max[i], -50, 50)))
            for c in [0.5, 1.0, 2.0]:
                b_list.append(xi / (xi + c))
            
            centers = [x_min[i], x_mid[i], x_max[i]]
            for c in centers:
                b_list.append(np.exp(-((xi - c) ** 2) / (2 * x_spread[i] ** 2)))
        
        # Cross-feature ratio features
        for i in range(X_t.shape[1]):
            for j in range(X_t.shape[1]):
                if i != j:
                    b_list.append(X_t[:, i] / (X_t[:, i] + X_t[:, j] + 1e-6))
        
        # Check for NaNs or Infs
        for idx, b in enumerate(b_list):
            if not np.all(np.isfinite(b)):
                b[np.isnan(b)] = 0
                b[np.isinf(b)] = 0
                b_list[idx] = b
        
        Feature_Matrix_test = np.column_stack(b_list)
        
        X_sel_test = Feature_Matrix_test[:, sel_idx]
        pred = X_sel_test @ w_final
        return pred

    return predict
