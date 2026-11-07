"""Improved program using Forward Selection on an expanded symbolic feature basis with normalization."""
# EVOLVE-BLOCK-START
import numpy as np

def fit(X_train, y_train):
    """
    Fits a regression model using Forward Selection on an expanded set of symbolic features.
    Includes feature normalization for numerical stability.
    """
    X = X_train
    y = y_train
    n_samples = X.shape[0]
    d = X.shape[1]
    
    x = X[:, 0] if d > 0 else np.zeros(n_samples)
    
    # Define a rich set of candidate features
    eps = 1e-8
    
    # Helper to generate features for a given x vector
    def generate_features(x_vec):
        feats = []
        # Polynomials
        for k in range(1, 9):
            feats.append(x_vec ** k)
            
        # Trigonometric
        for k in range(1, 5):
            feats.append(np.sin(k * x_vec))
            feats.append(np.cos(k * x_vec))
            
        # Hyperbolic
        feats.append(np.tanh(x_vec))
        feats.append(np.sinh(np.clip(x_vec, -10, 10)))
        feats.append(np.cosh(np.clip(x_vec, -10, 10)))
        # Safe sech and csch
        sech = 1.0 / np.cosh(np.clip(x_vec, -10, 10))
        csch = 1.0 / (np.sinh(np.clip(x_vec, -10, 10)) + np.where(np.sinh(np.clip(x_vec, -10, 10)) >= 0, eps, -eps))
        feats.append(sech)
        feats.append(csch)
        
        # Exponential (clipped to prevent overflow)
        x_clip = np.clip(x_vec, -10, 10)
        feats.append(np.exp(x_clip))
        feats.append(np.exp(-x_clip))
        feats.append(np.exp(np.clip(x_vec**2, -10, 10)))
        
        # Logarithmic
        feats.append(np.log(np.abs(x_vec) + eps))
        feats.append(np.log(x_vec**2 + eps))
        
        # Rational
        feats.append(1.0 / (x_vec + np.where(x_vec >= 0, eps, -eps)))
        feats.append(1.0 / (x_vec**2 + eps))
        
        # Mixed
        feats.append(x_vec * np.sin(x_vec))
        feats.append(x_vec * np.cos(x_vec))
        feats.append(x_vec * np.tanh(x_vec))
        feats.append(x_vec * np.exp(np.clip(x_vec, -10, 10)))
        feats.append(x_vec * np.log(np.abs(x_vec) + eps))
        
        # Safe sinc-like functions
        sinc_x = np.sin(x_vec) / (x_vec + np.where(x_vec >= 0, eps, -eps))
        cosc_x = np.cos(x_vec) / (x_vec + np.where(x_vec >= 0, eps, -eps))
        feats.append(sinc_x)
        feats.append(cosc_x)
        
        # Absolute value
        feats.append(np.abs(x_vec))
        
        # Square root
        feats.append(np.sqrt(np.abs(x_vec)))
        
        return np.column_stack(feats)

    # Generate training features
    F_train = generate_features(x)
    # Handle any NaNs or Infs
    F_train = np.nan_to_num(F_train, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Normalize features for better conditioning
    # We store mean and std for use in predict
    F_mean = np.mean(F_train, axis=0)
    F_std = np.std(F_train, axis=0)
    F_std[F_std == 0] = 1.0 # Avoid division by zero
    
    F_train_norm = (F_train - F_mean) / F_std
    
    n_features = F_train_norm.shape[1]
    
    # Start with intercept
    selected_indices = []
    
    # Function to fit and return SSE
    def fit_and_sse(indices):
        if len(indices) == 0:
            # Just intercept
            mean_y = np.mean(y)
            pred = np.full_like(y, mean_y)
        else:
            # Design matrix: Intercept + selected normalized features
            X_design = np.column_stack([np.ones(n_samples)] + [F_train_norm[:, i] for i in indices])
            try:
                # Use lstsq for OLS
                w, _, _, _ = np.linalg.lstsq(X_design, y, rcond=None)
                pred = X_design @ w
            except np.linalg.LinAlgError:
                return float('inf'), None
        sse = np.sum((y - pred) ** 2)
        return sse, pred

    # Initial fit with just intercept
    current_sse, _ = fit_and_sse([])
    
    max_features = 6 # Allow slightly more features
    min_improvement = 1e-5 * current_sse
    
    for _ in range(max_features):
        best_idx = -1
        best_sse = current_sse
        
        for i in range(n_features):
            if i in selected_indices:
                continue
            new_indices = selected_indices + [i]
            sse, _ = fit_and_sse(new_indices)
            if sse < best_sse:
                best_sse = sse
                best_idx = i
        
        # If no feature improves SSE significantly, stop
        if best_idx == -1 or (current_sse - best_sse) < min_improvement:
            break
            
        # Add the best feature
        selected_indices.append(best_idx)
        current_sse = best_sse
        # Update min improvement threshold relative to new SSE
        min_improvement = 1e-5 * current_sse

    # Final fit with the selected features to get the weights
    final_indices = selected_indices
    
    if len(final_indices) == 0:
        w_final = np.array([np.mean(y)])
    else:
        X_design = np.column_stack([np.ones(n_samples)] + [F_train_norm[:, i] for i in final_indices])
        w_final, _, _, _ = np.linalg.lstsq(X_design, y, rcond=None)
        
    def predict(X_test):
        x_t = X_test[:, 0] if X_test.shape[1] > 0 else np.zeros(X_test.shape[0])
        # Generate features
        F_test = generate_features(x_t)
        F_test = np.nan_to_num(F_test, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Normalize using training statistics
        F_test_norm = (F_test - F_mean) / F_std
        
        # Select the ones we trained on
        if len(final_indices) > 0:
            F_sel = F_test_norm[:, final_indices]
        else:
            F_sel = np.zeros((X_test.shape[0], 0))
        
        # Construct design matrix with intercept
        if F_sel.size > 0:
            X_design = np.column_stack([np.ones(X_test.shape[0])] + [F_sel])
        else:
            X_design = np.ones((X_test.shape[0], 1))
        
        # Predict
        pred = X_design @ w_final
        return pred

    return predict

# EVOLVE-BLOCK-END
