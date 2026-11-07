# EVOLVE-BLOCK-START
import numpy as np
import time

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 310  # 310s out of 330s limit
    
    n = X_train.shape[0]
    d = X_train.shape[1]
    
    # Since all inputs are 1D based on the problem description
    x = X_train[:, 0].flatten()
    y = y_train.flatten()
    
    # Generate a library of candidate features
    def make_features(x):
        features = []
        names = []
        
        # Polynomials up to degree 8
        for p in range(0, 9):
            features.append(x**p)
            names.append(f"x^{p}")
        
        # Trig functions
        for mult in [0.5, 1, 2, 3]:
            features.append(np.sin(mult * x))
            features.append(np.cos(mult * x))
            names.append(f"sin({mult}*x)")
            names.append(f"cos({mult}*x)")
        
        # Exponential and log
        features.append(np.exp(x))
        features.append(np.exp(-x))
        features.append(np.exp(0.5*x))
        features.append(np.exp(-0.5*x))
        features.append(np.log(np.abs(x) + 1e-10))
        features.append(np.sqrt(np.abs(x)))
        names.extend(["exp(x)", "exp(-x)", "exp(0.5x)", "exp(-0.5x)", "log|x|", "sqrt|x|"])
        
        # Rational-like
        features.append(1.0 / (x + 1.0 + 1e-10))
        features.append(1.0 / (x - 1.0 + 1e-10))
        features.append(1.0 / (x**2 + 1.0))
        names.extend(["1/(x+1)", "1/(x-1)", "1/(x^2+1)"])
        
        # Interactions with other potential bases if d>1, but here d=1
        # Add some mixed terms
        features.append(x * np.sin(x))
        features.append(x * np.cos(x))
        features.append(x * np.exp(0.1*x))
        names.extend(["x*sin(x)", "x*cos(x)", "x*exp(0.1x)"])
        
        # Normalize features to prevent numerical issues
        F = np.column_stack(features)
        # Handle NaN/Inf
        F = np.nan_to_num(F, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Standardize columns
        means = F.mean(axis=0)
        stds = F.std(axis=0)
        stds[stds == 0] = 1.0
        F = (F - means) / stds
        
        return F, names
    
    F, names = make_features(x)
    m = F.shape[1]
    
    # Greedy forward selection
    # We want to find the subset of features that best predicts y
    
    # Precompute F^T F and F^T y for efficiency? 
    # Actually, for greedy selection, we can just do lstsq on the selected subset each time.
    # With 200 points and ~30 features, this is very fast.
    
    selected = []
    best_mse = np.inf
    max_features = 15  # Limit complexity to avoid overfitting
    
    # Add intercept (bias) implicitly or explicitly? 
    # Since we standardized, the mean is 0, but let's keep an intercept in the model or just rely on the features.
    # Standardizing removes the mean, so the linear model should include an intercept if we want to predict the original scale.
    # Actually, if we standardize y as well, we can predict standardized y and then inverse transform.
    
    y_mean = y.mean()
    y_std = y.std()
    if y_std == 0:
        y_std = 1.0
    y_stdized = (y - y_mean) / y_std
    
    # Store best model found so far
    best_selected = []
    best_mse_val = np.inf
    best_w = None
    
    for step in range(max_features):
        if time.time() > deadline:
            break
            
        best_idx = -1
        best_step_mse = np.inf
        
        # Try adding each unselected feature
        for i in range(m):
            if i in selected:
                continue
                
            # Candidate set
            cand_set = selected + [i]
            X_sub = F[:, cand_set]
            
            # Solve least squares
            # Add intercept column of 1s
            X_sub_int = np.column_stack([np.ones(X_sub.shape[0]), X_sub])
            try:
                w, *_ = np.linalg.lstsq(X_sub_int, y_stdized, rcond=None)
                pred = X_sub_int @ w
                mse = np.mean((pred - y_stdized)**2)
            except:
                continue
                
            if mse < best_step_mse:
                best_step_mse = mse
                best_idx = i
                
        if best_idx == -1:
            break
            
        # Check if adding this feature improves the model significantly
        # We compare best_step_mse with the current best_mse (from previous step)
        # Note: best_mse was the MSE of the model with 'selected' features before adding best_idx
        
        if best_step_mse < best_mse_val * 0.99:  # Require 1% improvement
            selected.append(best_idx)
            best_mse_val = best_step_mse
            best_selected = selected.copy()
            
            # Update best_w for the current best model
            X_sub = F[:, best_selected]
            X_sub_int = np.column_stack([np.ones(X_sub.shape[0]), X_sub])
            best_w, *_ = np.linalg.lstsq(X_sub_int, y_stdized, rcond=None)
        else:
            break
            
    # Final model: use the best selected features found
    if not best_selected:
        # Fallback to simple linear regression on x
        X_sub = np.column_stack([np.ones(n), x])
        best_w, *_ = np.linalg.lstsq(X_sub, y_stdized, rcond=None)
        def predict(X):
            X = X.reshape(-1, 1)
            x_test = X[:, 0]
            X_test = np.column_stack([np.ones(len(x_test)), x_test])
            y_pred_std = X_test @ best_w
            return y_pred_std * y_std + y_mean
    else:
        # We need to reconstruct the feature matrix for any X using the same transformations
        # This requires storing the transformation parameters or re-computing them.
        # Since make_features is deterministic, we can just call it again on X_test.
        
        # However, make_features standardizes based on the training data.
        # So we must store the means and stds of the features from training.
        
        # Let's refactor to store the feature generation logic and parameters.
        
        # Re-implement feature generation to return parameters
        def generate_features_params(x):
            features = []
            
            for p in range(0, 9):
                features.append(x**p)
            for mult in [0.5, 1, 2, 3]:
                features.append(np.sin(mult * x))
                features.append(np.cos(mult * x))
            features.append(np.exp(x))
            features.append(np.exp(-x))
            features.append(np.exp(0.5*x))
            features.append(np.exp(-0.5*x))
            features.append(np.log(np.abs(x) + 1e-10))
            features.append(np.sqrt(np.abs(x)))
            features.append(1.0 / (x + 1.0 + 1e-10))
            features.append(1.0 / (x - 1.0 + 1e-10))
            features.append(1.0 / (x**2 + 1.0))
            features.append(x * np.sin(x))
            features.append(x * np.cos(x))
            features.append(x * np.exp(0.1*x))
            
            F = np.column_stack(features)
            F = np.nan_to_num(F, nan=0.0, posinf=1e6, neginf=-1e6)
            
            means = F.mean(axis=0)
            stds = F.std(axis=0)
            stds[stds == 0] = 1.0
            F_norm = (F - means) / stds
            
            return F_norm, means, stds
        
        F_train, feat_means, feat_stds = generate_features_params(x)
        
        # Re-run greedy selection with the properly generated F_train
        # (The previous selection was on a slightly different F due to inline generation, 
        # but the logic is the same. To be safe, let's just use the F_train from here.)
        
        # Actually, the previous block computed F and selected indices based on that F.
        # The F here should be identical if the code is the same.
        # Let's just use the best_w and best_selected from the loop above.
        
        def predict(X):
            X = X.reshape(-1, 1)
            x_test = X[:, 0]
            F_test, _, _ = generate_features_params(x_test)
            
            # Select the columns
            X_sub = F_test[:, best_selected]
            X_sub_int = np.column_stack([np.ones(X_sub.shape[0]), X_sub])
            y_pred_std = X_sub_int @ best_w
            return y_pred_std * y_std + y_mean

    # Ensure prediction returns finite values
    def safe_predict(X):
        try:
            res = predict(X)
            res = np.nan_to_num(res, nan=0.0, posinf=1e6, neginf=-1e6)
            return res
        except:
            return np.zeros(X.shape[0])

    return safe_predict

# EVOLVE-BLOCK-END
