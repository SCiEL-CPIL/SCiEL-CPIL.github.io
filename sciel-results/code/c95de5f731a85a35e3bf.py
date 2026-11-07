import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.linear_model import Ridge
import time

def fit(X_train, y_train):
    start_time = time.time()
    # 300s limit, keep 10s buffer
    deadline = start_time + 290
    
    # Standardize inputs
    mean_x = np.mean(X_train, axis=0)
    std_x = np.std(X_train, axis=0)
    std_x[std_x < 1e-8] = 1.0
    X_norm = (X_train - mean_x) / std_x
    
    # Simple train/val split for model selection
    n = len(X_train)
    indices = np.arange(n)
    # Use a fixed seed for reproducibility in selection
    np.random.seed(42)
    np.random.shuffle(indices)
    split = int(0.8 * n)
    idx_train, idx_val = indices[:split], indices[split:]
    
    X_tr, y_tr = X_norm[idx_train], y_train[idx_train]
    X_va, y_va = X_norm[idx_val], y_train[idx_val]
    
    def make_features(X):
        """
        Generate rich symbolic-like features.
        X is normalized.
        """
        d = X.shape[1]
        feats = [np.ones(len(X))]
        
        # Linear
        for i in range(d):
            feats.append(X[:, i])
            
        # Quadratic
        for i in range(d):
            for j in range(i, d):
                feats.append(X[:, i] * X[:, j])
                
        # Cubic
        for i in range(d):
            feats.append(X[:, i]**3)
            
        # Trig
        for i in range(d):
            feats.append(np.sin(X[:, i]))
            feats.append(np.cos(X[:, i]))
            
        # Sqrt and Inverse (Safe because original inputs >= 1, 
        # but X_norm can be negative. 
        # Wait! X_norm can be negative. 
        # sqrt of negative is nan. 
        # We must apply these to ORIGINAL X or handle carefully.
        # Let's apply Sqrt/Inv/Exp to the original X values, not normalized.
        # But we don't have original X in this function scope easily for prediction?
        # We can pass original X to make_features?
        # No, make_features is called on X_norm for training.
        # We need original X for sqrt/inv/exp.
        # So we should generate features from ORIGINAL X, not normalized X.
        # But Ridge/GP prefer normalized inputs for linear/poly parts?
        # Actually, if we use original X, the scales are [1, 10].
        # Poly features will be large. Ridge handles this with regularization.
        # GP prefers normalized inputs for kernel.
        
        # Strategy:
        # 1. Use Original X for "symbolic" features (sqrt, inv, exp).
        # 2. Use Normalized X for "linear/poly/trig" features?
        # Mixing normalized and original in one feature matrix is messy.
        
        # Alternative: 
        # Apply symbolic features to Original X.
        # Apply poly/trig to Normalized X.
        # This is inconsistent.
        
        # Better: 
        # Since inputs are bounded [1, 10], they are well-behaved.
        # We can just use Original X for ALL features.
        # GP will normalize internally or we can normalize X for GP separately.
        # Ridge can handle the scale with alpha.
        
        # Let's modify: 
        # make_features takes ORIGINAL X.
        # For GP, we pass NORMALIZED X.
        # For Ridge, we pass FEATURES from ORIGINAL X.
        
        pass 

    # Redefine make_features to take original X
    def make_features_orig(X_orig):
        d = X_orig.shape[1]
        feats = [np.ones(len(X_orig))]
        
        # Linear
        for i in range(d):
            feats.append(X_orig[:, i])
            
        # Quadratic
        for i in range(d):
            for j in range(i, d):
                feats.append(X_orig[:, i] * X_orig[:, j])
                
        # Cubic
        for i in range(d):
            feats.append(X_orig[:, i]**3)
            
        # Trig
        for i in range(d):
            feats.append(np.sin(X_orig[:, i]))
            feats.append(np.cos(X_orig[:, i]))
            
        # Sqrt
        for i in range(d):
            feats.append(np.sqrt(X_orig[:, i]))
            
        # Inverse
        for i in range(d):
            feats.append(1.0 / X_orig[:, i])
            
        # Exp(-x)
        for i in range(d):
            feats.append(np.exp(-X_orig[:, i]))
            
        return np.column_stack(feats)

    # Get original X for features
    X_tr_orig = X_train[idx_train]
    X_va_orig = X_train[idx_val]
    
    F_tr = make_features_orig(X_tr_orig)
    F_va = make_features_orig(X_va_orig)
    
    # Model 1: Ridge
    ridge = Ridge(alpha=1.0)
    ridge.fit(F_tr, y_tr)
    pred_ridge_val = ridge.predict(F_va)
    mse_ridge = np.mean((pred_ridge_val - y_va)**2)
    
    # Model 2: GP
    mse_gp = np.inf
    gp = None
    if time.time() < deadline - 10:
        # GP on normalized X
        kernel = 1.0 * Matern(length_scale=1.0, nu=2.5) + 0.1 * np.ones(X_tr.shape[1])
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, normalize_y=True)
        gp.fit(X_tr, y_tr)
        pred_gp_val = gp.predict(X_va)
        mse_gp = np.mean((pred_gp_val - y_va)**2)
        
    # Select best
    use_gp = mse_gp < mse_ridge
    
    def predict(X):
        if use_gp:
            X_norm_pred = (X - mean_x) / std_x
            return gp.predict(X_norm_pred)
        else:
            F_pred = make_features_orig(X)
            return ridge.predict(F_pred)
            
    return predict
