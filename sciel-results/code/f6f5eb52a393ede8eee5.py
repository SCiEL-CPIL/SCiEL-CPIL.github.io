import numpy as np
import time

def fit(X_train, y_train):
    start_time = time.time()
    DEADLINE = start_time + 280.0  # Leave 50s margin for 330s limit

    n, d = X_train.shape
    y_mean = np.mean(y_train)
    
    # If y is constant, return constant
    if np.std(y_train) < 1e-10:
        return lambda X: np.full(len(X), y_mean)

    # Feature generation function
    def generate_features(X):
        # X is (n, d)
        n_samples = X.shape[0]
        d_features = X.shape[1]
        
        # Base features for each dimension
        # 1. Identity and Polynomials
        # 2. Trig
        # 3. Exp
        # 4. Log
        
        base_feats = []
        
        for i in range(d_features):
            x = X[:, i]
            
            # Polynomials up to degree 3
            base_feats.append(np.ones(n_samples)) # Constant is added once globally, but here we add per-dim? No, let's add global constant.
            base_feats.append(x)
            base_feats.append(x**2)
            base_feats.append(x**3)
            
            # Trigonometric
            base_feats.append(np.sin(x))
            base_feats.append(np.cos(x))
            
            # Exponential (scaled to avoid overflow issues in linear algebra if values are huge, but max is 10)
            # exp(10) is ~22000, which is fine for float64
            base_feats.append(np.exp(x))
            
            # Logarithmic
            # x is in [1, ...] so log is safe
            base_feats.append(np.log(x))
            
            # Inverse
            base_feats.append(1.0 / x)

        # Add global constant (avoid duplicates if we added per dim, but let's just ensure one)
        # The loop above adds np.ones for each dim. We should only add one constant.
        # Let's reconstruct:
        
        cols = [np.ones(n_samples)]
        
        for i in range(d_features):
            x = X[:, i]
            # Poly
            cols.append(x)
            cols.append(x**2)
            cols.append(x**3)
            
            # Trig
            cols.append(np.sin(x))
            cols.append(np.cos(x))
            
            # Exp
            cols.append(np.exp(x))
            
            # Log
            cols.append(np.log(x))
            
            # Inverse
            cols.append(1.0 / x)
            
        # Interactions: Products of linear terms (x_i * x_j) and mixed (x_i * sin(x_j), etc.)
        # To keep size down, we only do products of the "main" linear-ish terms: x_i, sin(x_i), cos(x_i), exp(x_i)
        # Actually, let's just do products of x_i and x_j, and x_i * sin(x_j)
        
        # Product of raw inputs
        for i in range(d_features):
            for j in range(i + 1, d_features):
                cols.append(X[:, i] * X[:, j])
        
        # Mixed interactions (limited)
        for i in range(d_features):
            for j in range(d_features):
                if i != j:
                    cols.append(X[:, i] * np.sin(X[:, j]))
                    cols.append(X[:, i] * np.cos(X[:, j]))
                    
        # Stack
        return np.column_stack(cols)

    # Generate features for training
    X_train_feats = generate_features(X_train)
    
    # Standardize features for better conditioning of lstsq/ridge
    # However, standardizing changes the coefficients. We can just use lstsq directly on the raw features if they are well-scaled.
    # Exp(10) is 22000, Log(1) is 0. The scales vary.
    # Let's use Ridge Regression with a small lambda.
    
    # Try a few different lambdas or just use lstsq.
    # Since data is noiseless, if the true function is in the span, lstsq gives exact fit.
    # If not, lstsq gives the projection.
    
    # To handle potential ill-conditioning from exp/log, let's scale the features.
    # But scaling features changes the interpretation. For prediction, it doesn't matter as long as we use the same scaling for X_test.
    
    feat_mean = np.mean(X_train_feats, axis=0)
    feat_std = np.std(X_train_feats, axis=0)
    # Avoid division by zero
    feat_std[feat_std < 1e-10] = 1.0
    
    X_train_scaled = (X_train_feats - feat_mean) / feat_std
    
    # Fit using Ridge (small alpha for stability)
    alpha = 1e-6
    # Solve (X^T X + alpha I) w = X^T y
    # Or use numpy lstsq on augmented system? 
    # scipy.linalg.lstsq is good.
    
    try:
        # Add regularization term to system:
        # [X_scaled; sqrt(alpha)*I] w = [y; 0]
        # This is equivalent to Ridge.
        
        # Simpler: use np.linalg.lstsq on the original features might be unstable if collinear.
        # Let's use the augmented matrix approach for Ridge.
        m = X_train_scaled.shape[1]
        I = np.eye(m)
        X_aug = np.vstack([X_train_scaled, np.sqrt(alpha) * I])
        y_aug = np.concatenate([y_train, np.zeros(m)])
        
        w, residuals, rank, s = np.linalg.lstsq(X_aug, y_aug, rcond=None)
        
    except Exception:
        # Fallback to simple poly2 if something goes wrong
        def features_fallback(X):
            cols = [np.ones(len(X))] + [X[:, j] for j in range(X.shape[1])] + [X[:, i] * X[:, j] for i in range(X.shape[1]) for j in range(i, X.shape[1])]
            return np.column_stack(cols)
        w_fb, *_ = np.linalg.lstsq(features_fallback(X_train), y_train, rcond=None)
        return lambda X: features_fallback(X) @ w_fb

    def predict(X):
        # Check time
        if time.time() > DEADLINE:
             # Return mean as a safe fallback if we are running out of time (shouldn't happen for 200 points)
             return np.full(len(X), y_mean)
        
        X_feats = generate_features(X)
        X_scaled = (X_feats - feat_mean) / feat_std
        preds = X_scaled @ w
        
        # Ensure finite
        preds = np.nan_to_num(preds, nan=y_mean, posinf=y_mean, neginf=y_mean)
        
        return preds

    return predict
