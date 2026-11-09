import time
import numpy as np
import sys

def fit(X_train, y_train):
    """
    Fit a regression model and return a predict callable.
    Uses an adaptive linear basis expansion with Ridge Regression and 
    Cross-Validation to handle diverse functional forms robustly.
    """
    start_time = time.time()
    deadline = start_time + 280  # 20s safety margin before 300s limit

    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    n, d = X.shape
    x = X[:, 0]

    def elapsed():
        return time.time() - start_time

    def remaining():
        return deadline - time.time()

    # 1. Generate a rich set of basis functions
    # We need to be careful with domains. 
    # Ranges provided: [-3,3], [-2,2], [-2,2], [0.1,6]
    
    def generate_features(x_vec):
        """
        Generate a matrix of basis function values for input vector x_vec.
        """
        x_vec = np.asarray(x_vec, dtype=np.float64)
        n_pts = len(x_vec)
        
        # Handle potential division by zero or log of negative
        # For x in [-3,3] or [-2,2], 1/x is problematic at 0.
        # We use a safe version: 1 / (x + epsilon) or just skip if x=0.
        # Better: Use a set of features that are safe across all domains.
        
        cols = []
        
        # Constant
        cols.append(np.ones(n_pts))
        
        # Polynomials (degree 1 to 8)
        for p in range(1, 9):
            cols.append(x_vec ** p)
            
        # Fourier series (frequencies 1 to 15)
        # This helps if the function is oscillatory
        for k in range(1, 16):
            cols.append(np.sin(k * x_vec))
            cols.append(np.cos(k * x_vec))
            
        # Exponentials
        # x is in [-3, 3] max mostly. exp(3) ~ 20, exp(-3) ~ 0.05. Safe.
        # For x in [0.1, 6], exp(6) ~ 403. Safe.
        for b in [0.5, 1.0, 1.5, 2.0]:
            cols.append(np.exp(b * x_vec))
            cols.append(np.exp(-b * x_vec))
            
        # Reciprocals (safe version)
        # If x can be 0, 1/x is inf. We use 1/(x^2+1) which is smooth.
        cols.append(1.0 / (x_vec**2 + 1.0))
        cols.append(x_vec / (x_vec**2 + 1.0))
        
        # If the domain is strictly positive (like [0.1, 6]), log is useful.
        # But for [-3,3], log(x) is invalid.
        # We can detect if min(x) > 0.
        if np.min(x_vec) > 0:
            cols.append(np.log(x_vec))
            cols.append(np.log(x_vec + 1.0)) # log(1.1) to log(7)
            cols.append(1.0 / x_vec)
            cols.append(1.0 / x_vec**2)
            
        # Tanh and Sigmoid-like
        for s in [0.5, 1.0, 2.0]:
            cols.append(np.tanh(s * x_vec))
            cols.append(1.0 / (1.0 + np.exp(-s * x_vec))) # Sigmoid
            
        # Square root of absolute value (helps for y ~ sqrt(x) or |x|)
        cols.append(np.sqrt(np.abs(x_vec)))
        cols.append(np.abs(x_vec))
        
        # Construct matrix
        Phi = np.column_stack(cols)
        
        # Ensure finite values
        Phi = np.nan_to_num(Phi, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Standardize columns to improve conditioning for Ridge
        # This is crucial for numerical stability
        means = np.mean(Phi, axis=0)
        stds = np.std(Phi, axis=0)
        stds[stds < 1e-10] = 1.0
        Phi_std = (Phi - means) / stds
        
        return Phi_std, means, stds, Phi

    # Precompute features for training data
    Phi_train, means, stds, Phi_train_raw = generate_features(x)
    
    # Remove columns that are nearly constant or have very high correlation?
    # Ridge handles collinearity well, so we keep them but standardize.
    
    # 2. Cross-Validation for Ridge Regularization
    def cv_ridge(X_feat, y_vec, alphas, k=5):
        """
        Perform k-fold CV to find best alpha.
        """
        n_samples = len(y_vec)
        indices = np.arange(n_samples)
        np.random.seed(42)
        np.random.shuffle(indices)
        folds = np.array_split(indices, k)
        
        best_alpha = alphas[0]
        best_error = np.inf
        
        # Precompute X^T X + lambda I for each lambda? 
        # It's faster to just solve for each fold and each lambda if not too many.
        # n=200, k=5, len(alphas)=~20. 100 solves of 200xN systems. Very fast.
        
        for alpha in alphas:
            mse_sum = 0.0
            valid = True
            for i in range(k):
                val_idx = folds[i]
                tr_idx = np.concatenate([folds[j] for j in range(k) if j != i])
                
                X_tr = X_feat[tr_idx]
                y_tr = y_vec[tr_idx]
                X_val = X_feat[val_idx]
                y_val = y_vec[val_idx]
                
                try:
                    # Ridge solution: w = (X^T X + alpha I)^-1 X^T y
                    # Using solve is more stable
                    XtX = X_tr.T @ X_tr
                    Xty = X_tr.T @ y_tr
                    # Add alpha to diagonal
                    reg = alpha * np.eye(XtX.shape[0])
                    A = XtX + reg
                    w = np.linalg.solve(A, Xty)
                    
                    pred_val = X_val @ w
                    mse = np.mean((pred_val - y_val) ** 2)
                    if not np.isfinite(mse):
                        valid = False
                        break
                    mse_sum += mse
                except Exception:
                    valid = False
                    break
            
            if valid:
                mean_mse = mse_sum / k
                if mean_mse < best_error:
                    best_error = mean_mse
                    best_alpha = alpha
                    
        return best_alpha, best_error

    # Define alpha range
    # Since we standardized features, alpha range 1e-8 to 1e8 is reasonable
    alphas = np.logspace(-8, 8, 30)
    
    best_alpha = 1.0
    best_cv_error = np.inf
    
    if remaining() > 5:
        best_alpha, best_cv_error = cv_ridge(Phi_train, y, alphas)

    # 3. Fit final model with best alpha on all training data
    try:
        XtX = Phi_train.T @ Phi_train
        Xty = Phi_train.T @ y
        reg = best_alpha * np.eye(XtX.shape[0])
        A = XtX + reg
        w_final = np.linalg.solve(A, Xty)
    except Exception:
        # Fallback to simple least squares if ridge fails (unlikely)
        try:
            w_final, *_ = np.linalg.lstsq(Phi_train, y, rcond=None)
        except Exception:
            w_final = np.zeros(Phi_train.shape[1])
            # If everything fails, predict mean
            w_final[0] = np.mean(y) # Constant term is index 0

    # 4. Create predict function
    def predict(X_in):
        X_in = np.asarray(X_in, dtype=np.float64)
        try:
            x_in = X_in[:, 0]
            # Regenerate features for test points using the same logic
            # Note: We need to use the same means/stds? 
            # Wait, generate_features standardizes the NEW data independently.
            # This is WRONG for Ridge if the scaling is part of the model.
            # We MUST use the training means and stds to transform test data.
            
            # Let's modify generate_features to accept means/stds or return raw and we transform outside.
            # Actually, easier: generate raw features, then transform using train stats.
            
            Phi_in, _, _, _ = generate_features(x_in)
            
            # Wait, generate_features calculates its own means/stds.
            # We need to apply the TRAINING means and stds to the RAW features of X_in.
            # Let's refactor: create a function that returns RAW features, and a transformer.
            
            # Refactoring inside predict is risky/slow. 
            # Let's just re-calculate raw features and apply the stored means/stds.
            
            # We need a way to get raw features.
            # The current generate_features returns standardized.
            # Let's just recreate the raw matrix logic here.
            
            # To be safe and fast, I will re-implement the raw feature generation 
            # and apply the specific means/stds from training.
            
            def get_raw_features(x_vec):
                x_vec = np.asarray(x_vec, dtype=np.float64)
                n_pts = len(x_vec)
                cols = []
                cols.append(np.ones(n_pts))
                for p in range(1, 9):
                    cols.append(x_vec ** p)
                for k in range(1, 16):
                    cols.append(np.sin(k * x_vec))
                    cols.append(np.cos(k * x_vec))
                for b in [0.5, 1.0, 1.5, 2.0]:
                    cols.append(np.exp(b * x_vec))
                    cols.append(np.exp(-b * x_vec))
                cols.append(1.0 / (x_vec**2 + 1.0))
                cols.append(x_vec / (x_vec**2 + 1.0))
                if np.min(x_vec) > 0:
                    cols.append(np.log(x_vec))
                    cols.append(np.log(x_vec + 1.0))
                    cols.append(1.0 / x_vec)
                    cols.append(1.0 / x_vec**2)
                for s in [0.5, 1.0, 2.0]:
                    cols.append(np.tanh(s * x_vec))
                    cols.append(1.0 / (1.0 + np.exp(-s * x_vec)))
                cols.append(np.sqrt(np.abs(x_vec)))
                cols.append(np.abs(x_vec))
                
                Phi_raw = np.column_stack(cols)
                Phi_raw = np.nan_to_num(Phi_raw, nan=0.0, posinf=1e6, neginf=-1e6)
                return Phi_raw

            Phi_raw_in = get_raw_features(x_in)
            
            # Ensure dimensions match (in case x_in min > 0 was false in train but true in test? 
            # Unlikely given the problem statement ranges are fixed per equation. 
            # But to be safe, if dims don't match, we pad or slice. 
            # Given fixed ranges, dims should be identical.)
            if Phi_raw_in.shape[1] != len(means):
                 # Mismatch in feature count (e.g. log branch taken differently)
                 # This is a bug risk. 
                 # The ranges are fixed. 
                 # Eq 1: [-3,3] -> min < 0. No log.
                 # Eq 2: [-2,2] -> min < 0. No log.
                 # Eq 3: [-2,2] -> min < 0. No log.
                 # Eq 4: [0.1,6] -> min > 0. Log taken.
                 # So for Eq 4, both train and test will have log.
                 # For others, neither will.
                 # So shapes should match.
                 
                 # If not, fallback to constant
                 return np.full(len(X_in), np.mean(y))
            
            # Apply training scaling
            Phi_in_scaled = (Phi_raw_in - means) / stds
            
            pred = Phi_in_scaled @ w_final
            
            # Safety check
            if not np.all(np.isfinite(pred)):
                pred = np.where(np.isfinite(pred), pred, np.mean(y))
                
            return pred

        except Exception:
            return np.full(len(X_in), np.mean(y))

    return predict

# EVOLVE-BLOCK-END
