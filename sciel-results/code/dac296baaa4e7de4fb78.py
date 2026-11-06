"""Strategy: Ridge Regression with internal validation-based lambda selection."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def construct(training_x, training_subgrid_tendency):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit (allowing for overhead)
    
    x = np.asarray(training_x, dtype=np.float64).ravel()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()
    
    if len(x) < 6:
        return np.zeros(6)
    
    # 1. Data Preprocessing: Standardize x for numerical stability in polynomial basis
    # This helps Ridge Regression handle the ill-conditioned Vandermonde matrix better
    x_mean = np.mean(x)
    x_std = np.std(x)
    if x_std < 1e-12:
        x_std = 1.0
    x_norm = (x - x_mean) / x_std
    
    # 2. Construct Polynomial Features (Ascending powers 0..5)
    # We use the normalized x for the fit, but we must transform back to original x scale?
    # No, the task asks for coefficients c_i such that sum(c_i * x**i).
    # If we fit on x_norm, the coefficients are for x_norm.
    # We need coefficients for x.
    # Let's stick to fitting directly on x but regularize.
    # Actually, fitting high-degree polynomials on raw x can be numerically unstable.
    # However, the output must be in terms of original x.
    # Strategy: Fit on x_norm, then expand the polynomial back to x basis.
    # p(x_norm) = sum(a_i * x_norm^i)
    # x_norm = (x - mu) / sigma
    # This transformation is complex to invert for coefficients.
    # Alternative: Use np.polynomial.polynomial.polyfit with rcond? No, that's least squares.
    # Let's use the standard Vandermonde on x, but regularize.
    # To avoid ill-conditioning, we can scale x during the solve and rescale coefficients.
    
    # Let's try a simpler approach: Fit on x, but use a robust solver.
    # If x range is large, high powers are huge.
    # Let's check the range of x.
    if np.max(np.abs(x)) > 100:
         # Scale x for computation, then unscale coefficients
         scale = np.max(np.abs(x))
         x_scaled = x / scale
    else:
        scale = 1.0
        x_scaled = x

    # Degrees 0 to 5
    deg = 5
    V = np.vander(x_scaled, N=deg+1, increasing=True)
    
    # 3. Ridge Regression with Lambda Search
    # We want to find lambda that minimizes error on a held-out set?
    # The task says "Train on supplied... minimize error on independent validation".
    # We don't have validation data inside `construct`.
    # So we must use cross-validation or a heuristic.
    # Given the small data size likely (two trajectories), K-Fold CV is good.
    
    n_samples = len(x)
    
    # If data is very small, CV might be unstable. 
    # Let's use Leave-One-Out or K-Fold.
    # K=5 is standard.
    
    best_lambda = 0.0
    best_mse = np.inf
    best_coeffs = np.zeros(deg+1)
    
    # Logarithmic search for lambda
    # Lambda range: 1e-8 to 1e8
    lambdas = np.logspace(-8, 8, 17)
    
    # Simple K-Fold (K=5)
    k = 5
    indices = np.arange(n_samples)
    # Shuffle indices for CV (fixed seed for determinism? No, random is fine, but let's use a fixed seed for reproducibility in scoring if possible, or just np.random default)
    np.random.seed(42)
    np.random.shuffle(indices)
    folds = np.array_split(indices, k)
    
    for lam in lambdas:
        if time.time() > deadline - 10:
            break
            
        mse_sum = 0.0
        valid_folds = 0
        
        for i in range(k):
            if time.time() > deadline - 5:
                break
                
            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
            
            if len(train_idx) < deg + 1:
                continue
                
            X_train = V[train_idx]
            y_train = y[train_idx]
            X_val = V[val_idx]
            y_val = y[val_idx]
            
            # Ridge Solve: (X^T X + lam * I) c = X^T y
            # We only regularize the non-constant term? Usually all, or constant term not regularized.
            # Standard Ridge regularizes all. Let's regularize all for simplicity, 
            # or set lam[0]=0 to allow mean shift.
            # Let's regularize all coefficients for stability.
            
            XtX = X_train.T @ X_train
            Xty = X_train.T @ y_train
            
            # Add regularization
            # If we regularize the intercept, it shifts the mean. 
            # Usually better not to regularize the intercept.
            reg_matrix = np.eye(deg + 1) * lam
            reg_matrix[0, 0] = 0.0  # No regularization on intercept
            
            A = XtX + reg_matrix
            try:
                c_scaled = np.linalg.solve(A, Xty)
            except np.linalg.LinAlgError:
                c_scaled = np.linalg.lstsq(A, Xty, rcond=None)[0]
            
            # Predict on val
            y_pred_val = X_val @ c_scaled
            mse_val = np.mean((y_val - y_pred_val) ** 2)
            mse_sum += mse_val
            valid_folds += 1
            
        if valid_folds > 0:
            avg_mse = mse_sum / valid_folds
            if avg_mse < best_mse:
                best_mse = avg_mse
                best_lambda = lam
                # Re-solve on full training set with best lambda to get final coeffs
                XtX_full = V.T @ V
                Xty_full = V.T @ y
                reg_full = np.eye(deg + 1) * best_lambda
                reg_full[0, 0] = 0.0
                A_full = XtX_full + reg_full
                try:
                    c_scaled_full = np.linalg.solve(A_full, Xty_full)
                except np.linalg.LinAlgError:
                    c_scaled_full = np.linalg.lstsq(A_full, Xty_full, rcond=None)[0]
                
                # Transform coefficients back to original x scale
                # We solved for x_scaled = x / scale
                # p(x_scaled) = sum(c_i_scaled * (x/scale)^i)
                # p(x) = sum(c_i_scaled * scale^{-i} * x^i)
                # So c_i = c_i_scaled * scale^{-i}
                
                powers = np.arange(deg + 1)
                c_original = c_scaled_full * (scale ** (-powers))
                best_coeffs = c_original
    
    # If no valid fold or search failed, fallback to least squares (lambda=0)
    if best_mse == np.inf:
         # Fallback: Unregularized least squares on scaled x
         try:
             c_scaled_full = np.linalg.lstsq(V, y, rcond=None)[0]
             powers = np.arange(deg + 1)
             best_coeffs = c_scaled_full * (scale ** (-powers))
         except:
             best_coeffs = np.zeros(6)

    return best_coeffs
# EVOLVE-BLOCK-END
