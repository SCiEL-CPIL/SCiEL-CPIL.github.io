"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def construct(training_x, training_subgrid_tendency):
    """
    Fit a shared degree-five polynomial to the unresolved fast-scale forcing
    using Ridge Regression in a scaled domain for stability and generalization.
    
    Args:
        training_x: Flattened array of slow-state values
        training_subgrid_tendency: Flattened array of unresolved tendencies
        
    Returns:
        Array of 6 finite real polynomial coefficients in ascending powers of x
    """
    start_time = time.time()
    DEADLINE = 110.0  # Safety margin before 120s limit

    x = np.asarray(training_x, dtype=np.float64)
    y = np.asarray(training_subgrid_tendency, dtype=np.float64)
    
    # Remove any potential NaNs/Infs just in case, though inputs should be clean
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    
    if len(x) == 0:
        return np.zeros(6)

    # 1. Scale and shift x to [-1, 1] to improve conditioning of the Vandermonde matrix
    # We use the min/max of the training data for the scaling.
    # If range is zero, set to 1 to avoid division by zero.
    x_min = np.min(x)
    x_max = np.max(x)
    x_range = x_max - x_min
    if x_range < 1e-12:
        x_range = 1.0
        x_center = x_min
    else:
        x_center = (x_max + x_min) / 2.0
        
    x_scaled = (x - x_center) / (x_range / 2.0)
    
    # Ensure x_scaled is strictly within [-1, 1] for numerical safety in powers
    # np.clip is safe here as we are just defining the basis for the fit
    x_scaled = np.clip(x_scaled, -1.0, 1.0)

    # 2. Construct the design matrix for the scaled domain: [1, xs, xs^2, ..., xs^5]
    # Degree 5 means 6 coefficients.
    V_scaled = np.vander(x_scaled, N=6, increasing=True)
    
    # 3. Ridge Regression (L2 Regularization)
    # We search for a lambda that minimizes validation-like error (using a holdout from training)
    # or just pick a sensible default. Given the time budget, we can do a quick grid search
    # on the training data split to find a robust lambda.
    
    # Split training data for internal validation to pick lambda
    # Since we don't have external validation, we use a cross-validation-like approach or simple split.
    # Given N is likely large (flattened trajectories), a random 80/20 split is fast.
    
    rng = np.random.default_rng(42)
    n_samples = len(x)
    indices = rng.permutation(n_samples)
    split_idx = int(0.8 * n_samples)
    train_idx = indices[:split_idx]
    val_idx = indices[split_idx:]
    
    V_train = V_scaled[train_idx]
    y_train = y[train_idx]
    V_val = V_scaled[val_idx]
    y_val = y[val_idx]
    
    # Grid of lambda values
    # Lambda is the regularization parameter. 
    # We want to balance fit and complexity.
    lambdas = [1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
    
    best_lambda = 1e-4
    best_mse = np.inf
    best_coeff_scaled = np.zeros(6)
    
    for lam in lambdas:
        # Solve (V^T V + lambda * I) c = V^T y
        # Note: We usually don't penalize the intercept (0th order) as heavily, 
        # but for simplicity and given the polynomial nature, penalizing all is often robust enough.
        # A more sophisticated approach penalizes 1..5. Let's do that.
        reg_matrix = np.eye(6) * lam
        reg_matrix[0, 0] = 0.0  # Don't penalize the mean/intercept
        
        A = V_train.T @ V_train + reg_matrix
        b = V_train.T @ y_train
        
        try:
            c_scaled = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            continue
            
        # Evaluate on validation split
        y_pred_val = V_val @ c_scaled
        mse_val = np.mean((y_val - y_pred_val) ** 2)
        
        if mse_val < best_mse:
            best_mse = mse_val
            best_lambda = lam
            best_coeff_scaled = c_scaled
            
    # 4. Refine using the best lambda on ALL training data
    # This ensures we use all information for the final coefficients.
    A_full = V_scaled.T @ V_scaled + (np.eye(6) * best_lambda)
    A_full[0, 0] = 0.0  # Don't penalize intercept
    b_full = V_scaled.T @ y
    
    try:
        c_scaled_full = np.linalg.solve(A_full, b_full)
    except np.linalg.LinAlgError:
        c_scaled_full = best_coeff_scaled

    # 5. Transform coefficients from scaled domain to original domain
    # We have P(x) = sum_{i=0}^5 c_i_scaled * ((x - center) / half_range)^i
    # Let u = (x - center) / half_range
    # P(x) = sum_{i=0}^5 c_i_scaled * u^i
    # We need coefficients a_k such that P(x) = sum_{k=0}^5 a_k * x^k
    
    # This is a polynomial transformation.
    # u = alpha * x + beta
    # where alpha = 1 / half_range, beta = -center / half_range
    alpha = 1.0 / (x_range / 2.0)
    beta = -x_center / (x_range / 2.0)
    
    # We can compute the transformation matrix T such that c_orig = T @ c_scaled
    # Or simply evaluate the polynomial on a set of points and fit? No, that's unstable.
    # Use the binomial expansion or iterative substitution.
    
    # Let's use numpy's polynomial tools if possible, or manual expansion.
    # np.polynomial.polynomial.polymul and polyadd can help, but let's do it manually for speed/clarity.
    
    # c_scaled[i] is coeff for u^i.
    # u^i = (alpha*x + beta)^i = sum_{j=0}^i binom(i, j) alpha^j beta^(i-j) x^j
    # So a_j = sum_{i=j}^5 c_scaled[i] * binom(i, j) * alpha^j * beta^(i-j)
    
    a_orig = np.zeros(6)
    for i in range(6):
        ci = c_scaled_full[i]
        if ci == 0: continue
        for j in range(i + 1):
            # Binomial coefficient C(i, j)
            # Small numbers, can compute directly
            binom_ij = 1
            for k in range(j):
                binom_ij = binom_ij * (i - k) / (k + 1)
                
            term = ci * binom_ij * (alpha ** j) * (beta ** (i - j))
            a_orig[j] += term
            
    # Check time
    if time.time() - start_time > DEADLINE:
        # Return best found so far (which is a_orig calculated above)
        pass

    # Final sanity check: ensure coefficients are finite
    if not np.all(np.isfinite(a_orig)):
        # Fallback to simple linear fit in original domain if transformation failed
        try:
            c_lin = np.polynomial.polynomial.polyfit(x, y, 1)
            a_orig = np.zeros(6)
            a_orig[:2] = c_lin
        except:
            a_orig = np.zeros(6)

    return a_orig
# EVOLVE-BLOCK-END
