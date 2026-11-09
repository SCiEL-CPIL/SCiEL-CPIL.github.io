"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct(training_x, training_subgrid_tendency):
    t_start = time.time()
    deadline = 110.0  # Safety margin before 120s limit
    
    x = np.asarray(training_x, dtype=np.float64).ravel()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()
    
    n = min(x.size, y.size)
    if n == 0:
        return np.zeros(6, dtype=np.float64)
    
    x = x[:n]
    y = y[:n]
    
    # Remove non-finite values
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = x.size
    
    if n == 0:
        return np.zeros(6, dtype=np.float64)
    
    # Strategy:
    # 1. Normalize x to [-1, 1] for numerical stability.
    # 2. Use a robust weighted least squares approach where weights are determined by a 
    #    Gaussian Process-like kernel or simply a robust loss function (Huber) to reduce 
    #    the influence of outliers/noise.
    # 3. Alternatively, use a composite objective: fit the polynomial to minimize the 
    #    sum of squared residuals but with a regularization term that penalizes high-frequency 
    #    oscillations (roughness penalty). This is a structural change from simple LS or 
    #    kernel smoothing followed by LS. It's a Tikhonov regularized fit on the raw data 
    #    but with a specific roughness penalty on the polynomial coefficients or derivatives.
    #    
    #    Let's try a different structural change: **Bayesian Linear Regression** or 
    #    **Regularized Least Squares with a specific penalty structure**.
    #    
    #    Actually, the inspiration 3 mentioned Tikhonov regularization. The current program 
    #    does Kernel Regression -> Polyfit. 
    #    
    #    Let's try: **Direct Regularized Least Squares with a roughness penalty**.
    #    Minimize: ||y - Vx c||^2 + lambda * ||P c||^2
    #    where Vx is the Vandermonde matrix, c is the coefficients, and P is a penalty matrix 
    #    that penalizes the magnitude of higher-order coefficients or their second derivatives.
    #    This is structurally different from the two-step denoising + fit. It is a single-step 
    #    optimization that balances data fit and smoothness.
    
    # Normalize x to [-1, 1]
    x_min, x_max = np.min(x), np.max(x)
    if x_max > x_min:
        x_norm = (x - (x_max + x_min) / 2.0) / ((x_max - x_min) / 2.0)
    else:
        x_norm = np.zeros_like(x)
    
    # Create Vandermonde matrix for degree 5 polynomial in ascending powers
    # V[:, i] = x_norm ** i
    V = np.vander(x_norm, N=6, increasing=True)
    
    # Regularization: Penalize higher order coefficients
    # We want to keep the polynomial smooth. A simple penalty is on the coefficients themselves,
    # weighted by their order. Or a roughness penalty on the second derivative.
    # Let's use a diagonal penalty matrix W where W[i] = lambda * (i+1)^2 or similar.
    # This shrinks higher-order terms more aggressively.
    
    # We can also try to optimize lambda. But let's start with a fixed, reasonable lambda.
    # A common choice is to make the regularization term comparable to the data fit term.
    
    # Let's implement a simple regularized least squares:
    # min_c ||y - V c||^2 + lambda * c^T D c
    # where D is a diagonal matrix.
    
    # To choose lambda, we can use a simple heuristic or a small grid search if time permits.
    # For now, let's use a fixed lambda based on the scale of y.
    # lambda = alpha * (y^T y) / n / (trace(D))
    
    # Let's use a penalty that encourages lower-order terms.
    # D = diag([1, 1, 10, 100, 1000, 10000])
    # This is a strong prior that higher-order terms should be small.
    
    # A better approach: Use the residual variance to scale the penalty.
    # Let's try a few lambdas and pick the best one based on validation on a subset of training data.
    
    # Split training data into train and validation for hyperparameter selection
    # This is a structural change: using an internal validation set to tune the regularizer.
    
    # Shuffle data
    indices = np.arange(n)
    np.random.shuffle(indices)
    
    # Use 90% for tuning, 10% for validation
    val_size = max(1, int(0.1 * n))
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    
    x_val = x_norm[val_indices]
    y_val = y[val_indices]
    x_train = x_norm[train_indices]
    y_train = y[train_indices]
    
    V_val = np.vander(x_val, N=6, increasing=True)
    V_train = np.vander(x_train, N=6, increasing=True)
    
    # Define a range of lambda values to try
    # lambda controls the strength of regularization
    # We want to penalize the magnitude of the coefficients, especially higher orders.
    # Let's use a diagonal penalty matrix D.
    
    # A common choice is D = diag([0, 0, 1, 10, 100, 1000])
    # This penalizes coefficients 2, 3, 4, 5.
    
    D = np.diag([1.0, 1.0, 10.0, 100.0, 1000.0, 10000.0])
    
    # We can scale D by a factor lambda.
    # Let's try lambda in logspace.
    
    # To speed up, we can precompute the Cholesky decomposition of V^T V + lambda D.
    # But since V is large, we just solve the linear system for each lambda.
    
    # Let's try a few lambdas: 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0
    # We need to scale lambda relative to the problem size.
    # A good scale is lambda ~ (y^T y) / n / (trace(D))
    # Let's just try a range of values and see which one gives the best validation error.
    
    y_var = np.var(y)
    if y_var < 1e-12:
        return np.zeros(6, dtype=np.float64)
    
    # Scale lambda
    base_lambda = 1.0 / (n * np.trace(D))
    
    # Try different multipliers
    multipliers = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
    
    best_coeff = None
    best_val_error = np.inf
    
    for mult in multipliers:
        if time.time() - t_start > deadline - 5.0:
            break
        
        lam = base_lambda * mult
        A = V_train.T @ V_train + lam * D
        b = V_train.T @ y_train
        
        try:
            c_train = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            c_train = np.linalg.lstsq(A, b, rcond=None)[0]
        
        # Evaluate on validation set
        pred_val = V_val @ c_train
        val_error = np.mean((pred_val - y_val) ** 2)
        
        if val_error < best_val_error:
            best_val_error = val_error
            best_coeff = c_train
    
    if best_coeff is None:
        # Fallback to simple least squares
        try:
            best_coeff = np.linalg.lstsq(V, y, rcond=None)[0]
        except:
            return np.zeros(6, dtype=np.float64)
    
    # Now, we have the coefficients in the normalized space.
    # We need to convert them back to the original space.
    # x_norm = (x - x_mean) / x_std
    # x = x_std * x_norm + x_mean
    #
    # The polynomial in x_norm is: p(x_norm) = sum(c_i * x_norm^i)
    # We want q(x) = p((x - x_mean) / x_std)
    #
    # This is a polynomial transformation.
    # Let's use numpy.polynomial.polynomial.polyfromroots or similar?
    # Actually, we can just evaluate the polynomial on a grid and refit?
    # No, that's inefficient.
    #
    # Let's do the algebraic transformation.
    # p(z) = c0 + c1 z + c2 z^2 + ... + c5 z^5
    # z = (x - m) / s
    # x = s z + m
    #
    # We want to express p((x-m)/s) as a polynomial in x.
    # This can be done by expanding the terms.
    #
    # Alternatively, we can just return the coefficients in the normalized space?
    # NO, the task says "approximating the unresolved tendency as sum(c_i*x**i)" where x is the local slow state.
    # So we MUST convert back to the original x scale.
    
    # Let's perform the conversion.
    # We can use the fact that the polynomial in x is the same function, just with different coefficients.
    # We can evaluate the normalized polynomial on a dense grid of x values and then fit a new polynomial in x.
    # This is a robust and simple way to do the conversion, avoiding complex algebra.
    
    # Create a grid of x values
    x_grid = np.linspace(x_min, x_max, 100)
    
    # Evaluate the normalized polynomial on the grid
    x_norm_grid = (x_grid - (x_max + x_min) / 2.0) / ((x_max - x_min) / 2.0)
    V_grid = np.vander(x_norm_grid, N=6, increasing=True)
    y_grid = V_grid @ best_coeff
    
    # Fit a new polynomial in x to these points
    try:
        # Use polyfit to get coefficients in descending powers
        coeff_desc = np.polyfit(x_grid, y_grid, 5)
        # Convert to ascending order
        coeff_asc = coeff_desc[::-1]
        
        # Ensure all coefficients are finite
        if not np.all(np.isfinite(coeff_asc)):
            # Fallback to linear fit if numerical issues
            coeff_desc = np.polyfit(x_grid, y_grid, 1)
            coeff_asc = np.zeros(6)
            coeff_asc[0] = coeff_desc[1]
            coeff_asc[1] = coeff_desc[0]
            
    except Exception:
        # Fallback to simple linear fit on raw data if everything fails
        try:
            coeff = np.polynomial.polynomial.polyfit(x, y, 1)
            out = np.zeros(6)
            out[:2] = coeff
            return out
        except Exception:
            return np.zeros(6, dtype=np.float64)
            
    return coeff_asc
# EVOLVE-BLOCK-END
