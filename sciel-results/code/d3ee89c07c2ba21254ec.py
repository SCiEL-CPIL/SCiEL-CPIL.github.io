"""Strategy: Nadaraya-Watson denoising + degree-5 polynomial fit with time guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def construct(training_x, training_subgrid_tendency):
    start_time = time.time()
    # Set a conservative deadline to ensure we return well before the 150s kill limit
    deadline = start_time + 120.0 

    x = np.asarray(training_x, dtype=np.float64).ravel()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()
    
    n = len(x)
    
    # Basic sanity checks
    if n < 6:
        return np.zeros(6)
        
    # Remove any NaNs or Infs
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = len(x)
    
    if n < 6:
        return np.zeros(6)
        
    if np.all(x == x[0]) or np.all(y == y[0]):
        return np.zeros(6)

    # --- Step 1: Nadaraya-Watson Denoising ---
    # Estimate bandwidth using Silverman's rule of thumb on x
    x_std = np.std(x)
    if x_std < 1e-10:
        x_std = 1.0
        
    # Adjust bandwidth for dimensionality (1D here) and sample size
    # Standard Silverman: 1.06 * sigma * n^(-1/5)
    bw = 1.06 * x_std * (n ** (-0.2))
    
    # Ensure bandwidth is not too small to cause numerical issues or too large to flatten everything
    # Heuristic: at least 1/20th of the range, at most 1/2 of the range
    x_range = x.max() - x.min()
    if x_range > 1e-10:
        bw = np.clip(bw, x_range / 50.0, x_range / 2.0)
    else:
        bw = 1.0

    # To manage memory and time for large N, we can subsample the kernel evaluation
    # if N is very large, but for typical climate model datasets (N ~ 10^4-10^5), 
    # direct evaluation in chunks is safer.
    
    y_smoothed = np.empty(n)
    
    # Process in chunks to avoid creating massive temporary arrays (N x N)
    chunk_size = 2000 
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        x_chunk = x[start:end]
        
        # Calculate weights: K((x_j - x_i) / bw)
        # x_chunk shape: (chunk, 1)
        # x shape: (1, n)
        diff = (x_chunk[:, None] - x[None, :]) / bw
        # Gaussian kernel
        weights = np.exp(-0.5 * diff * diff)
        
        w_sum = np.sum(weights, axis=1)
        # Avoid division by zero
        w_sum[w_sum < 1e-300] = 1e-300
        
        y_smoothed[start:end] = np.sum(weights * y[None, :], axis=1) / w_sum
        
        # Time check
        if time.time() > deadline - 10.0:
            # If we are running out of time, fall back to using original y for remaining points
            # and proceed to polynomial fit immediately
            y_smoothed[end:] = y[end:]
            break

    # --- Step 2: Polynomial Fit ---
    # We want to fit y_smoothed = sum_{i=0}^5 c_i * x^i
    # For numerical stability, we fit on scaled x: t = (x - x_mean) / x_std
    # y = p(t) = sum_{i=0}^5 a_i * t^i
    # Then transform a_i to c_i.
    
    x_mean = np.mean(x)
    if x_std < 1e-10:
        x_std = 1.0
        
    x_scaled = (x - x_mean) / x_std
    
    # Fit degree 5 polynomial on scaled x
    try:
        # Use least squares on the Vandermonde matrix of scaled x
        V_scaled = np.vander(x_scaled, N=6, increasing=True)
        # Add tiny regularization to handle potential collinearity or rank deficiency
        # Ridge regression: (V^T V + lambda I) a = V^T y
        # Lambda is very small to just stabilize, not bias significantly
        lambda_reg = 1e-8
        A = V_scaled.T @ V_scaled + lambda_reg * np.eye(6)
        b = V_scaled.T @ y_smoothed
        a = np.linalg.solve(A, b)
    except Exception:
        # Fallback to standard polyfit if solve fails
        try:
            a = np.polynomial.polynomial.polyfit(x_scaled, y_smoothed, 5)
        except Exception:
            # Last resort: constant fit
            return np.array([np.mean(y_smoothed), 0, 0, 0, 0, 0])

    # --- Step 3: Transform Coefficients ---
    # We have p(t) = sum_{i=0}^5 a_i t^i where t = (x - x_mean) / x_std
    # We need c such that p(x) = sum_{j=0}^5 c_j x^j
    # t = (1/x_std) * x - (x_mean/x_std)
    # Let A = 1/x_std, B = -x_mean/x_std
    # t = A*x + B
    # p(x) = sum_{i=0}^5 a_i (A*x + B)^i
    # Expand (A*x + B)^i = sum_{j=0}^i binom(i, j) (A*x)^j B^(i-j)
    #                = sum_{j=0}^i binom(i, j) A^j B^(i-j) x^j
    
    c = np.zeros(6)
    A_coeff = 1.0 / x_std
    B_coeff = -x_mean / x_std
    
    for i in range(6):
        if a[i] == 0:
            continue
        for j in range(i + 1):
            # Calculate binom(i, j)
            binom = 1.0
            for k in range(j):
                binom *= (i - k) / (k + 1)
            
            term = a[i] * binom * (A_coeff ** j) * (B_coeff ** (i - j))
            c[j] += term

    # --- Step 4: Final Safety Checks ---
    if not np.all(np.isfinite(c)):
        # If transformation produced non-finite values (e.g. overflow), fallback to direct fit on raw x
        try:
            c = np.polynomial.polynomial.polyfit(x, y_smoothed, 5)
        except:
            c = np.zeros(6)
            
    # Clip extremely large coefficients to prevent overflow during evaluation
    max_mag = np.max(np.abs(c))
    if max_mag > 1e10:
        c = np.clip(c, -1e10, 1e10)
        
    return c

# EVOLVE-BLOCK-END
