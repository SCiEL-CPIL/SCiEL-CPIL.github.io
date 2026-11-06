"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def construct(training_x, training_subgrid_tendency):
    deadline = time.time() + 100  # safety margin before 120s limit
    
    x = np.asarray(training_x, dtype=np.float64)
    y = np.asarray(training_subgrid_tendency, dtype=np.float64)
    
    # Remove NaN/Inf if any
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    
    if len(x) < 6:
        return np.zeros(6)
    
    # Try full degree-5 polynomial with various regularization strengths
    # The subgrid tendency in Lorenz-96 is h*c*mean(Y), which is an odd function
    # of X on average (by symmetry), so we try odd-only first, then full
    
    best_coeff = np.zeros(6)
    best_mse = np.inf
    
    # Normalize x for better conditioning
    x_mean = np.mean(x)
    x_std = np.std(x) if np.std(x) > 1e-12 else 1.0
    x_norm = (x - x_mean) / x_std
    
    # Build Vandermonde matrix for degree-5 in normalized space
    V = np.vander(x_norm, N=6, increasing=True)
    
    # Try several regularization strengths
    for lam in [0.0, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 0.1, 1.0, 10.0, 100.0]:
        if time.time() > deadline - 10:
            break
            
        try:
            # Regularized least squares
            A = V.T @ V + lam * np.eye(6)
            b = V.T @ y
            coeff_norm = np.linalg.solve(A, b)
            
            # Convert from normalized to original scale
            # If we fit y = sum(c_i * x_norm^i), and x_norm = (x - x_mean)/x_std
            # We need to convert back to original x powers
            # This is complex; instead, fit directly in original scale with scaling
            
            # Evaluate in normalized space
            pred = V @ coeff_norm
            mse = np.mean((pred - y) ** 2)
            
            if mse < best_mse:
                best_mse = mse
                best_coeff_norm = coeff_norm
        except:
            continue
    
    if time.time() > deadline - 5:
        # Return best we have (zeros if nothing worked)
        return np.zeros(6)
    
    # Convert normalized coefficients to original scale
    # y = sum_{i=0}^{5} c_i * ((x - x_mean)/x_std)^i
    #   = sum_{i=0}^{5} c_i * (1/x_std^i) * (x - x_mean)^i
    # Expand (x - x_mean)^i = sum_{j=0}^{i} binom(i,j) * x^j * (-x_mean)^(i-j)
    
    coeff_orig = np.zeros(6)
    for i in range(6):
        ci = best_coeff_norm[i] / (x_std ** i)
        for j in range(i + 1):
            from math import comb
            coeff_orig[j] += ci * comb(i, j) * ((-x_mean) ** (i - j))
    
    # Verify coefficients are finite
    if not np.all(np.isfinite(coeff_orig)):
        return np.zeros(6)
    
    return coeff_orig
# EVOLVE-BLOCK-END
