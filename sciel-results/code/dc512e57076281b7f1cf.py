import numpy as np
import time

def construct(training_x, training_subgrid_tendency):
    start_time = time.time()
    deadline = 110.0  # Safety margin before 120s limit
    
    x = np.asarray(training_x, dtype=np.float64).flatten()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).flatten()
    
    n = len(x)
    if n == 0:
        return np.zeros(6)
    
    if n < 6:
        # Not enough data for degree 5, use lower degree
        deg = min(5, n - 1)
        coeff = np.zeros(6)
        if deg >= 0:
            coeff[:deg+1] = np.polynomial.polynomial.polyfit(x, y, deg)
        return coeff
    
    # Center and scale x for numerical stability
    x_mean = np.mean(x)
    x_std = np.std(x)
    if x_std < 1e-12:
        x_std = 1.0
    
    x_scaled = (x - x_mean) / x_std
    
    # Build Vandermonde matrix for degree 5 in scaled x
    # V[i, j] = x_scaled[i]^j
    V = np.vander(x_scaled, N=6, increasing=True)
    
    # Regularized Least Squares (Ridge Regression) to prevent overfitting
    # The unregularized degree-5 fit likely overfits, causing poor generalization.
    VT = V.T
    A = VT @ V
    b = VT @ y
    
    # Adaptive regularization strength
    diag_mean = np.mean(np.diag(A))
    if diag_mean < 1e-12:
        diag_mean = 1.0
        
    lam = 1e-3 * diag_mean
    
    I = np.eye(6)
    A_reg = A + lam * I
    
    try:
        coeff_scaled = np.linalg.solve(A_reg, b)
    except np.linalg.LinAlgError:
        # Fallback to unregularized lstsq if solve fails
        coeff_scaled, residuals, rank, sv = np.linalg.lstsq(V, y, rcond=None)
        
    # Convert from scaled polynomial to original polynomial
    # y = sum(d_j * (x - x_mean)^j) where d_j = coeff_scaled[j] / x_std**j
    # (x - x_mean)^j = sum_{k=0}^{j} C(j,k) * x^k * (-x_mean)^(j-k)
    # C_orig[k] = sum_{j=k}^{5} d_j * C(j,k) * (-x_mean)^(j-k)
    
    coeff_orig = np.zeros(6)
    
    # Precompute binomial coefficients for j up to 5
    binom = np.zeros((6, 6))
    for j in range(6):
        for k in range(j + 1):
            if k == 0 or k == j:
                binom[j, k] = 1.0
            else:
                binom[j, k] = binom[j-1, k-1] + binom[j-1, k]
                
    for k in range(6):
        sum_val = 0.0
        for j in range(k, 6):
            d_j = coeff_scaled[j] / (x_std ** j)
            sum_val += d_j * binom[j, k] * ((-x_mean) ** (j - k))
        coeff_orig[k] = sum_val
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > deadline:
        # Return what we have (shouldn't happen for simple lstsq)
        return coeff_orig
    
    # Verify coefficients are finite
    if not np.all(np.isfinite(coeff_orig)):
        # Fallback
        coeff = np.polynomial.polynomial.polyfit(x, y, 5)
        return coeff
    
    return coeff_orig
