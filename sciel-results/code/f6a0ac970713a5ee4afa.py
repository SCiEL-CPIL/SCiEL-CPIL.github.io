"""Strategy: Stable Polynomial Fit using Chebyshev Basis with Conversion to Monomial Coefficients."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def construct(training_x, training_subgrid_tendency):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    x = np.asarray(training_x, dtype=np.float64).ravel()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()
    
    n = len(x)
    if n < 10:
        return np.zeros(6)
    
    deg = 5
    
    # 1. Scale x to [-1, 1] for Chebyshev stability
    # Using min-max scaling to map range of x to [-1, 1]
    x_min = np.min(x)
    x_max = np.max(x)
    x_range = x_max - x_min
    
    if x_range < 1e-12:
        # Degenerate case, return zeros or simple fit
        return np.zeros(6)
        
    # Map x to z in [-1, 1]
    # z = (2 * (x - x_min) / x_range) - 1
    # x = (z + 1) * x_range / 2 + x_min
    # x = z * (x_range / 2) + (x_min + x_max) / 2
    # Let a = x_range / 2, b = (x_min + x_max) / 2
    a = x_range / 2.0
    b = (x_min + x_max) / 2.0
    
    z = (x - b) / a
    
    # 2. Construct Chebyshev Basis Matrix T
    # T[i, j] = T_j(z_i) where T_j is the j-th Chebyshev polynomial of first kind
    # T_0(x) = 1
    # T_1(x) = x
    # T_{n+1}(x) = 2*x*T_n(x) - T_{n-1}(x)
    
    T = np.zeros((n, deg + 1))
    T[:, 0] = 1.0
    if deg >= 1:
        T[:, 1] = z
        for k in range(2, deg + 1):
            T[:, k] = 2.0 * z * T[:, k-1] - T[:, k-2]
    
    # 3. Fit Chebyshev Coefficients
    # We can use simple least squares here as the Chebyshev basis is well-conditioned
    try:
        cheb_coeffs, residuals, rank, sv = np.linalg.lstsq(T, y, rcond=None)
    except np.linalg.LinAlgError:
        return np.zeros(6)
    
    # 4. Convert Chebyshev Coefficients to Monomial Coefficients in Scaled Space (z)
    # We need to express p(z) = sum_{k=0}^deg c_k z^k
    # Given p(z) = sum_{k=0}^deg a_k T_k(z)
    
    # We can do this by evaluating the Chebyshev polynomial expansion at a set of points
    # and solving for the monomial coefficients, OR by using the recurrence relation.
    # A robust way is to construct the conversion matrix.
    
    # Let's use the property that T_k(z) can be written as a polynomial in z.
    # We will build a matrix M such that T_k(z) = sum_{j=0}^k M[k, j] z^j
    # Then p(z) = sum_{k=0}^deg a_k T_k(z) = sum_{j=0}^deg (sum_{k=j}^deg a_k M[k, j]) z^j
    
    # Build M
    M = np.zeros((deg + 1, deg + 1))
    M[0, 0] = 1.0
    if deg >= 1:
        M[1, 1] = 1.0
        for k in range(2, deg + 1):
            # T_k = 2z T_{k-1} - T_{k-2}
            # The polynomial for T_{k-1} is M[k-1, :]
            # z * T_{k-1} shifts the coefficients by 1
            # M[k, 1:] = 2 * M[k-1, :-1]
            M[k, 1:] = 2.0 * M[k-1, :-1]
            # - T_{k-2}
            M[k, :] -= M[k-2, :]
            
    # Now, monomial_coeffs_z[j] = sum_{k=j}^deg cheb_coeffs[k] * M[k, j]
    monomial_coeffs_z = np.zeros(deg + 1)
    for j in range(deg + 1):
        monomial_coeffs_z[j] = np.dot(cheb_coeffs[j:], M[j:, j])
        
    # 5. Convert Monomial Coefficients from z-space to x-space
    # p(x) = p(z(x)) where z = (x - b) / a
    # z^k = ( (x - b) / a )^k = (1/a^k) * (x - b)^k
    # (x - b)^k = sum_{j=0}^k binom(k, j) x^j (-b)^(k-j)
    
    monomial_coeffs_x = np.zeros(deg + 1)
    
    for k in range(deg + 1):
        coeff_k_z = monomial_coeffs_z[k]
        if coeff_k_z == 0.0:
            continue
            
        # Expand (x - b)^k
        # We need to add coeff_k_z / a^k * (x - b)^k to the polynomial in x
        # (x - b)^k = sum_{j=0}^k C(k, j) x^j (-b)^(k-j)
        
        factor = coeff_k_z / (a ** k)
        
        for j in range(k + 1):
            # binom(k, j)
            binom_val = 1.0
            # Calculate binomial coefficient k choose j
            # Using factorial might be slow/overflow, but k is small (<=5)
            # Simple calculation
            # C(k, j) = k! / (j! (k-j)!)
            # For small k, we can just compute it
            # Or use the recurrence
            # Let's just use a helper or simple loop for small k
            # C(k, j)
            c_val = 1.0
            for i in range(j):
                c_val *= (k - i) / (i + 1.0)
                
            term = factor * c_val * ((-b) ** (k - j))
            monomial_coeffs_x[j] += term
            
    # 6. Check Time
    if time.time() > deadline:
        return monomial_coeffs_x
        
    return monomial_coeffs_x
# EVOLVE-BLOCK-END
