import numpy as np

def construct(training_x, training_subgrid_tendency):
    x = np.asarray(training_x, dtype=float)
    y = np.asarray(training_subgrid_tendency, dtype=float)
    
    # Handle potential edge cases
    if len(x) == 0:
        return np.zeros(6)
        
    # Remove NaNs
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    
    if len(x) < 6:
        # Not enough data for degree 5, fall back to lower degree or zeros
        # But task implies enough data. Just return zeros or linear if possible.
        if len(x) == 0:
            return np.zeros(6)
        deg = min(5, len(x) - 1)
        if deg < 0:
            return np.zeros(6)
        coeffs_shifted = np.polynomial.polynomial.polyfit(x, y, deg)
        # Pad to 6
        coeffs_shifted = np.pad(coeffs_shifted, (0, 5 - len(coeffs_shifted)))
    else:
        coeffs_shifted = np.polynomial.polynomial.polyfit(x, y, 5)

    # Center x to improve conditioning
    mu = np.mean(x)
    
    # Transform coefficients from (x - mu) basis to x basis
    # P(x) = sum_{k=0}^5 a_k (x - mu)^k
    # (x - mu)^k = sum_{j=0}^k binom(k, j) x^j (-mu)^(k-j)
    # b_j = sum_{k=j}^5 a_k * binom(k, j) * (-mu)^(k-j)
    
    coeffs_orig = np.zeros(6)
    for j in range(6):
        s = 0.0
        for k in range(j, 6):
            # binom(k, j)
            # k is small (<=5), so we can compute directly
            # binom(k, j) = k! / (j! * (k-j)!)
            # Using math.comb is available in Python 3.8+
            import math
            binom_val = math.comb(k, j)
            s += coeffs_shifted[k] * binom_val * ((-mu) ** (k - j))
        coeffs_orig[j] = s
        
    return coeffs_orig
