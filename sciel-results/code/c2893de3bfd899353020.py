"""
Rewritten program for unresolved-tendency closure via polynomial fit.
Uses scaled degree-5 polynomial regression with coefficient transformation back to original basis.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time


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

    if n < 6:
        # Not enough data for degree 5, fall back to linear or constant
        if n == 0:
            return np.zeros(6, dtype=np.float64)
        if n == 1:
            coeff = np.zeros(6)
            coeff[0] = y[0]
            return coeff
        # Fit linear if n < 6
        coeff = np.zeros(6)
        try:
            c_lin = np.polynomial.polynomial.polyfit(x, y, 1)
            coeff[:2] = c_lin
            return coeff
        except Exception:
            return np.zeros(6, dtype=np.float64)

    # Check time
    if time.time() - t_start > deadline:
        return np.zeros(6, dtype=np.float64)

    # Scaling for numerical stability
    mu = np.mean(x)
    sigma = np.std(x)
    if sigma < 1e-12:
        sigma = 1.0
    
    x_scaled = (x - mu) / sigma

    # Construct Vandermonde matrix for scaled x, degree 5
    # V[i, j] = x_scaled[i]^j
    V = np.vander(x_scaled, N=6, increasing=True)
    
    # Solve least squares
    # Using lstsq for robustness
    try:
        c_scaled, residuals, rank, s = np.linalg.lstsq(V, y, rcond=None)
    except np.linalg.LinAlgError:
        # Fallback to polyfit if lstsq fails (unlikely)
        c_scaled = np.polynomial.polynomial.polyfit(x_scaled, y, 5)
    
    # Check time
    if time.time() - t_start > deadline:
        return np.zeros(6, dtype=np.float64)

    # Transform coefficients from scaled basis to original basis
    # We have P_scaled(u) = sum_{k=0}^5 c_scaled[k] * u^k
    # where u = (x - mu) / sigma
    # We want P_orig(x) = sum_{j=0}^5 c_orig[j] * x^j
    # P_orig(x) = P_scaled((x - mu) / sigma)
    # = sum_{k=0}^5 c_scaled[k] * ( (x - mu) / sigma )^k
    # = sum_{k=0}^5 (c_scaled[k] / sigma^k) * (x - mu)^k
    # Expand (x - mu)^k = sum_{j=0}^k binom(k, j) * x^j * (-mu)^(k-j)
    # So c_orig[j] = sum_{k=j}^5 (c_scaled[k] / sigma^k) * binom(k, j) * (-mu)^(k-j)
    
    c_orig = np.zeros(6, dtype=np.float64)
    for j in range(6):
        sum_val = 0.0
        for k in range(j, 6):
            # Binomial coefficient k choose j
            # Since k is small (<=5), we can compute directly or use math.comb
            # k! / (j! * (k-j)!)
            # Precomputing binom for small numbers is fast
            binom_val = 1.0
            if k > j:
                # Calculate C(k, j)
                # C(k, j) = k * (k-1) * ... * (k-j+1) / j!
                num = 1.0
                for i in range(j):
                    num *= (k - i)
                den = 1.0
                for i in range(1, j+1):
                    den *= i
                binom_val = num / den
            
            term = (c_scaled[k] / (sigma ** k)) * binom_val * ((-mu) ** (k - j))
            sum_val += term
        c_orig[j] = sum_val

    return c_orig
# EVOLVE-BLOCK-END
