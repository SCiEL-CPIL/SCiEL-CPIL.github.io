import numpy as np
import time
from math import comb
from scipy.stats import norm

def construct(training_x, training_subgrid_tendency):
    t_start = time.time()
    deadline = time.time() + 115.0  # Safety margin before 120s limit

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

    if n < 6:
        deg = max(0, n - 1)
        try:
            coeff = np.polynomial.polynomial.polyfit(x, y, deg)
            full_coeff = np.zeros(6, dtype=np.float64)
            full_coeff[:len(coeff)] = coeff
            return full_coeff
        except Exception:
            return np.zeros(6, dtype=np.float64)

    # Scale x to [-1, 1] for GPR and Chebyshev fit
    x_min, x_max = x.min(), x.max()
    x_range = x_max - x_min
    if x_range < 1e-12:
        return np.array([y.mean(), 0, 0, 0, 0, 0], dtype=np.float64)

    x_center = (x_min + x_max) / 2.0
    x_half_range = x_range / 2.0
    x_scaled = (x - x_center) / x_half_range

    # --- Step 1: Denoise with GPR ---
    # To keep it fast, subsample if n is large
    n_max_gpr = 400
    if n > n_max_gpr:
        # Random subsample for GPR training
        rng = np.random.default_rng(42)
        idx_sub = rng.choice(n, n_max_gpr, replace=False)
        x_gpr = x_scaled[idx_sub]
        y_gpr = y[idx_sub]
    else:
        x_gpr = x_scaled
        y_gpr = y
        
    n_gpr = x_gpr.size
    
    # Hyperparameters for RBF Kernel
    # Length scale: ~ 1/10 of the range in scaled units (range is 2)
    # So l ~ 0.2
    l = 0.2 
    # Noise variance: estimate from y
    sigma_n2 = np.var(y) * 0.1
    
    # Build Kernel Matrix K + sigma_n2 * I
    # K_ij = exp(-(x_i - x_j)^2 / (2 * l^2))
    diff = x_gpr[:, None] - x_gpr[None, :]
    K = np.exp(-diff**2 / (2 * l**2))
    K += sigma_n2 * np.eye(n_gpr)
    
    # Solve for weights: w = K^{-1} y
    try:
        w = np.linalg.solve(K, y_gpr)
    except np.linalg.LinAlgError:
        # Fallback to least squares if singular
        try:
            w, _, _, _ = np.linalg.lstsq(K, y_gpr, rcond=None)
        except Exception:
            # Fallback to original y if GPR fails
            w = None

    if w is not None:
        # Predict means for all x_scaled
        # K_star = exp(-(x_scaled - x_gpr)^2 / (2 * l^2))
        diff_star = x_scaled[:, None] - x_gpr[None, :]
        K_star = np.exp(-diff_star**2 / (2 * l**2))
        y_denoised = K_star @ w
    else:
        y_denoised = y

    # --- Step 2: Fit Chebyshev Polynomial to Denoised Data ---
    
    # Build Chebyshev Vandermonde matrix (degree 0 to 5)
    T = np.zeros((n, 6), dtype=np.float64)
    T[:, 0] = 1.0
    T[:, 1] = x_scaled
    T[:, 2] = 2.0 * x_scaled * T[:, 1] - T[:, 0]
    T[:, 3] = 2.0 * x_scaled * T[:, 2] - T[:, 1]
    T[:, 4] = 2.0 * x_scaled * T[:, 3] - T[:, 2]
    T[:, 5] = 2.0 * x_scaled * T[:, 4] - T[:, 3]

    # Fit using Least Squares (data is denoised)
    # Add small regularization to ensure stability
    TtT = T.T @ T
    Tty = T.T @ y_denoised
    
    # Small lambda for numerical stability
    lam = 1e-8 * np.trace(TtT) / 6.0
    A = TtT + lam * np.eye(6)
    
    try:
        cheb_coeff = np.linalg.solve(A, Tty)
    except np.linalg.LinAlgError:
        try:
            cheb_coeff, _, _, _ = np.linalg.lstsq(A, Tty, rcond=None)
        except Exception:
            return np.zeros(6, dtype=np.float64)

    if not np.all(np.isfinite(cheb_coeff)):
        return np.zeros(6, dtype=np.float64)

    # Convert Chebyshev coeffs to Monomial coeffs in scaled variable z
    a0, a1, a2, a3, a4, a5 = cheb_coeff
    
    b0 = a0 - a2 + a4
    b1 = a1 - 3*a3 + 5*a5
    b2 = 2*a2 - 8*a4
    b3 = 4*a3 - 20*a5
    b4 = 8*a4
    b5 = 16*a5
    
    b = np.array([b0, b1, b2, b3, b4, b5], dtype=np.float64)
    
    # Convert from monomial coeffs in z to monomial coeffs in x
    c = x_center
    s = x_half_range
    
    coeff_orig = np.zeros(6, dtype=np.float64)
    
    for j in range(6):
        bj = b[j]
        if abs(bj) < 1e-15:
            continue
        inv_s_pow = 1.0 / (s ** j)
        for k in range(j + 1):
            binom_coeff = comb(j, k)
            power_c = (-c) ** (j - k)
            coeff_orig[k] += bj * (binom_coeff * power_c) * inv_s_pow
            
    if not np.all(np.isfinite(coeff_orig)):
        return np.zeros(6, dtype=np.float64)

    return coeff_orig.astype(np.float64)
