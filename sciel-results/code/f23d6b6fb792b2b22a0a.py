"""Search for optimal Hermite combination coefficients minimizing C4 upper bound."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

# Deadline guard
START_TIME = time.time()
TIME_LIMIT = 180.0  # 210s kill, keep safety margin


def hermite_h(n, x):
    """Physicist's Hermite polynomial H_n(x)."""
    if n == 0:
        return np.ones_like(x)
    elif n == 1:
        return 2 * x
    elif n == 2:
        return 4 * x**2 - 2
    elif n == 3:
        return 8 * x**3 - 12 * x
    elif n == 4:
        return 16 * x**4 - 48 * x**2 + 12
    elif n == 8:
        return 256 * x**8 - 3072 * x**6 + 11520 * x**4 - 13440 * x**2 + 1680
    elif n == 12:
        return 4096 * x**12 - 73728 * x**10 + 552960 * x**8 - 2150400 * x**6 + 3991680 * x**4 - 3024000 * x**2 + 92400
    else:
        H = np.ones_like(x)
        if n >= 1:
            H1 = 2 * x
            for k in range(2, n + 1):
                H, H1 = H1, 2 * x * H1 - 2 * (k - 1) * H
        return H1 if n >= 1 else H


def g_of_x(x, c0, c4, c8):
    """g(x) = (c0*H0 + c4*H4 + c8*H8 + c12*H12) * exp(-x^2/2)
    with c12 chosen so g(0) = 0."""
    H0 = np.ones_like(x)
    H4 = hermite_h(4, x)
    H8 = hermite_h(8, x)
    H12 = hermite_h(12, x)
    c12 = -(c0 + 12 * c4 + 1680 * c8) / 92400.0
    poly = c0 * H0 + c4 * H4 + c8 * H8 + c12 * H12
    return poly * np.exp(-0.5 * x**2)


def compute_c4(c0, c4, c8):
    """Compute the C4 upper bound for the given coefficients."""
    x = np.linspace(-15, 15, 20001)
    dx = x[1] - x[0]
    g = g_of_x(x, c0, c4, c8)
    
    linf_g = np.max(np.abs(g))
    
    n = len(x)
    G = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(g))) * dx
    linf_G = np.max(np.abs(G))
    
    c4_val = linf_g * linf_G
    
    return c4_val


def objective(coeffs):
    """Objective function to minimize: the C4 upper bound."""
    c0, c4, c8 = coeffs
    try:
        val = compute_c4(c0, c4, c8)
        if not np.isfinite(val):
            return 1e10
        return val
    except:
        return 1e10


def construct():
    """Find optimal coefficients (c0, c4, c8) minimizing the C4 upper bound."""
    deadline = START_TIME + TIME_LIMIT - 10.0
    
    # Seed coefficients
    seed_coeffs = np.array([-113.0/100, 1.0/25, 1.0/3240])
    best_coeffs = seed_coeffs.copy()
    best_val = objective(best_coeffs)
    
    # Structural Change: Robust Local Search (Hill Climbing)
    # Instead of blind global optimization which might diverge,
    # we perform a series of local refinements that only accept improvements.
    
    # 1. Coarse Grid Search around seed
    # Define a small neighborhood
    step = 0.05
    grid_range = np.arange(-3, 4, 1) # -3 to 3 steps
    
    # To save time, we do a sparse grid first
    for i in grid_range:
        for j in grid_range:
            for k in grid_range:
                if time.time() > deadline:
                    break
                candidate = seed_coeffs + np.array([i*step, j*step, k*step])
                val = objective(candidate)
                if val < best_val:
                    best_val = val
                    best_coeffs = candidate.copy()
    
    # 2. Fine Grid Search around the best coarse point
    step_fine = 0.01
    grid_range_fine = np.arange(-2, 3, 1)
    for i in grid_range_fine:
        for j in grid_range_fine:
            for k in grid_range_fine:
                if time.time() > deadline:
                    break
                candidate = best_coeffs + np.array([i*step_fine, j*step_fine, k*step_fine])
                val = objective(candidate)
                if val < best_val:
                    best_val = val
                    best_coeffs = candidate.copy()
    
    # 3. Nelder-Mead polish from the best grid point
    if time.time() < deadline:
        try:
            res = minimize(objective, best_coeffs, method='Nelder-Mead',
                          options={'maxiter': 100, 'xatol': 1e-6, 'fatol': 1e-8})
            if res.fun < best_val:
                best_val = res.fun
                best_coeffs = res.x
        except:
            pass
            
    # 4. Final check: if we are worse than seed, revert to seed
    if best_val > objective(seed_coeffs):
        best_coeffs = seed_coeffs

    return np.array([best_coeffs[0], best_coeffs[1], best_coeffs[2]])
# EVOLVE-BLOCK-END
