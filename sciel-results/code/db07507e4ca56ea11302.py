"""
First autocorrelation inequality: minimize normalized max of autoconvolution
for a nonnegative step function with 600 equal intervals on [-1/4, 1/4].

Strategy: 
1. Use spectral (FFT-based) computation of autoconvolution for speed.
2. Employ a two-phase optimization:
   Phase 1: Optimize in a reduced cosine basis to find a good smooth shape.
   Phase 2: Refine in full 600-dim space using projected gradient descent
            with careful step-size selection and peak-targeting.
3. Use a smooth max (log-sum-exp) surrogate during optimization for differentiability,
   then evaluate the true max at the end.
"""

import numpy as np
from scipy.optimize import minimize
import time

N = 600
DT = (1.0 / 4.0 - (-1.0 / 4.0)) / N  # interval width = 0.5/600 = 1/1200
TIME_LIMIT = 130  # safety margin before 150s kill


def fft_autocorr(h):
    """Compute discrete autocorrelation of h using FFT.
    Returns array of length N where result[m] = sum_k h[k] * h[k+m] (with proper indexing).
    """
    n_fft = 2 * N
    H = np.fft.fft(h, n=n_fft)
    ac = np.fft.ifft(H * np.conj(H)).real[:N]
    return ac


def compute_score(h):
    """Compute the normalized maximum of the autoconvolution.
    
    For a step function with heights h on intervals of width DT:
    - The continuous autoconvolution peak is approximated by max(discrete_autocorr) * DT
    - The integral is sum(h) * DT
    - Normalized peak = peak / (integral^2) = max(ac) * DT / (sum(h)*DT)^2 = max(ac) / (DT * sum(h)^2)
    """
    h = np.maximum(h, 0)
    total = np.sum(h)
    if total < 1e-12:
        return 0.0
    ac = fft_autocorr(h)
    peak = np.max(ac) * DT
    integral = total * DT
    normalized_peak = peak / (integral ** 2)
    return normalized_peak


def smooth_max(values, temperature=0.01):
    """Smooth approximation of max using log-sum-exp."""
    values = np.asarray(values)
    m = np.max(values)
    if m < 1e-12:
        return 0.0
    return m + temperature * np.log(np.sum(np.exp((values - m) / temperature)))


def objective_smooth(h, temperature=0.01):
    """Smooth objective for optimization."""
    h = np.maximum(h, 0)
    total = np.sum(h)
    if total < 1e-12:
        return 0.0
    ac = fft_autocorr(h)
    # Normalize: we want max(ac) / (sum(h)^2 * DT)
    # For smooth max: use smooth_max(ac) / (sum(h)^2 * DT)
    sm = smooth_max(ac, temperature)
    return sm / (total ** 2 * DT)


def cosine_basis_coeffs_to_heights(coeffs, n_basis=40):
    """Convert cosine basis coefficients to step function heights.
    h[i] = sum_k coeffs[k] * cos(2*pi*k*i/N) + coeffs[0] (DC offset)
    Ensure nonnegativity.
    """
    i = np.arange(N)
    h = np.zeros(N)
    for k in range(n_basis):
        h += coeffs[k] * np.cos(2 * np.pi * k * i / N)
    h = np.maximum(h, 0)
    return h


def objective_cosine(coeffs, n_basis=40, temperature=0.01):
    """Objective in cosine basis space."""
    h = cosine_basis_coeffs_to_heights(coeffs, n_basis)
    return objective_smooth(h, temperature)


def projected_gradient_step(h, grad, step_size):
    """Take a gradient step and project onto nonnegative orthant."""
    h_new = h - step_size * grad
    h_new = np.maximum(h_new, 0)
    return h_new


def compute_gradient_smooth(h, temperature=0.01):
    """Compute gradient of the smooth objective with respect to h.
    
    This is complex, so we use numerical gradient as fallback, but let's try analytical.
    
    Actually, for efficiency, let's use finite differences on the smooth objective.
    """
    h = np.maximum(h, 0)
    total = np.sum(h)
    if total < 1e-12:
        return np.zeros(N)
    
    ac = fft_autocorr(h)
    sm = smooth_max(ac, temperature)
    obj = sm / (total ** 2 * DT)
    
    # Numerical gradient
    eps = 1e-6
    grad = np.zeros(N)
    for i in range(N):
        h_plus = h.copy()
        h_plus[i] += eps
        ac_plus = fft_autocorr(h_plus)
        sm_plus = smooth_max(ac_plus, temperature)
        total_plus = np.sum(h_plus)
        obj_plus = sm_plus / (total_plus ** 2 * DT)
        grad[i] = (obj_plus - obj) / eps
        if time.time() > TIME_LIMIT + 20:  # extra safety
            break
    
    return grad


def local_refinement(h, time_budget=20):
    """Local refinement using projected gradient descent with adaptive step size."""
    start = time.time()
    h = np.maximum(h.copy(), 0)
    current_obj = objective_smooth(h, 0.01)
    best_h = h.copy()
    best_obj = current_obj
    
    # Use a smaller temperature for finer optimization
    temperature = 0.005
    
    step_size = 0.01
    patience = 0
    max_patience = 200
    
    while time.time() - start < time_budget:
        # Compute gradient numerically (expensive but accurate)
        # For speed, use a subset of coordinates or analytical gradient
        # Let's use analytical gradient via FFT differentiation
        
        # Actually, let's use a simpler approach: coordinate descent
        # Pick random coordinates and try to improve them
        
        n_coords = 50
        idx = np.random.choice(N, size=n_coords, replace=False)
        
        improved = False
        for i in idx:
            if time.time() - start > time_budget:
                break
            
            h_old = h[i]
            
            # Try a few perturbations
            for delta in [-0.1, -0.05, -0.01, 0.01, 0.05, 0.1]:
                h[i] = max(h_old + delta, 0)
                obj = objective_smooth(h, temperature)
                if obj < current_obj - 1e-10:
                    current_obj = obj
                    improved = True
                    if obj < best_obj:
                        best_obj = obj
                        best_h = h.copy()
                    break
                else:
                    h[i] = h_old
        
        if not improved:
            patience += 1
            step_size *= 0.9
            if patience > max_patience:
                break
        else:
            patience = 0
    
    return best_h, best_obj


def polish_pass(h, time_budget=10):
    """Final polish using scipy minimize with tight tolerances."""
    start = time.time()
    h = np.maximum(h.copy(), 0)
    
    def obj_func(x):
        x = np.maximum(x, 0)
        return compute_score(x)
    
    bounds = [(0, None)] * N
    
    try:
        res = minimize(obj_func, h, method='L-BFGS-B', bounds=bounds,
                      options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8})
        if res.fun < compute_score(h):
            return np.maximum(res.x, 0), res.fun
    except Exception:
        pass
    
    return h, compute_score(h)


def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    # Phase 1: Optimize in cosine basis
    n_basis = 30
    
    # Initialize with DC + some low-frequency components
    coeffs0 = np.zeros(n_basis)
    coeffs0[0] = 1.0  # DC offset
    
    best_h = cosine_basis_coeffs_to_heights(coeffs0, n_basis)
    best_score = compute_score(best_h)
    
    # Optimize in cosine basis
    if time.time() < deadline - 40:
        try:
            res = minimize(objective_cosine, coeffs0, args=(n_basis, 0.01),
                          method='L-BFGS-B',
                          options={'maxiter': 300, 'ftol': 1e-8, 'gtol': 1e-6})
            if res.fun < best_score:
                best_h = cosine_basis_coeffs_to_heights(res.x, n_basis)
                best_score = compute_score(best_h)
        except Exception:
            pass
    
    # Phase 2: Local refinement in full space
    if time.time() < deadline - 15:
        best_h, best_score = local_refinement(best_h, time_budget=15)
    
    # Phase 3: Final polish
    if time.time() < deadline - 5:
        best_h, best_score = polish_pass(best_h, time_budget=10)
    
    # Ensure nonnegativity
    best_h = np.maximum(best_h, 0)
    
    # Final score with true max (not smooth)
    final_score = compute_score(best_h)
    
    return [float(x) for x in best_h]
