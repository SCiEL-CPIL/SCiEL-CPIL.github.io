"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 100

# EVOLVE-BLOCK-START

def _overlap_max(h):
    """Compute maximum overlap of h with shifted 1-h over all integer shifts."""
    w = 2.0 / N
    h = np.asarray(h, dtype=float)
    n = len(h)
    max_ov = 0.0
    for k in range(n):
        vals = np.minimum(h, 1.0 - np.roll(h, -k))
        ov = w * np.sum(vals)
        if ov > max_ov:
            max_ov = ov
    return max_ov


def _smooth_max(vals, beta=100.0):
    """Smooth approximation of max using log-sum-exp."""
    m = np.max(vals)
    if m < -1e10:
        return 0.0
    shifted = vals - m
    return m + np.log(np.sum(np.exp(beta * shifted))) / beta


def _overlap_max_smooth(h, beta=100.0):
    """Smooth max overlap for gradient computation."""
    w = 2.0 / N
    h = np.asarray(h, dtype=float)
    n = len(h)
    overlaps = np.zeros(n)
    for k in range(n):
        vals = np.minimum(h, 1.0 - np.roll(h, -k))
        overlaps[k] = w * np.sum(vals)
    return _smooth_max(overlaps, beta)


def _overlap_max_grad(h, beta=100.0):
    """Gradient of smooth max overlap via finite differences (vectorized)."""
    h = np.asarray(h, dtype=float)
    eps = 1e-6
    grad = np.zeros_like(h)
    f0 = _overlap_max_smooth(h, beta)
    for i in range(len(h)):
        h[i] += eps
        fp = _overlap_max_smooth(h, beta)
        h[i] -= 2 * eps
        fm = _overlap_max_smooth(h, beta)
        h[i] += eps
        grad[i] = (fp - fm) / (2 * eps)
    return f0, grad


def _fourier_to_h(params):
    """Convert Fourier parameters to h vector.
    params = [a0, a1, b1, a2, b2, a3, b3]
    h[i] = a0 + a1*cos(2*pi*i/N) + b1*sin(2*pi*i/N) + ...
    Then project to [0,1] and normalize sum to N/2.
    """
    n = len(params)
    a0 = params[0]
    x = np.arange(N)
    h = np.full(N, a0)
    for m in range(1, (n - 1) // 2 + 1):
        am = params[2 * m - 1]
        bm = params[2 * m]
        h += am * np.cos(2 * np.pi * m * x / N) + bm * np.sin(2 * np.pi * m * x / N)
    
    # Project to [0, 1]
    h = np.clip(h, 0.0, 1.0)
    
    # Normalize sum to N/2
    diff = N / 2.0 - np.sum(h)
    if abs(diff) > 1e-12:
        h = h + diff / N
        h = np.clip(h, 0.0, 1.0)
        diff2 = N / 2.0 - np.sum(h)
        if abs(diff2) > 1e-12:
            mask = (h > 1e-10) & (h < 1.0 - 1e-10)
            if np.any(mask):
                h[mask] += diff2 / np.sum(mask)
                h = np.clip(h, 0.0, 1.0)
    return h


def construct():
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    # Uniform baseline
    h0 = np.full(N, 0.5)
    best_h = h0.copy()
    best_score = _overlap_max(h0)
    
    # Fourier parameterization: [a0, a1, b1, a2, b2, a3, b3]
    # a0 should be ~0.5 for mean to be 0.5
    n_params = 7  # DC + 3 harmonics
    
    def fourier_objective(params):
        h = _fourier_to_h(params)
        return _overlap_max(h)
    
    # Multiple random restarts in Fourier space
    rng = np.random.default_rng(42)
    fourier_bounds = [(0.0, 1.0)]  # a0
    for _ in range(3):
        fourier_bounds.append((-0.5, 0.5))  # am
        fourier_bounds.append((-0.5, 0.5))  # bm
    
    # Try several initial points
    initial_guesses = []
    # Start near uniform
    initial_guesses.append(np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    # Some structured patterns
    initial_guesses.append(np.array([0.5, 0.3, 0.0, -0.2, 0.0, 0.1, 0.0]))
    initial_guesses.append(np.array([0.5, -0.3, 0.0, 0.2, 0.0, -0.1, 0.0]))
    initial_guesses.append(np.array([0.5, 0.0, 0.3, 0.0, -0.2, 0.0, 0.1]))
    
    # Random restarts
    for _ in range(10):
        guess = [0.5]
        for _ in range(3):
            guess.append(rng.uniform(-0.4, 0.4))
            guess.append(rng.uniform(-0.4, 0.4))
        initial_guesses.append(np.array(guess))
    
    for guess in initial_guesses:
        if time.time() > deadline:
            break
        
        try:
            res = minimize(
                fourier_objective, guess,
                method='Nelder-Mead',
                bounds=fourier_bounds,
                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            h_cand = _fourier_to_h(res.x)
            score_cand = _overlap_max(h_cand)
            if score_cand < best_score:
                best_score = score_cand
                best_h = h_cand.copy()
        except Exception:
            pass
    
    # Gradient-based polish in Fourier space
    if time.time() < deadline - 30:
        # Find best Fourier params from best_h via least squares
        try:
            x = np.arange(N)
            A = np.ones((N, 1))
            for m in range(1, 4):
                A = np.column_stack([A, np.cos(2 * np.pi * m * x / N), np.sin(2 * np.pi * m * x / N)])
            coeffs, _, _, _ = np.linalg.lstsq(A, best_h, rcond=None)
            
            def fourier_obj_grad(params):
                h = _fourier_to_h(params)
                return _overlap_max(h)
            
            res2 = minimize(
                fourier_obj_grad, coeffs,
                method='Nelder-Mead',
                bounds=fourier_bounds,
                options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            h_cand2 = _fourier_to_h(res2.x)
            score_cand2 = _overlap_max(h_cand2)
            if score_cand2 < best_score:
                best_score = score_cand2
                best_h = h_cand2.copy()
        except Exception:
            pass
    
    # Final projection to ensure exact constraints
    best_h = np.clip(best_h, 0.0, 1.0)
    diff = N / 2.0 - np.sum(best_h)
    if abs(diff) > 1e-12:
        best_h = best_h + diff / N
        best_h = np.clip(best_h, 0.0, 1.0)
        diff2 = N / 2.0 - np.sum(best_h)
        if abs(diff2) > 1e-12:
            mask = (best_h > 1e-10) & (best_h < 1.0 - 1e-10)
            if np.any(mask):
                best_h[mask] += diff2 / np.sum(mask)
                best_h = np.clip(best_h, 0.0, 1.0)
    
    if _overlap_max(best_h) <= best_score + 1e-10:
        return best_h
    return h0

# EVOLVE-BLOCK-END
