"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 100

def _score(h):
    """Compute max overlap of h with shifted (1-h), treating h as zero outside [0,2]."""
    n = len(h)
    shifted = np.zeros(n)
    shifted[:-1] = h[1:]
    overlap = h * (1.0 - shifted)
    return float(np.max(overlap))

def _normalize(h):
    """Enforce exact sum N/2 and bounds [0,1]."""
    h = h.copy()
    for _ in range(20):
        s = np.sum(h)
        diff = N/2.0 - s
        if abs(diff) < 1e-15:
            break
        h = h + diff / N
        h = np.clip(h, 0.0, 1.0)
    # Final exact correction
    s = np.sum(h)
    diff = N/2.0 - s
    h = h + diff / N
    h = np.clip(h, 0.0, 1.0)
    return h

def _smooth_max(x, alpha=100.0):
    """Smooth approximation of max using log-sum-exp."""
    x_max = np.max(x)
    return x_max + (1.0/alpha) * np.log(np.sum(np.exp(alpha * (x - x_max))))

def _eval_fourier(params, K):
    """
    Reconstruct h from Fourier coefficients.
    params: [a1, b1, ..., aK, bK]
    h_i = 0.5 + sum_{k=1}^K (a_k cos(2 pi k i / N) + b_k sin(2 pi k i / N))
    """
    i = np.arange(N)
    h = np.full(N, 0.5)
    for k in range(1, K + 1):
        a_k = params[2*(k-1)]
        b_k = params[2*(k-1)+1]
        h += a_k * np.cos(2 * np.pi * k * i / N) + b_k * np.sin(2 * np.pi * k * i / N)
    return h

def _eval_spline(params, num_knots):
    """
    Reconstruct h from piecewise linear spline values at knots.
    params: values at knots (including endpoints).
    """
    knot_indices = np.linspace(0, N-1, num_knots)
    # Interpolate to get h
    h = np.interp(np.arange(N), knot_indices, params)
    return h

def _optimize_structure(h_init, K=None, num_knots=None, method='fourier', maxiter=500):
    """
    Optimize parameters of a structured model to minimize max overlap.
    """
    start_time = time.time()
    
    if method == 'fourier':
        # Extract initial coefficients via FFT (approximate)
        fft_coeffs = np.fft.rfft(h_init)
        # We use real sine/cosine basis
        # h_i = 0.5 + sum ...
        # We just initialize params with small random values or from FFT
        # For simplicity, let's start with 0s or small noise, or try to fit
        # Actually, let's just start with 0s and let the optimizer find them.
        # Or better, project h_init onto the subspace.
        # Projection: a_k = (2/N) sum h_i cos(...), b_k = (2/N) sum h_i sin(...)
        i = np.arange(N)
        params0 = np.zeros(2*K)
        for k in range(1, K + 1):
            params0[2*(k-1)] = (2.0/N) * np.sum(h_init * np.cos(2 * np.pi * k * i / N))
            params0[2*(k-1)+1] = (2.0/N) * np.sum(h_init * np.sin(2 * np.pi * k * i / N))
        
        def objective(params):
            h = _eval_fourier(params, K)
            # Penalty for bounds
            penalty = 100.0 * (np.sum(np.maximum(0, h - 1.0)**2) + np.sum(np.maximum(0, -h)**2))
            # Smooth max
            shifted = np.zeros(N)
            shifted[:-1] = h[1:]
            overlap = h * (1.0 - shifted)
            smax = _smooth_max(overlap, alpha=50.0)
            return smax + penalty
            
        def constraints(params):
            h = _eval_fourier(params, K)
            # h >= 0
            return h
        # SLSQP requires constraints as functions returning array for ineq
        # But we have N constraints. This might be slow.
        # Let's use penalty only for bounds in objective, and rely on smooth max.
        
        result = minimize(
            objective,
            params0,
            method='L-BFGS-B', # L-BFGS-B is good for this size and no explicit constraints needed if penalty is strong
            options={'maxiter': maxiter, 'ftol': 1e-12}
        )
        
        if result.success or result.fun < 1.0:
            h_opt = _eval_fourier(result.x, K)
            h_opt = _normalize(h_opt)
            return h_opt, _score(h_opt)
        return None, float('inf')

    elif method == 'spline':
        # Initialize with h_init sampled at knots
        knot_indices = np.linspace(0, N-1, num_knots)
        params0 = np.interp(knot_indices, np.arange(N), h_init)
        
        def objective(params):
            h = _eval_spline(params, num_knots)
            penalty = 100.0 * (np.sum(np.maximum(0, h - 1.0)**2) + np.sum(np.maximum(0, -h)**2))
            shifted = np.zeros(N)
            shifted[:-1] = h[1:]
            overlap = h * (1.0 - shifted)
            smax = _smooth_max(overlap, alpha=50.0)
            return smax + penalty

        result = minimize(
            objective,
            params0,
            method='L-BFGS-B',
            options={'maxiter': maxiter, 'ftol': 1e-12}
        )
        
        if result.success or result.fun < 1.0:
            h_opt = _eval_spline(result.x, num_knots)
            h_opt = _normalize(h_opt)
            return h_opt, _score(h_opt)
        return None, float('inf')
        
    return None, float('inf')

def _local_polish(h_init, maxiter=500):
    """
    Full local polish on the N-dimensional vector using SLSQP and smooth max.
    """
    def _smooth_objective(params, alpha):
        h = params
        shifted = np.zeros(N)
        shifted[:-1] = h[1:]
        overlap = h * (1.0 - shifted)
        smax = _smooth_max(overlap, alpha=alpha)
        # Penalty for sum constraint if not handled by SLSQP
        sum_penalty = 100.0 * (np.sum(h) - N/2.0)**2
        return smax + sum_penalty

    def _constraint_sum(params):
        return np.sum(params) - N/2.0

    constraints = [{'type': 'eq', 'fun': _constraint_sum}]
    bounds = [(0.0, 1.0)] * N

    result = minimize(
        lambda x: _smooth_objective(x, 200.0),
        h_init,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': maxiter, 'ftol': 1e-14, 'disp': False}
    )
    
    if result.fun < 1.0:
        h_opt = _normalize(result.x)
        return h_opt, _score(h_opt)
    return None, float('inf')

def construct():
    start_time = time.time()
    deadline = start_time + 135.0

    # 1. Generate diverse initial guesses
    A = (3.0 - math.sqrt(5.0)) / 2.0
    B = 1.0 - A
    h0 = np.zeros(N)
    half = N // 2
    h0[:half] = A
    h0[half:] = B
    
    candidates = [h0]
    
    # Add random smooth candidates
    np.random.seed(42)
    for _ in range(3):
        h_rand = 0.5 + 0.2 * np.sin(np.random.uniform(0, 2*np.pi) + np.arange(N)*0.1)
        h_rand = np.clip(h_rand, 0, 1)
        h_rand = _normalize(h_rand)
        candidates.append(h_rand)

    best_h = h0.copy()
    best_score = _score(h0)

    # 2. Try Structured Optimizations (Fourier and Spline)
    # This is the structural change: low-dimensional parameterization
    
    # Fourier with K=5
    try:
        if time.time() < deadline - 60.0:
            h_struct, s_struct = _optimize_structure(best_h, K=5, method='fourier')
            if s_struct < best_score:
                best_h = h_struct
                best_score = s_struct
    except Exception:
        pass

    # Fourier with K=10
    try:
        if time.time() < deadline - 45.0:
            h_struct, s_struct = _optimize_structure(best_h, K=10, method='fourier')
            if s_struct < best_score:
                best_h = h_struct
                best_score = s_struct
    except Exception:
        pass

    # Spline with 10 knots
    try:
        if time.time() < deadline - 30.0:
            h_struct, s_struct = _optimize_structure(best_h, num_knots=10, method='spline')
            if s_struct < best_score:
                best_h = h_struct
                best_score = s_struct
    except Exception:
        pass

    # Spline with 20 knots
    try:
        if time.time() < deadline - 15.0:
            h_struct, s_struct = _optimize_structure(best_h, num_knots=20, method='spline')
            if s_struct < best_score:
                best_h = h_struct
                best_score = s_struct
    except Exception:
        pass

    # 3. Local Polish on the best candidate found
    try:
        if time.time() < deadline - 5.0:
            h_polished, s_polished = _local_polish(best_h, maxiter=1000)
            if s_polished < best_score:
                best_h = h_polished
                best_score = s_polished
    except Exception:
        pass

    return best_h
