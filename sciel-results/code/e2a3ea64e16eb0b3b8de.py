"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

N = 400
# Deadline guard: evaluator kills at 150s. We target finishing within ~120s.
TIME_LIMIT = 110.0  # safety margin

def _autoconv(x):
    """Compute full autoconvolution using FFT."""
    n = len(x)
    size = 2 * n - 1
    # Use next power of 2 for FFT efficiency
    fsize = 1 << (size - 1).bit_length()
    X = np.fft.rfft(x, n=fsize)
    ac = np.fft.irfft(X * X, n=fsize)[:size]
    return ac

def _score_signal(x):
    """
    Score: max |autoconv| / (sum(x))^2
    Lower is better.
    """
    integral = np.sum(x)
    if abs(integral) < 1e-12:
        return 1e10
    ac = _autoconv(x)
    max_abs = np.max(np.abs(ac))
    return max_abs / (integral * integral)

def _construct_from_blocks(params, N):
    """
    Construct signal from a set of rectangular blocks.
    params: [a1, p1, w1, a2, p2, w2, ..., aK, pK, wK]
    ai: amplitude, pi: start position (float), wi: width (float)
    """
    K = len(params) // 3
    x = np.zeros(N)
    for k in range(K):
        a = params[3*k]
        p = params[3*k + 1]
        w = params[3*k + 2]
        # Clip to valid range
        p = np.clip(p, 0, N - 1)
        w = np.clip(w, 0, N)
        start = int(np.floor(p))
        end = int(np.ceil(p + w))
        start = max(0, min(N, start))
        end = max(0, min(N, end))
        if end > start:
            x[start:end] += a
    return x

def construct():
    start_time = time.time()
    
    # Strategy:
    # 1. Try a few fixed small-K block configurations using Differential Evolution.
    # 2. Polish the best result using local minimization on the discrete signal if time permits.
    
    best_x = np.ones(N)
    best_score = _score_signal(best_x)
    
    # Number of blocks to try
    for K in [2, 3, 4]:
        if time.time() - start_time > TIME_LIMIT - 20:
            break
            
        # Parameters: for each block, (amplitude, position, width)
        # Bounds:
        # amplitude: [-1, 1] (sign matters, magnitude can be scaled)
        # position: [0, N]
        # width: [1, N]
        
        bounds = []
        for _ in range(K):
            bounds.append((-1.0, 1.0))  # amplitude
            bounds.append((0.0, float(N)))  # position
            bounds.append((1.0, float(N)))  # width
            
        def objective(params):
            x = _construct_from_blocks(params, N)
            return _score_signal(x)
        
        # Use Differential Evolution
        try:
            res = differential_evolution(
                objective,
                bounds=bounds,
                seed=42,
                maxiter=50,
                popsize=10,
                tol=1e-4,
                polish=False,
                callback=None
            )
            
            if time.time() - start_time > TIME_LIMIT - 10:
                # Still evaluate the best found so far
                x_candidate = _construct_from_blocks(res.x, N)
                s = _score_signal(x_candidate)
                if s < best_score:
                    best_score = s
                    best_x = x_candidate
                continue
                
            x_candidate = _construct_from_blocks(res.x, N)
            s = _score_signal(x_candidate)
            
            if s < best_score:
                best_score = s
                best_x = x_candidate
                
                # Polish pass: minimize over the discrete vector x
                # We can try a few steps of gradient-free local search or use scipy minimize on the continuous relaxation?
                # The signal is discrete (indices 0..N-1). The block construction gave us a specific pattern.
                # Let's try to refine the values of x directly using Nelder-Mead or Powell?
                # N is 400, so a full local optimization on 400 variables is expensive.
                # Instead, we can try to adjust the block parameters further with a tighter DE or local search.
                
                # Let's do a quick local polish on the block parameters using scipy.optimize.minimize
                if time.time() - start_time < TIME_LIMIT - 5:
                    res2 = minimize(
                        objective,
                        res.x,
                        method='Nelder-Mead',
                        options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-8}
                    )
                    if res2.success or res2.fun < s:
                        s2 = res2.fun
                        if s2 < best_score:
                            best_score = s2
                            best_x = _construct_from_blocks(res2.x, N)
                            
        except Exception:
            pass

    # One more check: maybe a simple alternating sign pattern or other heuristic is better?
    # For K=2, two blocks of opposite sign might cancel out in the integral?
    # Wait, the denominator is (sum(x))^2. If sum(x) is small, the score explodes unless max_abs is also tiny.
    # We want sum(x) large and max_abs small.
    # So we want constructive interference in the integral but destructive in the autoconvolution sidelobes.
    
    # The block approach with DE should find good configurations.
    
    # Ensure we return a numpy array
    return best_x

# EVOLVE-BLOCK-START
# The code above is the implementation.
# We need to keep the interface exactly as requested.
# The current program has:
# N = 400
# def construct(): return np.ones(N)
# My code replaces the body of construct and adds helper functions.
# I should ensure N is defined.

# EVOLVE-BLOCK-END
