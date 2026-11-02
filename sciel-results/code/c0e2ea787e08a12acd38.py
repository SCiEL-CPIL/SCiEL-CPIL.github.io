"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 50
DEADLINE = 150  # seconds
START_TIME = time.time()

def _time_remaining():
    return DEADLINE - (time.time() - START_TIME)

def _autoconvolution_fast(h):
    """Compute autoconvolution using FFT for speed."""
    n = len(h)
    size = 1
    while size < 2 * n - 1:
        size *= 2
    h_padded = np.zeros(size)
    h_padded[:n] = h
    fft_h = np.fft.fft(h_padded)
    conv = np.fft.ifft(fft_h * fft_h).real
    return conv[:2 * n - 1]

def _score(h):
    """Compute the ratio: (sum h^2) / (sum h * max autoconvolution)."""
    h = np.asarray(h, dtype=float)
    if np.any(h < -1e-15):
        h = np.maximum(h, 0)
    l2_sq = np.sum(h ** 2)
    l1 = np.sum(h)
    if l1 < 1e-15 or l2_sq < 1e-15:
        return 0.0
    conv = _autoconvolution_fast(h)
    sup_conv = np.max(conv)
    if sup_conv < 1e-15:
        return 0.0
    return l2_sq / (l1 * sup_conv)

def _objective_normalized(x):
    """
    Objective function for optimization.
    We fix L1 norm to 1 to remove scale invariance.
    Maximize (sum x^2) / max(autoconv(x))
    Equivalent to minimizing -log( (sum x^2) / max(autoconv(x)) )
    """
    if _time_remaining() < 5:
        return 1e10
    
    # Ensure non-negativity and normalize L1 to 1
    x = np.maximum(x, 0)
    l1 = np.sum(x)
    if l1 < 1e-15:
        return 1e10
    
    x_norm = x / l1
    
    l2_sq = np.sum(x_norm ** 2)
    if l2_sq < 1e-15:
        return 1e10
    
    conv = _autoconvolution_fast(x_norm)
    sup_conv = np.max(conv)
    if sup_conv < 1e-15:
        return 1e10
    
    ratio = l2_sq / sup_conv
    if ratio <= 0:
        return 1e10
    
    return -np.log(ratio)

def construct():
    """Construct 50 nonnegative heights to maximize the ratio."""
    global START_TIME
    START_TIME = time.time()
    
    best_h = np.ones(N) / N  # Uniform baseline
    best_score = _score(best_h)
    
    # Multi-start optimization
    # We optimize x such that sum(x)=1 is enforced by normalization in objective
    # But L-BFGS-B needs bounds. We will optimize raw x >= 0.
    
    bounds = [(0, None)] * N
    max_restarts = 50  # Adjust based on time
    
    # Initial guesses:
    # 1. Uniform
    # 2. Random sparse
    # 3. Random dense
    
    rng = np.random.default_rng(42)
    
    for i in range(max_restarts):
        if _time_remaining() < 10:
            break
        
        # Generate initial guess
        if i == 0:
            x0 = np.ones(N) / N
        elif i % 3 == 1:
            # Sparse: pick random 10-20 indices
            k = rng.integers(10, 20)
            indices = rng.choice(N, size=k, replace=False)
            x0 = np.zeros(N)
            x0[indices] = 1.0 / k
        else:
            # Dense random
            x0 = rng.random(N)
            x0 /= np.sum(x0)
        
        try:
            result = minimize(
                _objective_normalized, 
                x0, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-12}
            )
            
            if result.fun < 1e9:
                h_candidate = np.maximum(result.x, 0)
                # Normalize to L1=1 for final score calculation consistency
                l1 = np.sum(h_candidate)
                if l1 > 1e-15:
                    h_candidate = h_candidate / l1
                    score = _score(h_candidate)
                    if score > best_score:
                        best_score = score
                        best_h = h_candidate.copy()
        except Exception:
            continue
    
    return list(best_h)

# EVOLVE-BLOCK-START
def construct():
    """Construct 50 nonnegative heights to maximize the ratio."""
    global START_TIME
    START_TIME = time.time()
    
    best_h = np.ones(N) / N  # Uniform baseline
    best_score = _score(best_h)
    
    # Multi-start optimization
    # We optimize x such that sum(x)=1 is enforced by normalization in objective
    # But L-BFGS-B needs bounds. We will optimize raw x >= 0.
    
    bounds = [(0, None)] * N
    max_restarts = 50  # Adjust based on time
    
    # Initial guesses:
    # 1. Uniform
    # 2. Random sparse
    # 3. Random dense
    
    rng = np.random.default_rng(42)
    
    for i in range(max_restarts):
        if _time_remaining() < 10:
            break
        
        # Generate initial guess
        if i == 0:
            x0 = np.ones(N) / N
        elif i % 3 == 1:
            # Sparse: pick random 10-20 indices
            k = rng.integers(10, 20)
            indices = rng.choice(N, size=k, replace=False)
            x0 = np.zeros(N)
            x0[indices] = 1.0 / k
        else:
            # Dense random
            x0 = rng.random(N)
            x0 /= np.sum(x0)
        
        try:
            result = minimize(
                _objective_normalized, 
                x0, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-12}
            )
            
            if result.fun < 1e9:
                h_candidate = np.maximum(result.x, 0)
                # Normalize to L1=1 for final score calculation consistency
                l1 = np.sum(h_candidate)
                if l1 > 1e-15:
                    h_candidate = h_candidate / l1
                    score = _score(h_candidate)
                    if score > best_score:
                        best_score = score
                        best_h = h_candidate.copy()
        except Exception:
            continue
    
    return list(best_h)
# EVOLVE-BLOCK-END
