"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize
import random

N = 50

def _ratio(h):
    """Compute ||h||_2^2 / (||h||_1 * ||h*h||_inf)"""
    h = np.asarray(h, dtype=float)
    norm1 = np.sum(h)
    if norm1 <= 0:
        return 0.0
    norm2_sq = np.sum(h**2)
    # Autoconvolution (linear convolution)
    conv = np.convolve(h, h, mode='full')
    conv_inf = np.max(conv)
    if conv_inf <= 0:
        return 0.0
    return norm2_sq / (norm1 * conv_inf)

def _ratio_neg(h):
    """Negative ratio for minimization"""
    return -_ratio(h)

# EVOLVE-BLOCK-START
def construct():
    """
    Maximize ||h||_2^2 / (||h||_1 * ||h*h||_inf) for 50 nonnegative heights.
    
    Strategy: Use optimization to find a good shape. Start from several
    initial guesses including sparse clusters and run local optimization.
    """
    deadline = time.time() + 100  # 100 second safety margin
    
    best_h = None
    best_score = 0.0
    
    # Try various initial configurations
    initial_guesses = []
    
    # 1. Single spike
    for pos in [0, 1, 2, 5, 10, 25]:
        h = np.zeros(N)
        h[pos] = 1.0
        initial_guesses.append(h)
    
    # 2. Two-point clusters with various ratios
    for ratio in [1.5, 2.0, 2.5, 3.0, 4.0, 5.0]:
        for pos in [0, 1, 2, 5, 10]:
            h = np.zeros(N)
            h[pos] = ratio / (ratio + 1)
            h[pos+1] = 1.0 / (ratio + 1)
            initial_guesses.append(h)
    
    # 3. Three-point clusters
    for a, b, c in [(3,2,1), (4,2,1), (5,3,1), (2,1,0.5)]:
        total = a + b + c
        for pos in [0, 1, 2]:
            h = np.zeros(N)
            h[pos] = a / total
            h[pos+1] = b / total
            h[pos+2] = c / total
            initial_guesses.append(h)
    
    # 4. Random sparse configurations
    random.seed(42)
    for _ in range(20):
        h = np.zeros(N)
        num_nonzero = random.randint(2, 5)
        positions = random.sample(range(N-2), num_nonzero)
        for pos in positions:
            h[pos] = random.random()
        if np.sum(h) > 0:
            h = h / np.sum(h) * np.sum(h)  # Keep L1 norm
        initial_guesses.append(h)
    
    # 5. Gaussian-like shapes
    for center in [5, 10, 25]:
        for sigma in [1.0, 2.0, 3.0]:
            x = np.arange(N)
            h = np.exp(-0.5 * ((x - center) / sigma)**2)
            if np.sum(h) > 0:
                initial_guesses.append(h / np.sum(h) * np.sum(h))
    
    # Evaluate all initial guesses
    for h in initial_guesses:
        score = _ratio(h)
        if score > best_score:
            best_score = score
            best_h = h.copy()
    
    # Run local optimization from the best few starting points
    top_starters = []
    for h in initial_guesses:
        score = _ratio(h)
        top_starters.append((score, h))
    top_starters.sort(key=lambda x: -x[0])
    
    for i, (score, h) in enumerate(top_starters[:5]):
        if time.time() > deadline:
            break
        
        # Use L-BFGS-B with bounds
        bounds = [(0, None)] * N
        
        try:
            result = minimize(
                _ratio_neg,
                h.copy(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            if result.success or result.fun < -score:
                opt_h = np.maximum(result.x, 0)  # Ensure nonnegativity
                opt_score = _ratio(opt_h)
                if opt_score > best_score:
                    best_score = opt_score
                    best_h = opt_h.copy()
        except Exception:
            pass
    
    # If we have time, try more random restarts
    if time.time() < deadline - 20:
        for _ in range(10):
            if time.time() > deadline:
                break
            
            h = np.zeros(N)
            num_nonzero = random.randint(2, 6)
            positions = random.sample(range(N-2), num_nonzero)
            for pos in positions:
                h[pos] = random.random() ** 2  # Bias towards sparser
            if np.sum(h) > 0:
                h = h / np.sum(h)
            
            bounds = [(0, None)] * N
            try:
                result = minimize(
                    _ratio_neg,
                    h.copy(),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                if result.success:
                    opt_h = np.maximum(result.x, 0)
                    opt_score = _ratio(opt_h)
                    if opt_score > best_score:
                        best_score = opt_score
                        best_h = opt_h.copy()
            except Exception:
                pass
    
    # Return as a list of floats
    if best_h is None:
        # Fallback: 2-point cluster
        h = [0.0] * N
        h[0] = 2.0 / 3.0
        h[1] = 1.0 / 3.0
        return h
    
    return list(best_h)
# EVOLVE-BLOCK-END
