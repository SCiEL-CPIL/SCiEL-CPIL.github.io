"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize, Bounds

N = 400
DEADLINE = 150.0

# EVOLVE-BLOCK-START
def _value(h):
    s = float(np.sum(h))
    if abs(s) < 1e-12:
        return 1e18
    conv = np.convolve(h, h)
    return abs(2.0 * N * float(np.max(conv)) / (s * s))

def _objective(x):
    h = np.tanh(x)
    return _value(h)

def construct():
    t0 = time.time()
    best_h = np.ones(N)
    best_val = _value(best_h)  # should be 2.0
    
    # Try a few structured patterns that might break the constant-vector optimality
    # 1. Sine-like patterns with non-zero mean
    for k in [1, 2, 3, 5, 7, 11, 13, 17, 23, 31, 37, 41, 47, 53, 61, 71, 79, 83, 97, 101]:
        if time.time() - t0 > DEADLINE - 5:
            break
        x = np.arange(N)
        for amp in [0.5, 1.0, 2.0, 3.0]:
            h = 1.0 + amp * np.sin(2 * np.pi * k * x / N)
            v = _value(h)
            if v < best_val:
                best_val = v
                best_h = h.copy()
    
    # 2. Cosine patterns
    for k in [1, 2, 3, 5, 7, 11, 13, 17, 23, 31, 37, 41, 47, 53, 61, 71, 79, 83, 97, 101]:
        if time.time() - t0 > DEADLINE - 5:
            break
        x = np.arange(N)
        for amp in [0.5, 1.0, 2.0, 3.0]:
            h = 1.0 + amp * np.cos(2 * np.pi * k * x / N)
            v = _value(h)
            if v < best_val:
                best_val = v
                best_h = h.copy()
    
    # 3. Exponential decay patterns
    for base in [0.9, 0.95, 0.99, 1.01, 1.05, 1.1]:
        if time.time() - t0 > DEADLINE - 5:
            break
        x = np.arange(N)
        h = base ** x
        v = _value(h)
        if v < best_val:
            best_val = v
            best_h = h.copy()
    
    # 4. Polynomial patterns
    for deg in [1, 2, 3, 4, 5]:
        if time.time() - t0 > DEADLINE - 5:
            break
        x = np.linspace(-1, 1, N)
        h = np.ones(N)
        for d in range(1, deg + 1):
            h = h + 0.1 * x**d
        v = _value(h)
        if v < best_val:
            best_val = v
            best_h = h.copy()
    
    # 5. Random perturbations with local optimization
    rng = np.random.default_rng(42)
    for trial in range(20):
        if time.time() - t0 > DEADLINE - 10:
            break
        x0 = rng.normal(0, 0.5, N)
        res = minimize(_objective, x0, method='L-BFGS-B', options={'maxiter': 100, 'maxfun': 200})
        if time.time() - t0 > DEADLINE - 5:
            break
        h = np.tanh(res.x)
        v = _value(h)
        if v < best_val:
            best_val = v
            best_h = h.copy()
    
    # 6. Try sign patterns with controlled sum
    for n_neg in [1, 2, 3, 5, 10, 20, 50, 100]:
        if time.time() - t0 > DEADLINE - 5:
            break
        for _ in range(5):
            h = np.ones(N)
            neg_idx = rng.choice(N, n_neg, replace=False)
            h[neg_idx] = -1.0
            s = np.sum(h)
            if abs(s) < 1e-12:
                continue
            v = _value(h)
            if v < best_val:
                best_val = v
                best_h = h.copy()
    
    # 7. Block patterns
    for block_size in [10, 20, 40, 50, 80, 100]:
        if time.time() - t0 > DEADLINE - 5:
            break
        for phase in range(block_size):
            h = np.ones(N)
            for i in range(N):
                if (i + phase) % (2 * block_size) >= block_size:
                    h[i] = -0.5
            s = np.sum(h)
            if abs(s) < 1e-12:
                continue
            v = _value(h)
            if v < best_val:
                best_val = v
                best_h = h.copy()
    
    return best_h
# EVOLVE-BLOCK-END
