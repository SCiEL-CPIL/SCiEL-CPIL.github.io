"""
Third autocorrelation inequality - discrete sign search with number-theoretic
initializations and simulated annealing.

The task: choose a signed step function h with 400 equal intervals on [-1/4,1/4]
to minimize the upper bound on C3, measured as:
    score = max|autoconv(h)| / (sum|h|)^2

Since we are choosing a "signed step function", we work with h_i in {+1, -1}
(or more generally real-valued, but the discrete case is the natural starting point
given the "signed step" language). The evaluator measures the raw value (lower is better).
"""

import numpy as np
import time

N = 400

# Deadline guard
_START_TIME = time.time()
TIME_LIMIT = 135.0  # safety margin before 150s kill


def _autoconv(h):
    """Compute autoconvolution of h using FFT. Returns array of length N (non-negative lags)."""
    n = len(h)
    size = 2 * n
    H = np.fft.rfft(h, size)
    conv = np.fft.irfft(H * H, size)[:n]
    return conv


def _raw_value(h):
    """Compute the raw C3 upper bound value: max|conv| / (sum|h|)^2."""
    conv = _autoconv(h)
    max_conv = np.max(np.abs(conv))
    S = np.sum(np.abs(h))
    if S < 1e-10:
        return np.inf
    return max_conv / (S * S)


def _legendre_symbol(n):
    """Legendre symbol (n/p) for prime p. Returns 1, -1, or 0."""
    if n % p == 0:
        return 0
    return 1 if pow(n, (p - 1) // 2, p) == 1 else -1


def _make_legendre_init(p):
    """Create a sign pattern based on Legendre symbols modulo prime p."""
    h = np.zeros(N)
    for i in range(N):
        # Map index to a value in [0, p)
        val = i % p
        ls = _legendre_symbol(val)
        if ls == 0:
            h[i] = 1.0  # 0 maps to +1
        else:
            h[i] = float(ls)
    return h


def _make_mobius_init():
    """Create a sign pattern based on the Möbius function."""
    def mobius(n):
        if n == 1:
            return 1
        # Factor n
        result = 1
        d = 2
        temp = n
        while d * d <= temp:
            if temp % d == 0:
                temp //= d
                if temp % d == 0:
                    return 0
                result *= -1
            d += 1
        if temp > 1:
            result *= -1
        return result

    h = np.zeros(N)
    for i in range(N):
        h[i] = float(mobius(i + 1))
    return h


def _make_random_init(seed):
    """Random ±1 initialization."""
    rng = np.random.RandomState(seed)
    return rng.choice([-1.0, 1.0], size=N)


def _make_block_init(seed):
    """Block-based initialization with varying block sizes."""
    rng = np.random.RandomState(seed)
    h = np.zeros(N)
    pos = 0
    while pos < N:
        block_len = rng.randint(5, 50)
        sign = 1.0 if rng.rand() > 0.5 else -1.0
        end = min(pos + block_len, N)
        h[pos:end] = sign
        pos = end
    return h


def _make_fourier_init(seed):
    """Fourier-based initialization: sum of a few sinusoids, then sign."""
    rng = np.random.RandomState(seed)
    x = np.arange(N)
    h = np.zeros(N)
    n_terms = rng.randint(2, 6)
    for _ in range(n_terms):
        freq = rng.randint(1, 10)
        phase = rng.uniform(0, 2 * np.pi)
        amp = rng.uniform(0.5, 1.5)
        h += amp * np.sin(2 * np.pi * freq * x / N + phase)
    h = np.sign(h)
    h[h == 0] = 1.0
    return h


def _simulated_annealing(h, time_budget, T0=2.0, T_end=0.01, cooling=0.9995):
    """
    Simulated annealing with single and multi-flip moves.
    h: array of ±1 values
    Returns (best_h, best_val)
    """
    h = h.copy()
    best_h = h.copy()
    best_val = _raw_value(h)
    current_val = best_val
    
    start = time.time()
    T = T0
    step = 0
    
    while time.time() - start < time_budget:
        step += 1
        
        # Choose move type: 70% single flip, 30% multi-flip (2-5 flips)
        if np.random.random() < 0.7:
            # Single flip
            i = np.random.randint(N)
            h[i] = -h[i]
        else:
            # Multi-flip
            n_flips = np.random.randint(2, 6)
            indices = np.random.choice(N, size=n_flips, replace=False)
            h[indices] = -h[indices]
        
        new_val = _raw_value(h)
        delta = new_val - current_val
        
        if delta < 0 or np.random.random() < np.exp(-delta / T):
            current_val = new_val
            if new_val < best_val:
                best_val = new_val
                best_h = h.copy()
        else:
            # Revert
            if np.random.random() < 0.7:
                h[i] = -h[i]
            else:
                h[indices] = -h[indices]
        
        # Cool down
        T *= cooling
        if T < T_end:
            T = T_end
        
        # Periodic restart if stuck
        if step % 5000 == 0:
            T = max(T, T0 * 0.3)
    
    return best_h, best_val


def _local_search(h, time_budget=5.0):
    """
    Greedy local search: try all single flips, accept if improvement.
    Repeat until no improvement or time runs out.
    """
    h = h.copy()
    best_val = _raw_value(h)
    start = time.time()
    
    improved = True
    while improved and time.time() - start < time_budget:
        improved = False
        for i in range(N):
            h[i] = -h[i]
            val = _raw_value(h)
            if val < best_val - 1e-12:
                best_val = val
                improved = True
            else:
                h[i] = -h[i]  # revert
            if time.time() - start > time_budget:
                break
        if time.time() - start > time_budget:
            break
    
    return h, best_val


def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    # Generate diverse initializations
    initializations = []
    
    # Number-theoretic inits
    for p in [401, 409, 419, 431, 433, 439, 443, 449]:
        initializations.append(_make_legendre_init(p))
    
    initializations.append(_make_mobius_init())
    
    # Random and structured inits
    for seed in range(10):
        initializations.append(_make_random_init(seed * 7919 + 13))
        initializations.append(_make_block_init(seed * 104729 + 7))
        initializations.append(_make_fourier_init(seed * 15485863 + 3))
    
    # Evaluate all initializations
    init_vals = []
    for h in initializations:
        val = _raw_value(h)
        init_vals.append((h, val))
    
    # Sort by value
    init_vals.sort(key=lambda x: x[1])
    
    best_h = init_vals[0][0].copy()
    best_val = init_vals[0][1]
    
    # Run SA on the best few initializations
    elapsed = time.time() - _START_TIME
    remaining = TIME_LIMIT - elapsed - 10  # leave 10s margin
    
    if remaining > 5:
        # Run SA on top 3 initializations
        n_sa = min(3, len(init_vals))
        per_sa = remaining / n_sa
        
        for idx in range(n_sa):
            if time.time() - _START_TIME > TIME_LIMIT - 5:
                break
            
            h_init, val_init = init_vals[idx]
            if val_init >= best_val and idx > 0:
                continue  # Skip if not better than current best
            
            budget = min(per_sa, 30.0)
            h_sa, val_sa = _simulated_annealing(h_init, time_budget=budget, 
                                                 T0=2.0, T_end=0.005, cooling=0.9998)
            
            if val_sa < best_val:
                best_val = val_sa
                best_h = h_sa.copy()
        
        # Local search polish on best
        elapsed = time.time() - _START_TIME
        remaining = TIME_LIMIT - elapsed - 5
        
        if remaining > 3:
            best_h, best_val = _local_search(best_h, time_budget=min(remaining, 10.0))
    
    return best_h
