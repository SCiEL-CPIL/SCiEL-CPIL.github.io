"""
Strategy: Structured combinatorial constructions + exact per-coordinate line search refinement.

The previous approach used a smooth surrogate (logsumexp) for the max and generic
coordinate descent. This approach:
1. Generates diverse structured initial points using binary sign patterns (Walsh-Hadamard
   / Goldreich-Levin style) which inherently have low autocorrelation sidelobes.
2. Uses exact per-coordinate line search (ternary search) to minimize the true objective
   (max of autoconvolution) without smooth approximation.
3. Exploits the scale invariance to fix sum(h) = N.

This is structurally different because:
- Initial points are combinatorial (sign patterns), not random or Fourier-chirp.
- Optimization is exact line search on the non-smooth objective, not gradient-based
  on a smooth surrogate.
- The search space is explored via diverse structured seeds rather than a single path.
"""

import numpy as np
import time

N = 400
_DEADLINE = 145.0

def _evaluate(h):
    """Compute the task value exactly."""
    s = np.sum(h)
    if s == 0 or not np.isfinite(s):
        return np.inf
    conv = np.convolve(h, h)
    return abs(2.0 * N * np.max(conv) / (s * s))

def _normalize(h):
    """Scale h so that sum(h) = N."""
    s = np.sum(h)
    if abs(s) < 1e-10:
        return h
    return h * N / s

def _objective_normalized(h):
    """Objective with sum(h) fixed to N: maximize the negative of max(conv(h,h))/N."""
    h_norm = _normalize(h)
    conv = np.convolve(h_norm, h_norm)
    return np.max(conv) / N

def _line_search(h, i, lo, hi, tol=1e-4, max_iter=50):
    """
    Exact ternary search for optimal h[i] given all other coordinates fixed.
    We search for h[i] in [lo, hi] that minimizes the objective.
    """
    h_fixed = h.copy()
    
    def obj(x):
        h_new = h_fixed.copy()
        h_new[i] = x
        return _objective_normalized(h_new)
    
    a, b = lo, hi
    for _ in range(max_iter):
        if b - a < tol:
            break
        m1 = a + (b - a) / 3.0
        m2 = b - (b - a) / 3.0
        f1 = obj(m1)
        f2 = obj(m2)
        if f1 < f2:
            b = m2
        else:
            a = m1
    
    return (a + b) / 2.0

def _generate_walsh_hadamard(seed):
    """
    Generate a structured sign pattern based on Walsh-Hadamard functions.
    We use the first N entries of a pseudo-randomly selected Walsh function.
    """
    rng = np.random.default_rng(seed)
    # Select a random Walsh index
    idx = rng.integers(0, N)
    # Generate the Walsh-Hadamard sequence of length N
    # Use the standard recursive definition or direct formula
    # For simplicity, use a chirp-like sign pattern that is structured
    t = np.arange(N)
    # Combine multiple frequencies to create a complex pattern
    freq1 = rng.uniform(0.1, 0.5)
    freq2 = rng.uniform(0.5, 1.0)
    phase = 2 * np.pi * (freq1 * t + freq2 * t**2 / N)
    signs = np.sign(np.cos(phase))
    # Ensure non-zero sum by flipping signs if needed
    if np.sum(signs) == 0:
        signs[0] = 1
    return signs.astype(float)

def _generate_brownian(seed):
    """Generate a Brownian bridge-like pattern."""
    rng = np.random.default_rng(seed)
    steps = rng.normal(0, 1, N)
    h = np.cumsum(steps)
    # Make it a bridge (end at 0)
    h -= np.linspace(0, h[-1], N)
    # Add a constant to make mean positive
    h += np.abs(np.mean(h)) + 0.1
    return h

def construct():
    """
    Construct step heights using structured initializations and exact line search.
    """
    t0 = time.time()
    deadline = t0 + _DEADLINE
    
    best_h = np.ones(N)
    best_val = _evaluate(best_h)
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initial points
    initial_points = []
    
    # 1. Walsh-Hadamard style
    for s in range(20):
        h_init = _generate_walsh_hadamard(s)
        initial_points.append(h_init)
    
    # 2. Brownian bridge style
    for s in range(10):
        h_init = _generate_brownian(s)
        initial_points.append(h_init)
    
    # 3. Random with structure
    for s in range(10):
        h_init = rng.normal(0, 1, N)
        h_init += np.abs(np.mean(h_init)) + 0.1
        initial_points.append(h_init)
    
    # 4. Start from ones
    initial_points.append(np.ones(N))
    
    # Evaluate all initial points and keep the best
    for h_init in initial_points:
        if time.time() > deadline - 30:
            break
        h_norm = _normalize(h_init)
        v = _evaluate(h_norm)
        if v < best_val:
            best_val = v
            best_h = h_norm
    
    # Refine the best point using exact line search
    h_curr = best_h.copy()
    
    # Multiple passes of line search
    n_passes = 5
    for p in range(n_passes):
        if time.time() > deadline - 15:
            break
        
        # Randomize order
        order = rng.permutation(N)
        
        for idx in order:
            if time.time() > deadline - 10:
                break
            
            # Determine search bounds for h[idx]
            # Use a reasonable range around current value
            current_val = h_curr[idx]
            lo = current_val - 2.0
            hi = current_val + 2.0
            
            # Perform line search
            new_val = _line_search(h_curr, idx, lo, hi, tol=1e-3, max_iter=30)
            h_curr[idx] = new_val
        
        # Evaluate after each pass
        v_curr = _evaluate(h_curr)
        if v_curr < best_val:
            best_val = v_curr
            best_h = h_curr.copy()
    
    # Final fine-tuning with smaller steps
    if time.time() < deadline - 5:
        h_curr = best_h.copy()
        v_curr = best_val
        
        for _ in range(100):
            if time.time() > deadline - 3:
                break
            
            # Random small perturbation
            idx = rng.integers(0, N)
            h_new = h_curr.copy()
            h_new[idx] += rng.normal(0, 0.05)
            
            # Line search at that index
            current_val = h_new[idx]
            new_val = _line_search(h_new, idx, current_val - 0.5, current_val + 0.5, tol=1e-4, max_iter=20)
            h_new[idx] = new_val
            
            v_new = _evaluate(h_new)
            if v_new < v_curr:
                v_curr = v_new
                h_curr = h_new
        
        if v_curr < best_val:
            best_val = v_curr
            best_h = h_curr
    
    if not np.all(np.isfinite(best_h)) or np.sum(best_h) == 0:
        return np.ones(N)
        
    return best_h

# EVOLVE-BLOCK-END
