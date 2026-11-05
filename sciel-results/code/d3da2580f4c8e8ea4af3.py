"""
Strategy: Direct Coordinate Ascent with Smooth Surrogate.

The previous approach used a low-frequency Fourier basis, which restricts the
solution to a specific subspace and may miss optimal high-frequency structures
needed to flatten the autocorrelation peak. This new strategy operates directly
on the full 400-dimensional vector h.

Key components:
1. **Surrogate Objective**: The term `max(conv(h, h))` is non-differentiable.
   We approximate it with a soft-max: `log(sum(exp(conv/tau))) * tau`.
   This provides a smooth landscape for optimization.
2. **Coordinate Ascent**: We iterate over indices `k`, adjusting `h[k]` to
   minimize the surrogate objective. Since the objective is scale-invariant,
   we can fix the scale (e.g., normalize sum to 1) or handle the scale
   explicitly. Normalizing sum(h)=1 simplifies the objective to minimizing
   `max(conv(h,h))` (up to constants).
3. **Line Search**: For each coordinate update, we perform a small line search
   to find the optimal step size.

This approach is a structural change from "model-based" (Fourier) to
"direct-search" (coordinate descent on full vector) with a different
optimization method (coordinate ascent vs L-BFGS-B on reduced params).
"""

import numpy as np
import time

N = 400
_DEADLINE = 145.0  # Safety margin for 150s limit

def _evaluate(h):
    """Compute the exact task value."""
    s = np.sum(h)
    if s == 0 or not np.isfinite(s):
        return np.inf
    conv = np.convolve(h, h)
    return abs(2.0 * N * np.max(conv) / (s * s))

def _surrogate_value(h, tau=0.1):
    """
    Compute a smooth surrogate for the objective.
    We fix the scale by normalizing h such that sum(h) = 1.
    Then V = 2 * max(conv(h, h)).
    We approximate max with soft-max.
    """
    s = np.sum(h)
    if abs(s) < 1e-12:
        return np.inf
    h_norm = h / s
    conv = np.convolve(h_norm, h_norm)
    # Soft max approximation
    # Note: conv can have negative values, but max is usually positive.
    # We shift by max to prevent overflow in exp.
    m = np.max(conv)
    soft_max = m + tau * np.log(np.sum(np.exp((conv - m) / tau)))
    # The actual value is 2 * soft_max
    return 2.0 * soft_max

def construct():
    t0 = time.time()
    deadline = t0 + _DEADLINE
    
    # Initialize with ones (seed)
    best_h = np.ones(N)
    best_val = _evaluate(best_h)
    
    # We will work in the normalized space sum(h)=1 to remove scale invariance.
    # Let h = h_work / sum(h_work).
    # Actually, it's easier to just maintain h and normalize in the surrogate.
    # But for coordinate ascent, it's better to work on a fixed manifold.
    # Let's force sum(h) = 1.
    h_work = np.ones(N) / N  # sum is 1
    
    current_val = _evaluate(h_work)
    best_val = min(best_val, current_val)
    best_h = h_work.copy()
    
    rng = np.random.default_rng(42)
    
    # Parameters for coordinate ascent
    tau = 0.05  # Temperature for soft-max
    max_iter = 50000
    step_sizes = np.array([0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001])
    
    # Precompute indices
    indices = np.arange(N)
    
    # Coordinate Ascent Loop
    # We iterate over coordinates in a shuffled order or cyclically.
    # Given the non-convexity, cyclic might be slow. Random order helps escape local minima.
    # But coordinate ascent usually works best with cyclic or greedy.
    # Let's try a mix: greedy selection of best coordinate is too expensive (N evals per step).
    # Cyclic is O(N) evals per full pass.
    # With N=400 and max_iter=50000, that's 125 full passes.
    # Each pass: 400 coordinate updates.
    # Each update: line search (7 steps).
    # Total evals: 125 * 400 * 7 = 350,000.
    # Each eval: convolve(400,400) is fast in numpy. ~1-2 ms?
    # 350,000 * 1.5ms = 525 seconds. Too slow.
    
    # Optimization:
    # 1. Use a smaller number of iterations.
    # 2. Use a cheaper surrogate? No, conv is the bottleneck.
    # 3. Use gradient descent instead?
    #    Gradient of soft-max w.r.t h is complex but doable.
    #    d/dh_k [ 2 * (m + tau * log(sum(exp((conv-m)/tau))) ) ]
    #    This requires backprop through conv.
    #    conv[i] = sum_j h[j] h[i-j].
    #    This is feasible but implementation-heavy.
    
    # Alternative: Use scipy.optimize.minimize on the full vector with the surrogate.
    # L-BFGS-B on 400 dims.
    # Objective: _surrogate_value
    # This is a structural change from the Fourier approach.
    
    def objective_full(h):
        return _surrogate_value(h, tau=tau)
    
    # Try multiple restarts with different initializations
    n_restarts = 10
    
    for r in range(n_restarts):
        if time.time() > deadline - 20:
            break
            
        # Initialization
        if r == 0:
            h_init = h_work.copy()
        else:
            # Random initialization with sum 1
            h_init = rng.normal(0, 1, N)
            h_init /= np.sum(h_init)
            # Mix with previous best to stay in a good basin
            h_init = 0.5 * h_init + 0.5 * best_h
            h_init /= np.sum(h_init)
            
        try:
            res = minimize(
                objective_full,
                h_init,
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if res.success or res.fun < objective_full(best_h):
                h_opt = res.x
                # Ensure finite and non-zero sum
                if np.all(np.isfinite(h_opt)) and abs(np.sum(h_opt)) > 1e-12:
                    v_opt = _evaluate(h_opt)
                    if v_opt < best_val:
                        best_val = v_opt
                        best_h = h_opt.copy()
                        
        except Exception:
            pass
    
    # Final refinement: Simple greedy local search on the best found h
    # This can fix small errors introduced by the optimizer.
    if time.time() < deadline - 10:
        h_curr = best_h.copy()
        v_curr = best_val
        step_size = 0.01
        for _ in range(2000):
            if time.time() > deadline - 5:
                break
            idx = rng.integers(0, N)
            h_new = h_curr.copy()
            h_new[idx] += rng.normal(0, step_size)
            # Normalize to keep sum=1 for consistency with surrogate, 
            # but _evaluate handles any scale.
            v_new = _evaluate(h_new)
            if v_new < v_curr:
                v_curr = v_new
                h_curr = h_new
                step_size *= 0.9
            else:
                step_size = min(step_size * 1.2, 0.05)
        
        if v_curr < best_val:
            best_val = v_curr
            best_h = h_curr

    if not np.all(np.isfinite(best_h)) or np.sum(best_h) == 0:
        return np.ones(N)
        
    return best_h

# Note: Need to import minimize
from scipy.optimize import minimize

# EVOLVE-BLOCK-END
