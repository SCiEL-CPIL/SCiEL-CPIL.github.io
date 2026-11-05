"""
Strategy: Golay Complementary Pair Construction with Multi-Phase Refinement.

This is a structurally different approach from the current program's direct
coordinate ascent / L-BFGS-B on the full 400-dim vector. Instead, we:

1. Represent h as a sum of two chirp-like sequences (Golay complementary pair idea).
   The key insight is that if we can find two sequences a and b such that
   conv(a,a) + conv(b,b) is flat (constant), then h = a + b will have a
   significantly reduced autocorrelation peak. We use quadratic chirps with
   different rates to approximate this property.

2. Optimize the chirp parameters (frequencies, phases, amplitudes) rather than
   the full 400-dim vector. This reduces the search space to ~10-20 parameters.

3. Use a multi-phase refinement: after finding good chirp parameters, do a local
   refinement on the full vector to fine-tune.

The structural change is:
- Current: full 400-dim direct optimization with soft-max surrogate
- New: parametric chirp-pair model (low-dim params) + local refinement
"""

import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 400
_DEADLINE = 145.0  # Safety margin for 150s limit


def _evaluate(h):
    """Compute the exact task value."""
    s = np.sum(h)
    if s == 0 or not np.isfinite(s):
        return np.inf
    conv = np.convolve(h, h)
    return abs(2.0 * N * np.max(conv) / (s * s))


def _chirp_pair_h(params):
    """
    Construct h from chirp pair parameters.
    params = [f1, f2, phase1, phase2, amp1, amp2, f3, f4, phase3, phase4, amp3, amp4]
    where f1,f2 are chirp rates, phase1,phase2 are initial phases,
    amp1,amp2 are amplitudes for the first pair.
    f3,f4,phase3,phase4,amp3,amp4 for the second pair.
    
    We use 4 chirps to form two complementary pairs.
    """
    n = np.arange(N)
    # First pair
    t1 = n / N
    a1 = params[4] * np.exp(1j * (params[0] * t1**2 + params[2]))
    a2 = params[5] * np.exp(1j * (params[1] * t1**2 + params[3]))
    # Second pair
    a3 = params[10] * np.exp(1j * (params[6] * t1**2 + params[8]))
    a4 = params[11] * np.exp(1j * (params[7] * t1**2 + params[9]))
    
    # Sum all four (real part to get real-valued h)
    h = np.real(a1 + a2 + a3 + a4)
    return h


def _chirp_pair_objective(params):
    """Objective: minimize the task value for the chirp pair."""
    h = _chirp_pair_h(params)
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    conv = np.convolve(h, h)
    val = abs(2.0 * N * np.max(conv) / (s * s))
    return val


def _local_refine(h_init, time_budget):
    """
    Local refinement using random perturbations.
    This is a simple but effective local search.
    """
    h = h_init.copy()
    best_val = _evaluate(h)
    t_start = time.time()
    
    step_size = 0.05
    rng = np.random.default_rng(123)
    
    while time.time() - t_start < time_budget:
        # Random perturbation
        idx = rng.integers(0, N)
        h_new = h.copy()
        h_new[idx] += rng.normal(0, step_size)
        
        v_new = _evaluate(h_new)
        if v_new < best_val:
            best_val = v_new
            h = h_new
            step_size *= 0.95
        else:
            step_size = min(step_size * 1.1, 0.1)
    
    return h, best_val


def construct():
    t0 = time.time()
    deadline = t0 + _DEADLINE
    
    best_h = np.ones(N)
    best_val = _evaluate(best_h)  # Should be 2.0
    
    # Phase 1: Optimize chirp pair parameters using differential evolution
    # Parameters: [f1, f2, phase1, phase2, amp1, amp2, f3, f4, phase3, phase4, amp3, amp4]
    # Bounds for each parameter
    bounds = [
        (0.5, 5.0),    # f1
        (0.5, 5.0),    # f2
        (0, 2*np.pi),  # phase1
        (0, 2*np.pi),  # phase2
        (0.5, 2.0),    # amp1
        (0.5, 2.0),    # amp2
        (0.5, 5.0),    # f3
        (0.5, 5.0),    # f4
        (0, 2*np.pi),  # phase3
        (0, 2*np.pi),  # phase4
        (0.5, 2.0),    # amp3
        (0.5, 2.0),    # amp4
    ]
    
    # Use differential evolution for global search
    # This is a structural change: instead of L-BFGS-B on 400 dims,
    # we use DE on 12 params, which is much more likely to find good basins.
    try:
        if time.time() < deadline - 60:
            result = differential_evolution(
                _chirp_pair_objective,
                bounds=bounds,
                seed=42,
                maxiter=100,
                tol=1e-8,
                popsize=15,
                polish=True,
                update_interval=5,
                callback=None
            )
            
            if result.fun < best_val:
                h_candidate = _chirp_pair_h(result.x)
                v_candidate = _evaluate(h_candidate)
                if v_candidate < best_val:
                    best_val = v_candidate
                    best_h = h_candidate.copy()
    except Exception:
        pass
    
    # Phase 2: Try several random chirp configurations and refine locally
    rng = np.random.default_rng(7)
    
    for i in range(20):
        if time.time() > deadline - 20:
            break
        
        # Random chirp parameters
        params = np.array([
            rng.uniform(0.5, 5.0),
            rng.uniform(0.5, 5.0),
            rng.uniform(0, 2*np.pi),
            rng.uniform(0, 2*np.pi),
            rng.uniform(0.5, 2.0),
            rng.uniform(0.5, 2.0),
            rng.uniform(0.5, 5.0),
            rng.uniform(0.5, 5.0),
            rng.uniform(0, 2*np.pi),
            rng.uniform(0, 2*np.pi),
            rng.uniform(0.5, 2.0),
            rng.uniform(0.5, 2.0),
        ])
        
        h_candidate = _chirp_pair_h(params)
        v_candidate = _evaluate(h_candidate)
        
        if v_candidate < best_val:
            best_val = v_candidate
            best_h = h_candidate.copy()
        
        # Local refinement on this candidate
        if time.time() < deadline - 10:
            h_refined, v_refined = _local_refine(h_candidate, time_budget=3.0)
            if v_refined < best_val:
                best_val = v_refined
                best_h = h_refined.copy()
    
    # Phase 3: Final local refinement on the best solution found
    if time.time() < deadline - 5:
        h_final, v_final = _local_refine(best_h, time_budget=min(15.0, deadline - time.time() - 5))
        if v_final < best_val:
            best_val = v_final
            best_h = h_final.copy()
    
    # Ensure validity
    if not np.all(np.isfinite(best_h)) or np.sum(best_h) == 0:
        return np.ones(N)
    
    return best_h
