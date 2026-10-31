"""Rewritten program: piecewise-linear knot optimization for Erdos minimum overlap."""
import numpy as np
from scipy.optimize import minimize
import time

N = 100
DEADLINE = 120.0
SAFETY_MARGIN = 5.0  # seconds before deadline to stop and return best


def _compute_overlap(h):
    """Compute the maximum overlap of h with shifted 1-h.
    
    The overlap at shift k is sum_i h[i] * (1-h[(i+k) % N]).
    We want to minimize the maximum over all shifts k.
    """
    n = len(h)
    one_minus_h = 1.0 - h
    max_overlap = 0.0
    for k in range(n):
        shifted = np.roll(one_minus_h, k)
        overlap = np.dot(h, shifted)
        if overlap > max_overlap:
            max_overlap = overlap
    return max_overlap


def _compute_overlap_smooth(h, t=10.0):
    """Smooth surrogate using log-sum-exp for the max over shifts."""
    n = len(h)
    one_minus_h = 1.0 - h
    overlaps = np.empty(n)
    for k in range(n):
        shifted = np.roll(one_minus_h, k)
        overlaps[k] = np.dot(h, shifted)
    # log-sum-exp for smooth max
    m = np.max(overlaps)
    return m + np.log(np.sum(np.exp(t * (overlaps - m)))) / t


def _knots_to_heights(knot_positions, knot_values, n):
    """Interpolate knot values at discrete points [0, n) using linear interpolation.
    
    knot_positions: array of positions in [0, 2*n) (periodic extension)
    knot_values: array of values in [0, 1]
    Returns array of length n.
    """
    # We work on [0, 2) scaled to [0, 2n)
    # Positions are in [0, 2n)
    xs = np.arange(n, dtype=float)
    
    # Make sure knots are sorted
    order = np.argsort(knot_positions)
    kp = knot_positions[order]
    kv = knot_values[order]
    
    # Periodic: extend by adding n to positions and wrapping
    # For interpolation on [0, n), use knots in [0, n) and wrap
    # Actually let's work on [0, n) with periodic boundary
    
    # Normalize positions to [0, n)
    kp = kp % n
    
    # Re-sort after modulo
    order2 = np.argsort(kp)
    kp = kp[order2]
    kv = kv[order2]
    
    # For periodic linear interpolation, we need to handle wrap-around
    # Use np.interp with periodic extension
    kp_ext = np.concatenate([kp - n, kp, kp + n])
    kv_ext = np.concatenate([kv, kv, kv])
    
    heights = np.interp(xs, kp_ext, kv_ext, period=n)
    return heights


def _objective(params, n_knots, n=N):
    """Objective: maximize overlap (we minimize negative, or just minimize the overlap).
    
    params: first n_knots are knot positions in [0, n), next n_knots are knot values.
    We enforce integral constraint by scaling.
    """
    kp = params[:n_knots]
    kv = params[n_knots:]
    
    # Clip values to [0, 1]
    kv = np.clip(kv, 0.0, 1.0)
    
    # Clip positions to [0, n)
    kp = np.mod(kp, n)
    
    heights = _knots_to_heights(kp, kv, n)
    
    # Enforce integral constraint: sum(heights) should be n/2
    # Scale values to meet constraint
    current_sum = np.sum(heights)
    if current_sum > 1e-10:
        # Scale so that sum = n/2
        scale = (n / 2.0) / current_sum
        heights = heights * scale
        # Re-clip to [0, 1] and renormalize if needed
        heights = np.clip(heights, 0.0, 1.0)
        # If clipping changed the sum, do iterative scaling
        for _ in range(10):
            s = np.sum(heights)
            if abs(s - n/2.0) < 1e-6:
                break
            heights = heights * (n/2.0) / s
            heights = np.clip(heights, 0.0, 1.0)
    
    # Compute smooth max overlap
    return _compute_overlap_smooth(heights, t=5.0)


def _raw_objective(params, n_knots, n=N):
    """Raw (nonsmooth) objective for final evaluation."""
    kp = params[:n_knots]
    kv = params[n_knots:]
    kv = np.clip(kv, 0.0, 1.0)
    kp = np.mod(kp, n)
    heights = _knots_to_heights(kp, kv, n)
    current_sum = np.sum(heights)
    if current_sum > 1e-10:
        scale = (n / 2.0) / current_sum
        heights = heights * scale
        heights = np.clip(heights, 0.0, 1.0)
        for _ in range(10):
            s = np.sum(heights)
            if abs(s - n/2.0) < 1e-6:
                break
            heights = heights * (n/2.0) / s
            heights = np.clip(heights, 0.0, 1.0)
    return _compute_overlap(heights)


def construct():
    start_time = time.time()
    best_heights = np.full(N, 0.5)
    best_score = _compute_overlap(best_heights)
    
    # Try different numbers of knots
    for n_knots in [4, 6, 8, 12]:
        if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
            break
        
        # Initial guess: symmetric knots
        kp_init = np.linspace(0, N, n_knots, endpoint=False)
        # Alternate high and low values
        kv_init = np.where(np.arange(n_knots) % 2 == 0, 1.0, 0.0)
        params_init = np.concatenate([kp_init, kv_init])
        
        # Nelder-Mead optimization
        try:
            result = minimize(
                _objective,
                params_init,
                args=(n_knots,),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-4, 'fatol': 1e-5, 'adaptive': True}
            )
        except Exception:
            continue
        
        if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
            break
        
        # Evaluate the result
        kp = np.mod(result.x[:n_knots], N)
        kv = np.clip(result.x[n_knots:], 0.0, 1.0)
        heights = _knots_to_heights(kp, kv, N)
        
        # Enforce integral
        current_sum = np.sum(heights)
        if current_sum > 1e-10:
            scale = (N / 2.0) / current_sum
            heights = heights * scale
            heights = np.clip(heights, 0.0, 1.0)
            for _ in range(20):
                s = np.sum(heights)
                if abs(s - N/2.0) < 1e-8:
                    break
                heights = heights * (N/2.0) / s
                heights = np.clip(heights, 0.0, 1.0)
        
        # Final check: ensure sum is exactly N/2
        diff = N/2.0 - np.sum(heights)
        if abs(diff) > 1e-10:
            # Distribute the difference evenly
            heights += diff / N
            heights = np.clip(heights, 0.0, 1.0)
            # Re-adjust
            for _ in range(50):
                s = np.sum(heights)
                if abs(s - N/2.0) < 1e-10:
                    break
                heights += (N/2.0 - s) / N
                heights = np.clip(heights, 0.0, 1.0)
        
        score = _compute_overlap(heights)
        if score < best_score:
            best_score = score
            best_heights = heights.copy()
    
    # If time remains, do a polish pass with more iterations on the best configuration
    if time.time() - start_time < DEADLINE - SAFETY_MARGIN:
        n_knots_best = 8  # Use the middle size
        # Reconstruct params from best_heights by fitting knots
        # Simple approach: use uniform knots and fit values
        kp_fit = np.linspace(0, N, n_knots_best, endpoint=False)
        # Sample heights at knot positions
        kv_fit = np.interp(kp_fit, np.arange(N), best_heights, period=N)
        params_fit = np.concatenate([kp_fit, kv_fit])
        
        try:
            result2 = minimize(
                _objective,
                params_fit,
                args=(n_knots_best,),
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-5, 'fatol': 1e-6, 'adaptive': True}
            )
        except Exception:
            result2 = None
        
        if result2 is not None and time.time() - start_time < DEADLINE - SAFETY_MARGIN:
            kp = np.mod(result2.x[:n_knots_best], N)
            kv = np.clip(result2.x[n_knots_best:], 0.0, 1.0)
            heights = _knots_to_heights(kp, kv, N)
            current_sum = np.sum(heights)
            if current_sum > 1e-10:
                scale = (N / 2.0) / current_sum
                heights = heights * scale
                heights = np.clip(heights, 0.0, 1.0)
                for _ in range(20):
                    s = np.sum(heights)
                    if abs(s - N/2.0) < 1e-8:
                        break
                    heights = heights * (N/2.0) / s
                    heights = np.clip(heights, 0.0, 1.0)
            
            diff = N/2.0 - np.sum(heights)
            if abs(diff) > 1e-10:
                heights += diff / N
                heights = np.clip(heights, 0.0, 1.0)
                for _ in range(50):
                    s = np.sum(heights)
                    if abs(s - N/2.0) < 1e-10:
                        break
                    heights += (N/2.0 - s) / N
                    heights = np.clip(heights, 0.0, 1.0)
            
            score = _compute_overlap(heights)
            if score < best_score:
                best_score = score
                best_heights = heights.copy()
    
    # Final enforcement: ensure exact sum
    diff = N/2.0 - np.sum(best_heights)
    if abs(diff) > 1e-10:
        best_heights += diff / N
        best_heights = np.clip(best_heights, 0.0, 1.0)
        for _ in range(100):
            s = np.sum(best_heights)
            if abs(s - N/2.0) < 1e-12:
                break
            best_heights += (N/2.0 - s) / N
            best_heights = np.clip(best_heights, 0.0, 1.0)
    
    return best_heights
