"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, LinearConstraint
import time

def _overlap_cost(h):
    """
    Compute the maximum overlap C5 = max_i sum_j h[j] * (1-h[(j-i) % N]).
    Since h is in [0,1], 1-h is also in [0,1].
    The overlap at shift i is sum_j h[j] * (1-h[(j-i)%N]).
    We want to minimize the maximum over i.
    """
    N = len(h)
    max_overlap = 0.0
    for i in range(N):
        # shift h by i
        shifted = np.roll(h, i)
        overlap = np.sum(h * (1.0 - shifted))
        if overlap > max_overlap:
            max_overlap = overlap
    return max_overlap

def _smooth_max_overlap(h, temperature=1.0):
    """
    Smooth surrogate for max overlap using log-sum-exp.
    """
    N = len(h)
    overlaps = np.zeros(N)
    for i in range(N):
        shifted = np.roll(h, i)
        overlaps[i] = np.sum(h * (1.0 - shifted))
    # log-sum-exp for smooth max
    m = np.max(overlaps)
    if m == 0:
        return 0.0
    log_sum_exp = m + np.log(np.sum(np.exp((overlaps - m) / temperature)) / N) * temperature
    # Actually, standard log-sum-exp: log(sum(exp(x_i))) = m + log(sum(exp(x_i - m)))
    # We want a smooth approximation to max, which is (1/T) * log(sum(exp(T * x_i)))
    # Let's use that
    T = temperature
    scaled = T * overlaps
    m_scaled = np.max(scaled)
    if m_scaled == 0:
        return 0.0
    smooth_max = (m_scaled + np.log(np.sum(np.exp(scaled - m_scaled)) / N)) / T
    return smooth_max

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # 120s limit, keep safety margin
    
    N = 100
    # Sum constraint: sum(h) = N/2 = 50
    # Bounds: 0 <= h[i] <= 1
    
    # Initial guess: uniform 0.5
    h0 = np.full(N, 0.5)
    
    # Try a few different initializations and optimization strategies
    best_h = h0.copy()
    best_score = _overlap_cost(h0)
    
    # Constraints
    bounds = [(0.0, 1.0)] * N
    A_eq = np.ones((1, N))
    b_eq = np.array([N / 2.0])
    
    # Use SLSQP for the nonsmooth objective first (it can handle it somewhat)
    # Then use smooth surrogate for better convergence
    
    # Strategy 1: Direct optimization with SLSQP
    try:
        res = minimize(
            _overlap_cost,
            h0,
            method='SLSQP',
            bounds=bounds,
            constraints=[{'type': 'eq', 'fun': lambda x: np.sum(x) - N/2.0}],
            options={'maxiter': 1000, 'ftol': 1e-12}
        )
        if res.success and res.fun < best_score:
            best_score = res.fun
            best_h = res.x.copy()
    except Exception:
        pass
    
    # Strategy 2: Smooth surrogate optimization
    # Use log-sum-exp with decreasing temperature
    for temp in [10.0, 5.0, 2.0, 1.0, 0.5, 0.2]:
        if time.time() > deadline - 30.0:
            break
        try:
            def smooth_obj(x):
                return _smooth_max_overlap(x, temperature=temp)
            
            res = minimize(
                smooth_obj,
                best_h,
                method='SLSQP',
                bounds=bounds,
                constraints=[{'type': 'eq', 'fun': lambda x: np.sum(x) - N/2.0}],
                options={'maxiter': 500, 'ftol': 1e-12}
            )
            if res.success:
                actual_score = _overlap_cost(res.x)
                if actual_score < best_score:
                    best_score = actual_score
                    best_h = res.x.copy()
        except Exception:
            pass
    
    # Strategy 3: Local polish with tight tolerance
    if time.time() < deadline - 10.0:
        try:
            res = minimize(
                _overlap_cost,
                best_h,
                method='SLSQP',
                bounds=bounds,
                constraints=[{'type': 'eq', 'fun': lambda x: np.sum(x) - N/2.0}],
                options={'maxiter': 2000, 'ftol': 1e-15}
            )
            if res.success and res.fun < best_score:
                best_score = res.fun
                best_h = res.x.copy()
        except Exception:
            pass
    
    # Ensure sum is exactly N/2 and values are in [0,1]
    best_h = np.clip(best_h, 0.0, 1.0)
    # Adjust to satisfy sum constraint exactly
    diff = N/2.0 - np.sum(best_h)
    if abs(diff) > 1e-10:
        # Distribute the difference
        best_h += diff / N
        best_h = np.clip(best_h, 0.0, 1.0)
        # If clipping changed the sum, do a simple projection
        for _ in range(100):
            diff = N/2.0 - np.sum(best_h)
            if abs(diff) < 1e-12:
                break
            # Add/subtract uniformly where possible
            mask = (best_h > 0.0) & (best_h < 1.0)
            if np.any(mask):
                best_h[mask] += diff / np.sum(mask)
                best_h = np.clip(best_h, 0.0, 1.0)
            else:
                break
    
    # Final verification
    final_score = _overlap_cost(best_h)
    if final_score < best_score:
        best_score = final_score
    
    return best_h
# EVOLVE-BLOCK-END
