"""Erdos minimum overlap: Low-dimensional parametric curve projection approach."""
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

N = 100
DEADLINE = 145.0  # Safety margin

def _score(h):
    corr = np.correlate(h, 1 - h, mode="full")
    return np.max(corr) / N * 2

def _is_valid(h):
    if not np.all(np.isfinite(h)):
        return False
    if np.any(h < -1e-9) or np.any(h > 1 + 1e-9):
        return False
    if abs(np.sum(h) - N / 2) > 1e-9 * N:
        return False
    return True

def _fix_sum(h):
    target = N / 2.0
    h = h.copy()
    h = np.clip(h, 0.0, 1.0)
    for _ in range(50):
        diff = target - np.sum(h)
        if abs(diff) < 1e-12:
            break
        if diff > 0:
            mask = h < 1.0 - 1e-9
            if not np.any(mask):
                break
            room = 1.0 - h[mask]
            total_room = np.sum(room)
            if total_room < 1e-9:
                break
            add_factor = min(1.0, diff / total_room)
            h[mask] += room * add_factor
        else:
            mask = h > 1e-9
            if not np.any(mask):
                break
            val = h[mask]
            total_val = np.sum(val)
            if total_val < 1e-9:
                break
            sub_factor = min(1.0, -diff / total_val)
            h[mask] -= val * sub_factor
    diff = target - np.sum(h)
    if abs(diff) > 1e-12:
        if diff > 0:
            idx = np.argmin(h)
            if h[idx] < 1.0:
                h[idx] += diff
        else:
            idx = np.argmax(h)
            if h[idx] > 0.0:
                h[idx] += diff
        h = np.clip(h, 0.0, 1.0)
    return h

def _param_to_h(params):
    """
    Convert 13 parameters to a height sequence of length 100.
    Params: [bias, a1, f1, p1, a2, f2, p2, a3, f3, p3, a4, f4, p4]
    We use 4 sinusoids to allow for more complex structures.
    bias in [-0.5, 0.5]
    a in [0, 1]
    f in [0.01, 2.0]
    p in [0, 2*pi]
    """
    bias = params[0]
    a1, f1, p1 = params[1:4]
    a2, f2, p2 = params[4:7]
    a3, f3, p3 = params[7:10]
    a4, f4, p4 = params[10:13]
    
    x = np.arange(N)
    h = bias + a1 * np.sin(2 * np.pi * f1 * x + p1) \
               + a2 * np.sin(2 * np.pi * f2 * x + p2) \
               + a3 * np.sin(2 * np.pi * f3 * x + p3) \
               + a4 * np.sin(2 * np.pi * f4 * x + p4)
    
    # Clip to [0, 1]
    h = np.clip(h, 0.0, 1.0)
    
    # Fix sum to be exactly 50
    h = _fix_sum(h)
    
    return h

def _objective(params):
    h = _param_to_h(params)
    return _score(h)

def construct():
    start_time = time.time()
    
    def deadline_check():
        return time.time() > start_time + DEADLINE

    # Initial best from seed
    best_h = np.full(N, 0.5)
    best_score = _score(best_h)

    # Define bounds for the 13 parameters
    # bias: [-0.5, 0.5]
    # a: [0, 1]
    # f: [0.01, 2.0]
    # p: [0, 2*pi]
    bounds = [(-0.5, 0.5),
              (0, 1), (0.01, 2.0), (0, 2*np.pi),
              (0, 1), (0.01, 2.0), (0, 2*np.pi),
              (0, 1), (0.01, 2.0), (0, 2*np.pi),
              (0, 1), (0.01, 2.0), (0, 2*np.pi)]

    # Use Differential Evolution to explore the 13D space
    # This is robust and handles non-smoothness reasonably well
    try:
        if not deadline_check():
            res = differential_evolution(
                _objective,
                bounds,
                seed=42,
                maxiter=50,
                popsize=15,
                tol=1e-8,
                mut=(0.5, 1.0),
                reccomb=0.9,
                polish=False,
                init='latinhypercube'
            )
            if res.success or res.fun < best_score:
                h_candidate = _param_to_h(res.x)
                if _is_valid(h_candidate):
                    s_candidate = _score(h_candidate)
                    if s_candidate < best_score:
                        best_score = s_candidate
                        best_h = h_candidate.copy()
    except Exception:
        pass

    # Refine using local optimization (L-BFGS-B or Nelder-Mead) on the best found params
    if not deadline_check():
        # We need to recover the best params from the previous step or do a small grid search if DE failed
        # Since we don't store res.x globally, we'll just run a few local optimizations from random starts in param space
        rng = np.random.RandomState(12345)
        
        # Try a few random starts for local refinement
        for i in range(10):
            if deadline_check():
                break
            
            # Random start in param space
            params0 = np.array([
                rng.uniform(-0.5, 0.5),
                rng.uniform(0, 1), rng.uniform(0.01, 2.0), rng.uniform(0, 2*np.pi),
                rng.uniform(0, 1), rng.uniform(0.01, 2.0), rng.uniform(0, 2*np.pi),
                rng.uniform(0, 1), rng.uniform(0.01, 2.0), rng.uniform(0, 2*np.pi),
                rng.uniform(0, 1), rng.uniform(0.01, 2.0), rng.uniform(0, 2*np.pi)
            ])
            
            try:
                res2 = minimize(
                    _objective,
                    params0,
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'fatol': 1e-10}
                )
                if res2.fun < best_score:
                    h_candidate = _param_to_h(res2.x)
                    if _is_valid(h_candidate):
                        s_candidate = _score(h_candidate)
                        if s_candidate < best_score:
                            best_score = s_candidate
                            best_h = h_candidate.copy()
            except Exception:
                pass

    # Final validation
    if not _is_valid(best_h):
        best_h = np.full(N, 0.5)
    
    return best_h
