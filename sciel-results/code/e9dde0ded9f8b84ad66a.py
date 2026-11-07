import numpy as np
import time
from scipy.optimize import minimize
import itertools

N = 100
TARGET_SUM = N / 2.0
DEADLINE = 150.0
START_TIME = time.time()

def _elapsed():
    return time.time() - START_TIME

def _raw_value(h):
    # Compute max correlation / N * 2
    # h must be 1D array
    # 1-h is the other sequence
    corr = np.correlate(h, 1 - h, mode="full")
    max_corr = np.max(corr)
    return max_corr / N * 2

def _is_valid(h):
    if not np.all(np.isfinite(h)):
        return False
    if np.any(h < -1e-9) or np.any(h > 1 + 1e-9):
        return False
    if abs(np.sum(h) - TARGET_SUM) > 1e-7 * N:
        return False
    return True

def _project_and_refine(h):
    # Simple projection to sum constraint
    # This is a heuristic projection, might violate bounds
    # Better to rely on optimizer constraints or penalty
    pass

class ParametricModel:
    def __init__(self, n_sines=2):
        self.n_sines = n_sines
        # params: [A1, w1, p1, A2, w2, p2, ...]
        self.n_params = n_sines * 3

    def get_h(self, params):
        # params are flat array
        h = np.zeros(N)
        for j in range(self.n_sines):
            A = params[j*3]
            w = params[j*3+1]
            p = params[j*3+2]
            h += A * np.sin(w * np.arange(N) + p)
        
        # Adjust mean to satisfy sum constraint
        # Sum(h) = N * mean(h) + sum of sine parts
        # We want Sum(h) = TARGET_SUM
        # Let h_raw = h. Sum_raw = np.sum(h).
        # Add constant c to all h_i.
        # Sum(h + c) = Sum_raw + N*c = TARGET_SUM
        # c = (TARGET_SUM - Sum_raw) / N
        c = (TARGET_SUM - np.sum(h)) / N
        h += c
        return h

    def objective(self, params):
        h = self.get_h(params)
        if not _is_valid(h):
            return 1e6
        return _raw_value(h)

def _slsqp_optimize(model, initial_params):
    # Bounds for parameters
    # A: [-0.5, 0.5] (since h in [0,1], max deviation from 0.5 is 0.5)
    # w: [0.01, 10.0]
    # p: [0, 2*pi]
    bounds = []
    for j in range(model.n_sines):
        bounds.append((-0.5, 0.5)) # A
        bounds.append((0.01, 10.0)) # w
        bounds.append((0, 2*np.pi)) # p
    
    res = minimize(model.objective, initial_params, method='SLSQP', bounds=bounds, options={'maxiter': 100, 'ftol': 1e-8})
    if res.success:
        best_params = res.x
    else:
        best_params = initial_params
        
    h = model.get_h(best_params)
    if not _is_valid(h):
        # Fallback to seed if optimization fails badly
        return np.full(N, 0.5), 0.5
        
    val = _raw_value(h)
    return h, val

def _refine_full(h_init):
    # Coordinate descent polish
    h = h_init.copy()
    val = _raw_value(h)
    best_h = h.copy()
    best_val = val
    
    # Simple gradient-free local search
    # Try small perturbations
    step = 0.01
    for _ in range(50):
        if _elapsed() > DEADLINE - 10:
            break
        improved = False
        for i in range(N):
            # Try increasing and decreasing h[i]
            for delta in [step, -step]:
                h_new = h.copy()
                h_new[i] += delta
                # Re-project sum constraint? 
                # If we change h[i], sum changes. 
                # To keep sum constant, we must adjust another element or all elements.
                # Adjusting all elements slightly is safer for box constraints if we are not close to bounds.
                # Let's adjust all other elements to maintain sum.
                # sum(h_new) = sum(h) + delta.
                # We need sum(h_new) = TARGET_SUM.
                # So we must subtract delta/(N-1) from all other elements.
                if N > 1:
                    adjustment = -delta / (N - 1)
                    mask = np.ones(N, dtype=bool)
                    mask[i] = False
                    h_new[mask] += adjustment
                    
                if _is_valid(h_new):
                    v_new = _raw_value(h_new)
                    if v_new < best_val:
                        best_val = v_new
                        best_h = h_new.copy()
                        h = h_new.copy()
                        improved = True
        if not improved:
            step /= 2
            if step < 1e-5:
                break
                
    return best_h, best_val

def construct():
    if _elapsed() > DEADLINE - 5:
        return np.full(N, 0.5)

    best_h = np.full(N, 0.5)
    best_val = _raw_value(best_h) # 0.5

    # Multi-start with parametric model
    np.random.seed(42)
    
    # Try 1 sine, then 2 sines
    for n_sines in [1, 2]:
        model = ParametricModel(n_sines)
        
        # Generate a few random starts
        for start_idx in range(20):
            if _elapsed() > DEADLINE - 20:
                break
            
            # Random initialization
            params = np.zeros(model.n_params)
            for j in range(n_sines):
                params[j*3] = np.random.uniform(-0.3, 0.3)
                params[j*3+1] = np.random.uniform(0.1, 5.0)
                params[j*3+2] = np.random.uniform(0, 2*np.pi)
            
            # Add a specific "smooth" start
            if start_idx == 0:
                # Low frequency sine
                params[1] = 2 * np.pi / 20.0 
                params[0] = 0.2
                if n_sines > 1:
                    params[4] = 2 * np.pi / 5.0
                    params[3] = 0.1

            h_candidate, val_candidate = _slsqp_optimize(model, params)
            
            if val_candidate < best_val:
                best_val = val_candidate
                best_h = h_candidate.copy()
                
            # Polish the best candidate found so far if it's significantly better
            if val_candidate < best_val + 0.01 and start_idx % 5 == 0:
                if _elapsed() < DEADLINE - 30:
                    h_polished, val_polished = _refine_full(h_candidate)
                    if val_polished < best_val:
                        best_val = val_polished
                        best_h = h_polished.copy()

    # Final polish on the global best
    if _elapsed() < DEADLINE - 10:
        h_final, val_final = _refine_full(best_h)
        if val_final < best_val:
            best_val = val_final
            best_h = h_final.copy()

    return best_h

# EVOLVE-BLOCK-END
