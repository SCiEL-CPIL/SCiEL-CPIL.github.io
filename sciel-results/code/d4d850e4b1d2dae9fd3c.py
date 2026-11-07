import time
import numpy as np
from scipy.optimize import minimize

N = 100
N_CP = 10

def _interp_cp(cp):
    cp_pos = np.linspace(0, 1, N_CP)
    sample_pos = np.linspace(0, 1, N)
    return np.interp(sample_pos, cp_pos, cp)

def _value(h):
    corr = np.correlate(h, 1 - h, mode="full")
    return np.max(corr) / N * 2

def _project(h):
    h = np.clip(h, 0.0, 1.0)
    diff = np.sum(h) - N / 2.0
    for _ in range(20):
        if abs(diff) < 1e-12:
            break
        if diff > 0:
            mask = h > 1e-10
            if not np.any(mask): break
            h[mask] -= diff / np.sum(mask)
        else:
            mask = h < 1 - 1e-10
            if not np.any(mask): break
            h[mask] -= diff / np.sum(mask)
        h = np.clip(h, 0.0, 1.0)
        diff = np.sum(h) - N / 2.0
    return h

def _local_polish(h, max_iters=100):
    best_h = h.copy()
    best_val = _value(best_h)
    n = len(best_h)
    step_sizes = [0.01, 0.005, 0.002]
    
    for _ in range(max_iters):
        improved = False
        for i in range(n):
            for d in step_sizes:
                for sign in [1, -1]:
                    new_h = best_h.copy()
                    new_h[i] += sign * d
                    if new_h[i] < 0 or new_h[i] > 1:
                        continue
                    
                    # Pick a compensating index j
                    # Heuristic: opposite side to minimize local correlation peaks
                    j = (i + n // 2) % n
                    if j == i: j = (i + 1) % n
                    
                    new_hj = best_h[j] - sign * d
                    if new_hj < 0 or new_hj > 1:
                        continue
                    
                    new_h[j] = new_hj
                    val = _value(new_h)
                    if val < best_val - 1e-13:
                        best_val = val
                        best_h = new_h
                        improved = True
                        break
                if improved: break
            if improved: break
        if not improved: break
        
    return best_h

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    best_val = float('inf')
    best_h = np.full(N, 0.5)
    
    def obj(cp):
        h = _interp_cp(cp)
        return _value(h)
        
    def cons_eq(cp):
        h = _interp_cp(cp)
        return np.sum(h) - N / 2.0
        
    constraints = {'type': 'eq', 'fun': cons_eq}
    bounds = [(0.0, 1.0)] * N_CP
    
    rng = np.random.default_rng(42)
    starts = []
    
    # Random starts
    for _ in range(150):
        starts.append(rng.uniform(0, 1, N_CP))
        
    # Structured starts
    starts.append(np.linspace(0, 1, N_CP))
    starts.append(np.linspace(1, 0, N_CP))
    for f in [1, 2, 3, 4]:
        x = np.linspace(0, 1, N_CP)
        starts.append(0.5 + 0.4 * np.sin(2 * np.pi * f * x))
        starts.append(0.5 + 0.4 * np.cos(2 * np.pi * f * x))
    starts.append(np.array([1, 0.8, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.8, 1.0]))
    starts.append(np.array([0, 0.2, 0.5, 0.8, 1, 1, 0.8, 0.5, 0.2, 0]))
    starts.append(np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0]))
    starts.append(np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1]))
    
    for cp_start in starts:
        if time.time() > deadline - 10.0:
            break
        cp_start = np.clip(cp_start, 0, 1)
        try:
            res = minimize(
                obj,
                cp_start,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 200, 'ftol': 1e-12}
            )
            if res.fun < best_val:
                cp_opt = np.clip(res.x, 0, 1)
                h_opt = _interp_cp(cp_opt)
                h_opt = _project(h_opt)
                val = _value(h_opt)
                if val < best_val:
                    best_val = val
                    best_h = h_opt.copy()
        except Exception:
            pass
            
    # Local Polish
    if time.time() < deadline - 2.0:
        best_h = _local_polish(best_h, max_iters=50)
        best_val = _value(best_h)
        
    return best_h
