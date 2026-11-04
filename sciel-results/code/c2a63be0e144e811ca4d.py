import numpy as np
import time
from scipy.optimize import minimize
from scipy.signal import convolve

N = 50
START_TIME = time.time()
TIME_LIMIT = 110.0  # Safety margin for 120s limit

def evaluate_ratio(h):
    """Compute the exact ratio for a given vector h."""
    h = np.asarray(h)
    if np.any(h < -1e-12):
        return 0.0
    h = np.clip(h, 0, None)
    l1 = np.sum(h)
    if l1 < 1e-12:
        return 0.0
    h_norm = h / l1
    conv = convolve(h_norm, h_norm, mode='full')
    sup = np.max(conv)
    if sup < 1e-12:
        return 0.0
    l2_sq = np.sum(h_norm**2)
    return l2_sq / sup

def soft_max(x, t):
    """Stable soft-max approximation of max(x)."""
    x_max = np.max(x)
    # If t is very large, t*(x-x_max) can underflow for x < x_max, which is fine (exp(0)=1 for max, exp(-inf)=0)
    # But if t is small, it's just a mean-like thing.
    return x_max + np.log(np.sum(np.exp(t * (x - x_max)))) / t

def objective_func(h, t):
    """
    Objective to minimize: -log(ratio) using soft-max for sup.
    Ratio = L2^2 / (L1 * Sup). Since we normalize L1=1 internally, 
    we maximize L2^2 / Sup.
    Minimize -log(L2^2) + log(Sup).
    """
    h = np.asarray(h)
    if np.any(h < -1e-12):
        return 1e10
    h = np.clip(h, 0, None)
    l1 = np.sum(h)
    if l1 < 1e-12:
        return 1e10
    h_norm = h / l1
    conv = convolve(h_norm, h_norm, mode='full')
    
    sup_soft = soft_max(conv, t)
    
    l2_sq = np.sum(h_norm**2)
    if l2_sq < 1e-12 or sup_soft < 1e-12:
        return 1e10
    
    return -np.log(l2_sq) + np.log(sup_soft)

def construct():
    best_h = np.ones(N)
    best_score = evaluate_ratio(best_h)
    
    # Generate candidate index sets
    index_sets = []
    
    # 1. Regular spacing
    for k in range(2, 12):
        idx = list(range(0, N, k))
        if len(idx) > 0:
            index_sets.append(idx)
            
    # 2. Greedy max-min separation (Sidon-like heuristic)
    def greedy_max_min(num_points):
        points = [0]
        while len(points) < num_points:
            best_pos = -1
            best_dist = -1
            for i in range(N):
                if i in points:
                    continue
                # Calculate min distance to existing points
                min_dist = min(abs(i - p) for p in points)
                if min_dist > best_dist:
                    best_dist = min_dist
                    best_pos = i
            if best_pos == -1: break
            points.append(best_pos)
        return points
    
    for num_p in [4, 6, 8, 10, 12, 15]:
        index_sets.append(greedy_max_min(num_p))
        
    # 3. Random sparse sets
    np.random.seed(42)
    for _ in range(10):
        num_p = np.random.randint(4, 15)
        idx = np.random.choice(N, num_p, replace=False)
        index_sets.append(list(idx))

    # Optimization parameters
    t_values = [5.0, 20.0, 100.0]
    bounds = [(0, None)] * N

    for idx_set in index_sets:
        if time.time() - START_TIME > TIME_LIMIT:
            break
            
        # Initialize h
        h0 = np.zeros(N)
        h0[idx_set] = 1.0
        
        current_h = h0.copy()
        
        # Anneal the soft-max temperature
        for t in t_values:
            if time.time() - START_TIME > TIME_LIMIT:
                break
                
            try:
                res = minimize(
                    objective_func, 
                    current_h, 
                    args=(t,), 
                    method='L-BFGS-B', 
                    bounds=bounds,
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                current_h = res.x
                current_h = np.clip(current_h, 0, None)
            except Exception:
                pass
        
        # Final evaluation with exact max
        score = evaluate_ratio(current_h)
        
        if score > best_score:
            best_score = score
            best_h = current_h.copy()

    return list(best_h)
