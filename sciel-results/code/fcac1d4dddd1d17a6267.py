import numpy as np
from scipy.optimize import minimize
import time

N = 100
DEADLINE = 140.0  # seconds, with safety margin

def _overlap_for_shift(h, k):
    shifted = np.roll(h, -k)
    return np.sum(np.minimum(h, 1.0 - shifted))

def _max_overlap_raw(h):
    max_val = 0.0
    for k in range(N):
        val = _overlap_for_shift(h, k)
        if val > max_val:
            max_val = val
    return max_val

def _smoothed_objective(h, alpha=20.0):
    S = np.array([_overlap_for_shift(h, k) for k in range(N)])
    max_S = np.max(S)
    if max_S < 1e-15:
        return 0.0
    return (1.0/alpha) * (max_S + np.log(np.sum(np.exp(alpha * (S - max_S)))))

def construct():
    start_time = time.time()
    
    h0 = np.full(N, 0.5)
    
    constraints = [
        {'type': 'eq', 'fun': lambda h: np.sum(h) - N/2, 'jac': lambda h: np.ones(N)},
    ]
    bounds = [(0, 1)] * N
    
    best_h = h0.copy()
    best_score = _max_overlap_raw(h0)
    
    # Try multiple alphas for smoothing
    for alpha in [10.0, 20.0, 50.0]:
        if time.time() - start_time > DEADLINE - 20:
            break
        
        def obj(h):
            return _smoothed_objective(h, alpha)
        
        def obj_grad(h):
            # Gradient of smoothed objective
            S = np.array([_overlap_for_shift(h, k) for k in range(N)])
            max_S = np.max(S)
            exp_terms = np.exp(alpha * (S - max_S))
            weights = exp_terms / np.sum(exp_terms)
            # dS_k/dh_i
            grad = np.zeros(N)
            for k in range(N):
                shifted_idx = (np.arange(N) + k) % N
                # d min(h_i, 1-h_j)/dh_i = 1 if h_i < 1-h_j, else 0
                # d min(h_i, 1-h_j)/dh_j = -1 if h_i > 1-h_j, else 0
                h_i = h
                h_j = h[shifted_idx]
                # For each i, contribution to dS_k/dh_i
                # min(h_i, 1-h_j) where j = (i+k)%N
                # d/dh_i: 1 if h_i < 1-h_j
                # d/dh_j: -1 if h_i > 1-h_j
                cond_i = h_i < (1.0 - h_j)
                grad += weights[k] * cond_i.astype(float)
                # d/dh_j where j=(i+k)%N, i.e., for index j, i=(j-k)%N
                inv_idx = (np.arange(N) - k) % N
                cond_j = h[inv_idx] > (1.0 - h)
                grad -= weights[k] * cond_j.astype(float)
            return grad
        
        try:
            result = minimize(obj, h0, method='SLSQP', bounds=bounds, 
                             constraints=constraints, jac=obj_grad,
                             options={'maxiter': 500, 'ftol': 1e-12})
            if result.success or result.fun < best_score:
                h_candidate = np.clip(result.x, 0, 1)
                # Project
                diff = N/2 - np.sum(h_candidate)
                h_candidate += diff / N
                h_candidate = np.clip(h_candidate, 0, 1)
                score = _max_overlap_raw(h_candidate)
                if score < best_score:
                    best_score = score
                    best_h = h_candidate.copy()
        except Exception:
            pass
    
    return best_h
