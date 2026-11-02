import numpy as np
from scipy.optimize import minimize
import time

N = 100

def compute_overlap(h):
    """Compute the maximum overlap of h with shifted (1-h)."""
    # h is array of length N, values in [0,1], sum = N/2
    # overlap at shift k: sum_i h[i] * (1 - h[(i+k) % N])
    max_overlap = 0.0
    for k in range(N):
        shifted = np.roll(h, -k)
        overlap = np.sum(h * (1 - shifted))
        max_overlap = max(max_overlap, overlap)
    return max_overlap

def objective(x):
    """Objective: maximize the minimum autocorrelation, equivalent to minimizing max overlap."""
    h = x
    # Compute autocorrelation
    min_auto = np.inf
    for k in range(N):
        shifted = np.roll(h, -k)
        auto = np.sum(h * shifted)
        min_auto = min(min_auto, auto)
    # overlap(k) = N/2 - auto(k), so max overlap = N/2 - min auto
    return N/2 - min_auto

def construct():
    start_time = time.time()
    deadline = 110.0  # safety margin before 120s limit
    
    # Start with bang-bang: alternating 0s and 1s
    h0 = np.zeros(N)
    h0[::2] = 1.0
    # Ensure sum = N/2
    if h0.sum() != N/2:
        h0 = h0 / h0.sum() * (N/2)
    
    # Use scipy minimize with bounds
    bounds = [(0.0, 1.0)] * N
    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - N/2}
    ]
    
    best_h = h0.copy()
    best_score = compute_overlap(best_h)
    
    # Try multiple random restarts
    np.random.seed(42)
    for restart in range(20):
        if time.time() - start_time > deadline:
            break
        
        if restart == 0:
            h_init = h0.copy()
        else:
            # Random bang-bang initialization
            h_init = np.random.choice([0.0, 1.0], size=N)
            # Adjust to satisfy sum constraint
            current_sum = h_init.sum()
            if current_sum > N/2:
                # Set some 1s to 0s
                indices = np.where(h_init == 1.0)[0]
                np.random.shuffle(indices)
                for idx in indices:
                    if h_init.sum() <= N/2:
                        break
                    h_init[idx] = 0.0
            else:
                # Set some 0s to 1s
                indices = np.where(h_init == 0.0)[0]
                np.random.shuffle(indices)
                for idx in indices:
                    if h_init.sum() >= N/2:
                        break
                    h_init[idx] = 1.0
        
        # Optimize
        try:
            result = minimize(objective, h_init, method='SLSQP', bounds=bounds, 
                            constraints=constraints, options={'maxiter': 500, 'ftol': 1e-8})
            
            if result.success or result.fun < best_score:
                h_candidate = result.x
                # Clamp to [0,1]
                h_candidate = np.clip(h_candidate, 0.0, 1.0)
                # Normalize to satisfy sum constraint
                if h_candidate.sum() > 0:
                    h_candidate = h_candidate / h_candidate.sum() * (N/2)
                    h_candidate = np.clip(h_candidate, 0.0, 1.0)
                
                score = compute_overlap(h_candidate)
                if score < best_score:
                    best_score = score
                    best_h = h_candidate.copy()
        except Exception:
            continue
    
    # Final polish with more iterations
    if time.time() - start_time < deadline:
        try:
            result = minimize(objective, best_h, method='SLSQP', bounds=bounds, 
                            constraints=constraints, options={'maxiter': 1000, 'ftol': 1e-10})
            if result.success:
                h_candidate = np.clip(result.x, 0.0, 1.0)
                if h_candidate.sum() > 0:
                    h_candidate = h_candidate / h_candidate.sum() * (N/2)
                    h_candidate = np.clip(h_candidate, 0.0, 1.0)
                score = compute_overlap(h_candidate)
                if score < best_score:
                    best_score = score
                    best_h = h_candidate.copy()
        except Exception:
            pass
    
    return best_h
