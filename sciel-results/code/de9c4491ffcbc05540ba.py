import numpy as np
from scipy.optimize import minimize
import time

N = 100

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    # Initial guess: uniform
    h0 = np.full(N, 0.5)
    
    def objective(x):
        h = np.clip(x, 0, 1)
        # Enforce sum constraint by projecting
        # Actually, let's use a different approach: parameterize with h_i = 0.5 + d_i where sum(d_i) = 0
        # But for now, let's just compute overlap and add penalty for constraint violation
        s = np.sum(h)
        penalty = 1000 * (s - N/2)**2
        # Compute circular cross-correlation with 1-h
        one_minus_h = 1 - h
        max_overlap = 0
        for k in range(N):
            shifted = np.roll(one_minus_h, k)
            overlap = np.sum(h * shifted)
            if overlap > max_overlap:
                max_overlap = overlap
        return max_overlap + penalty
    
    def compute_score(h):
        """Compute the actual overlap score for a valid h."""
        one_minus_h = 1 - h
        max_overlap = 0
        for k in range(N):
            shifted = np.roll(one_minus_h, k)
            overlap = np.sum(h * shifted)
            if overlap > max_overlap:
                max_overlap = overlap
        return max_overlap
    
    # Try Nelder-Mead first with a reasonable number of iterations
    best_h = h0.copy()
    best_score = compute_score(best_h)
    
    try:
        result = minimize(objective, h0, method='Nelder-Mead', 
                         options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8})
        h = np.clip(result.x, 0, 1)
        # Project to satisfy sum constraint
        s = np.sum(h)
        if abs(s - N/2) > 1e-10:
            h = h + (N/2 - s) / N
            h = np.clip(h, 0, 1)
            # Re-project iteratively
            for _ in range(10):
                s = np.sum(h)
                if abs(s - N/2) < 1e-10:
                    break
                h = h + (N/2 - s) / N
                h = np.clip(h, 0, 1)
        
        score = compute_score(h)
        if score < best_score:
            best_h = h
            best_score = score
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline - 5:
        return best_h
    
    # Try a few more local optimizations from different starting points
    for trial in range(3):
        if time.time() > deadline - 10:
            break
        
        # Random perturbation
        np.random.seed(trial + 42)
        h_init = np.clip(np.random.rand(N) * 0.2 + 0.4, 0, 1)
        # Project to satisfy sum
        for _ in range(20):
            s = np.sum(h_init)
            if abs(s - N/2) < 1e-10:
                break
            h_init = h_init + (N/2 - s) / N
            h_init = np.clip(h_init, 0, 1)
        
        try:
            result = minimize(objective, h_init, method='Nelder-Mead',
                            options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-8})
            h = np.clip(result.x, 0, 1)
            s = np.sum(h)
            if abs(s - N/2) > 1e-10:
                for _ in range(10):
                    s = np.sum(h)
                    if abs(s - N/2) < 1e-10:
                        break
                    h = h + (N/2 - s) / N
                    h = np.clip(h, 0, 1)
            
            score = compute_score(h)
            if score < best_score:
                best_h = h
                best_score = score
        except Exception:
            pass
    
    return best_h
