import numpy as np
import time
from scipy.optimize import minimize

N = 100

def _score(h):
    """Calculate the maximum overlap of h with shifted (1-h)."""
    max_overlap = 0.0
    for k in range(N):
        h_part = h[:N-k]
        h_shifted = h[k:]
        overlap_k = np.sum(h_part * (1.0 - h_shifted))
        if overlap_k > max_overlap:
            max_overlap = overlap_k
    return max_overlap

def _objective(x):
    """Objective function for optimization. x represents h values."""
    h = x
    # Penalty for constraint violations
    sum_h = np.sum(h)
    target = N / 2.0
    sum_penalty = 1000.0 * abs(sum_h - target)
    
    # Penalty for bounds
    below = np.sum(np.maximum(0, -h))
    above = np.sum(np.maximum(0, h - 1.0))
    bound_penalty = 1000.0 * (below + above)
    
    # The actual overlap objective
    max_overlap = 0.0
    for k in range(N):
        h_part = h[:N-k]
        h_shifted = h[k:]
        overlap_k = np.sum(h_part * (1.0 - h_shifted))
        if overlap_k > max_overlap:
            max_overlap = overlap_k
    
    return max_overlap + sum_penalty + bound_penalty

def _soft_max_overlaps(h, temperature=0.1):
    """Soft approximation of max overlap for smoother gradients."""
    overlaps = []
    for k in range(N):
        h_part = h[:N-k]
        h_shifted = h[k:]
        overlaps.append(np.sum(h_part * (1.0 - h_shifted)))
    overlaps = np.array(overlaps)
    # Soft max
    max_val = np.max(overlaps)
    return max_val + temperature * np.log(np.sum(np.exp((overlaps - max_val) / temperature)))

def construct():
    start_time = time.time()
    deadline = 140.0  # 10s margin before 150s limit
    
    # Start with a good initial guess
    # Try a pattern: alternate between low and high values
    # or use a sinusoidal pattern
    
    np.random.seed(42)
    
    # Initial: constant 0.5
    h_init = np.full(N, 0.5)
    
    # Try several initializations
    best_h = h_init.copy()
    best_score = _score(best_h)
    
    # Try some structured initializations
    initializations = []
    
    # Alternating pattern
    alt = np.zeros(N)
    alt[0::2] = 1.0
    alt[1::2] = 0.0
    # Scale to sum = N/2
    alt_sum = np.sum(alt)
    if alt_sum > 0:
        alt = alt * (N/2.0) / alt_sum
    initializations.append(alt.copy())
    
    # Sinusoidal
    t = np.linspace(0, 2*np.pi, N, endpoint=False)
    for freq in [1, 2, 3, 5]:
        sin_h = 0.5 + 0.5 * np.sin(freq * t)
        initializations.append(sin_h.copy())
        
    # Random
    for _ in range(5):
        rand_h = np.random.rand(N)
        rand_h = (rand_h - np.mean(rand_h)) * 0.3 + 0.5
        rand_h = np.clip(rand_h, 0, 1)
        # Adjust sum
        diff = N/2.0 - np.sum(rand_h)
        rand_h += diff / N
        rand_h = np.clip(rand_h, 0, 1)
        initializations.append(rand_h.copy())
    
    # Evaluate all initializations
    for h_init in initializations:
        # Ensure valid
        h_init = np.clip(h_init, 0, 1)
        diff = N/2.0 - np.sum(h_init)
        h_init += diff / N
        h_init = np.clip(h_init, 0, 1)
        
        s = _score(h_init)
        if s < best_score:
            best_score = s
            best_h = h_init.copy()
    
    # Now use scipy optimization from the best start
    x0 = best_h.copy()
    
    # Use Nelder-Mead or L-BFGS-B
    # L-BFGS-B with bounds
    bounds = [(0.0, 1.0)] * N
    
    # First, try L-BFGS-B with soft max for smoother objective
    def _objective_smooth(x, temp=0.05):
        h = x
        sum_h = np.sum(h)
        target = N / 2.0
        sum_penalty = 500.0 * (sum_h - target)**2
        
        below = np.sum(np.maximum(0, -h))
        above = np.sum(np.maximum(0, h - 1.0))
        bound_penalty = 500.0 * (below + above)**2
        
        overlaps = []
        for k in range(N):
            h_part = h[:N-k]
            h_shifted = h[k:]
            overlaps.append(np.sum(h_part * (1.0 - h_shifted)))
        overlaps = np.array(overlaps)
        max_val = np.max(overlaps)
        soft_max = max_val + temp * np.log(np.sum(np.exp((overlaps - max_val) / temp)))
        
        return soft_max + sum_penalty + bound_penalty
    
    # Optimization iterations
    max_iterations = 20
    for i in range(max_iterations):
        if time.time() > deadline:
            break
        
        try:
            result = minimize(
                _objective_smooth,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            h_new = np.clip(result.x, 0, 1)
            # Fix sum
            diff = N/2.0 - np.sum(h_new)
            h_new += diff / N
            h_new = np.clip(h_new, 0, 1)
            
            s_new = _score(h_new)
            if s_new < best_score:
                best_score = s_new
                best_h = h_new.copy()
                x0 = h_new.copy()
        except Exception:
            break
    
    # Final local refinement with Nelder-Mead on the exact objective
    if time.time() < deadline - 5:
        try:
            result = minimize(
                _objective,
                best_h,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            
            h_new = np.clip(result.x, 0, 1)
            diff = N/2.0 - np.sum(h_new)
            h_new += diff / N
            h_new = np.clip(h_new, 0, 1)
            
            s_new = _score(h_new)
            if s_new < best_score:
                best_score = s_new
                best_h = h_new.copy()
        except Exception:
            pass
    
    # Ensure exact sum
    diff = N/2.0 - np.sum(best_h)
    best_h += diff / N
    best_h = np.clip(best_h, 0, 1)
    
    # Final correction: adjust one element to get exact sum
    diff = N/2.0 - np.sum(best_h)
    if abs(diff) > 1e-10:
        # Find an element we can adjust
        for i in range(N):
            if 0 < best_h[i] < 1:
                new_val = best_h[i] + diff
                if 0 <= new_val <= 1:
                    best_h[i] = new_val
                    break
        else:
            # Distribute
            best_h += diff / N
    
    return best_h
