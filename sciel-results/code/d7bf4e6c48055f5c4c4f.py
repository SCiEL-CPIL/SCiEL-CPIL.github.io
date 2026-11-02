"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _overlap_score(h):
    """Compute max over shifts s=1..N-1 of sum_i h_i * (1 - h_{i+s mod N})."""
    h = np.asarray(h)
    n = len(h)
    max_overlap = 0.0
    for s in range(1, n):
        shifted = np.roll(h, s)
        overlap = np.sum(h * (1.0 - shifted))
        if overlap > max_overlap:
            max_overlap = overlap
    return max_overlap

def _compute_all_overlaps(h):
    """Compute overlap for all shifts s=1..N-1."""
    h = np.asarray(h)
    n = len(h)
    overlaps = np.zeros(n - 1)
    for idx, s in enumerate(range(1, n)):
        shifted = np.roll(h, s)
        overlaps[idx] = np.sum(h * (1.0 - shifted))
    return overlaps

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # safety margin
    
    # Start with the known-good constant solution
    best_h = np.full(N, 0.5)
    best_score = _overlap_score(best_h)
    
    rng = np.random.default_rng(42)
    
    # Strategy: Use a convex relaxation approach
    # Minimize the sum of squared deviations of overlaps from their mean
    # This encourages all overlaps to be close to each other, reducing the max
    
    def _objective_and_grad(x):
        """Objective: minimize variance of overlaps + penalty for exceeding bound."""
        h = x[:N]
        # Compute all overlaps
        overlaps = _compute_all_overlaps(h)
        
        # Mean and variance of overlaps
        mean_ov = np.mean(overlaps)
        var_ov = np.var(overlaps)
        
        # Soft penalty for max overlap exceeding current best
        excess = np.maximum(0, overlaps - best_score)
        penalty = 100.0 * np.sum(excess**2)
        
        obj = var_ov + penalty
        return obj, None
    
    # Try multiple initializations
    for restart in range(8):
        if time.time() > deadline - 30:
            break
        
        if restart == 0:
            h0 = best_h.copy()
        elif restart == 1:
            # Alternating pattern
            h0 = np.where(np.arange(N) % 2 == 0, 0.7, 0.3)
            h0 = h0 - (h0.sum() - N/2) / N
            h0 = np.clip(h0, 0, 1)
        elif restart == 2:
            # Smooth variation
            t = np.linspace(0, 1, N)
            h0 = 0.5 + 0.3 * np.sin(2 * np.pi * t)
            h0 = np.clip(h0, 0, 1)
            h0 = h0 - (h0.sum() - N/2) / N
            h0 = np.clip(h0, 0, 1)
        elif restart == 3:
            # Another smooth pattern
            t = np.linspace(0, 1, N)
            h0 = 0.5 + 0.25 * np.cos(4 * np.pi * t)
            h0 = np.clip(h0, 0, 1)
            h0 = h0 - (h0.sum() - N/2) / N
            h0 = np.clip(h0, 0, 1)
        else:
            # Random
            h0 = rng.uniform(0, 1, N)
            h0 = h0 - (h0.sum() - N/2) / N
            h0 = np.clip(h0, 0, 1)
        
        # Use L-BFGS-B with bounds
        bounds = [(0, 1)] * N
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - N/2}]
        
        try:
            result = minimize(
                _objective_and_grad,
                h0,
                method='L-BFGS-B',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if result.success or result.fun < _overlap_score(best_h):
                h_new = result.x
                h_new = np.clip(h_new, 0, 1)
                # Fix sum
                diff = N/2 - h_new.sum()
                h_new = h_new + diff / N
                h_new = np.clip(h_new, 0, 1)
                
                score = _overlap_score(h_new)
                if score < best_score - 1e-9:
                    best_score = score
                    best_h = h_new.copy()
        except Exception as e:
            pass
        
        # Local refinement with subgradient on the best shift
        if time.time() > deadline - 15:
            break
        
        h = best_h.copy()
        for it in range(100):
            if time.time() > deadline - 15:
                break
            
            overlaps = _compute_all_overlaps(h)
            max_shift = np.argmax(overlaps) + 1  # +1 because we start from s=1
            max_ov = overlaps[max(0, max_shift - 1)]
            
            if max_ov < best_score - 1e-9:
                best_score = max_ov
                best_h = h.copy()
            
            # Subgradient of max overlap w.r.t. h
            s = max_shift
            grad = (1.0 - np.roll(h, s)) - np.roll(h, -s)
            
            # Adaptive learning rate
            lr = 0.05 / (1 + it * 0.1)
            h_new = h - lr * grad
            h_new = np.clip(h_new, 0, 1)
            diff = N/2 - h_new.sum()
            h_new = h_new + diff / N
            h_new = np.clip(h_new, 0, 1)
            
            h = h_new
    
    # Ensure exact sum
    best_h = np.clip(best_h, 0, 1)
    diff = N / 2 - best_h.sum()
    if abs(diff) > 1e-12:
        best_h = best_h + diff / N
        best_h = np.clip(best_h, 0, 1)
        remaining = N / 2 - best_h.sum()
        if abs(remaining) > 0:
            idx = np.argmax(best_h) if remaining > 0 else np.argmin(best_h)
            best_h[idx] += remaining
    
    return best_h
# EVOLVE-BLOCK-END
