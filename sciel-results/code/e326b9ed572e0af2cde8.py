"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    # Initial guess: constant 0.5
    h0 = np.full(N, 0.5)
    
    def overlap_at_shift(h, s):
        """Compute sum of h_i * (1 - h_{i+s mod N})"""
        shifted = np.roll(h, -s)
        return np.sum(h * (1.0 - shifted))
    
    def max_overlap(h):
        """Maximum overlap over all shifts 0..N-1"""
        h = np.clip(h, 0, 1)
        max_ov = 0.0
        for s in range(N):
            ov = overlap_at_shift(h, s)
            if ov > max_ov:
                max_ov = ov
        return max_ov
    
    # Use scipy to minimize max overlap
    # We'll use a smooth approximation: minimize sum of exp(t * overlap_s) / N for large t
    # This approximates the max
    
    best_h = h0.copy()
    best_score = max_overlap(h0)
    
    def objective(h):
        h_clipped = np.clip(h, 0, 1)
        # Sum constraint violation penalty
        sum_h = np.sum(h_clipped)
        penalty = 100.0 * (sum_h - N/2.0)**2
        
        # Smooth max of overlaps
        t = 20.0
        overlaps = np.array([overlap_at_shift(h_clipped, s) for s in range(N)])
        smooth_max = np.log(np.sum(np.exp(t * overlaps)) / N) / t
        
        return smooth_max + penalty
    
    def gradient(h):
        h_clipped = np.clip(h, 0, 1)
        sum_h = np.sum(h_clipped)
        # Gradient of penalty
        grad_penalty = 200.0 * (sum_h - N/2.0) * np.ones(N)
        
        # Gradient of smooth max
        t = 20.0
        overlaps = np.array([overlap_at_shift(h_clipped, s) for s in range(N)])
        exp_vals = np.exp(t * overlaps)
        weights = exp_vals / np.sum(exp_vals)
        
        grad_smooth = np.zeros(N)
        for s in range(N):
            shifted = np.roll(h_clipped, -s)
            # d/dh_i [h_i * (1 - h_{i+s})] = (1 - h_{i+s}) - h_i * delta_{i, i+s}
            # For shift s, the term is h_i(1-h_{(i+s)%N})
            # d/dh_j of sum_i h_i(1-h_{(i+s)%N}) = 
            #   for j: (1-h_{(j+s)%N}) - h_{(j-s)%N}
            grad_term = (1.0 - np.roll(h_clipped, -s)) - np.roll(h_clipped, s)
            grad_smooth += weights[s] * t * grad_term
        
        return grad_smooth + grad_penalty
    
    # Try L-BFGS-B with bounds
    bounds = [(0, 1)] * N
    
    try:
        result = minimize(
            objective, h0, jac=gradient, method='L-BFGS-B',
            bounds=bounds, options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6}
        )
        if result.success or result.fun < best_score:
            h_opt = np.clip(result.x, 0, 1)
            # Normalize to ensure sum = N/2 exactly
            current_sum = np.sum(h_opt)
            if current_sum > 0:
                h_opt = h_opt * (N/2.0) / current_sum
                h_opt = np.clip(h_opt, 0, 1)
            score = max_overlap(h_opt)
            if score < best_score:
                best_h = h_opt
                best_score = score
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline:
        return best_h
    
    # Try local search / refinement: binary-like solutions
    # Try alternating patterns with perturbations
    # Also try: h_i = 0.5 + 0.5*sin(2*pi*k*i/N) for various k
    
    for k in range(1, N//2 + 1):
        if time.time() > deadline:
            break
        h_k = 0.5 + 0.5 * np.sin(2 * np.pi * k * np.arange(N) / N)
        h_k = np.clip(h_k, 0, 1)
        score = max_overlap(h_k)
        if score < best_score:
            best_h = h_k.copy()
            best_score = score
    
    # Try cosine patterns too
    for k in range(1, N//2 + 1):
        if time.time() > deadline:
            break
        h_k = 0.5 + 0.5 * np.cos(2 * np.pi * k * np.arange(N) / N)
        h_k = np.clip(h_k, 0, 1)
        score = max_overlap(h_k)
        if score < best_score:
            best_h = h_k.copy()
            best_score = score
    
    # Final: ensure exact sum
    current_sum = np.sum(best_h)
    if abs(current_sum - N/2.0) > 1e-10:
        # Adjust by spreading the difference
        diff = N/2.0 - current_sum
        # Add diff/N to all, then clip and iterate
        for _ in range(10):
            best_h = best_h + diff / N
            best_h = np.clip(best_h, 0, 1)
            current_sum = np.sum(best_h)
            if abs(current_sum - N/2.0) < 1e-10:
                break
            diff = N/2.0 - current_sum
    
    return best_h
# EVOLVE-BLOCK-END
