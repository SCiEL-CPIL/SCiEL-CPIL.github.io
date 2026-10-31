"""Optimized program for Erdos minimum overlap task."""
import numpy as np
from scipy.optimize import minimize
import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140  # 140s deadline, 20s safety margin
    
    # Start with a good initial guess: alternating pattern to anti-correlate h with h(x+1)
    # h(x) high where h(x+1) is high, low where h(x+1) is low
    # Try a smooth transition pattern
    x = np.linspace(0, 2, N, endpoint=False)
    
    # Initial guess: use a cosine wave with period 1 (so h(x) and h(x+1) are in phase)
    h0 = 0.5 + 0.5 * np.cos(2 * np.pi * x)
    h0 = np.clip(h0, 0, 1)
    # Normalize so sum = N/2
    h0 = h0 / h0.sum() * (N / 2)
    h0 = np.clip(h0, 0, 1)
    
    best_h = h0.copy()
    best_score = float('inf')
    
    def compute_overlap(h):
        """Compute maximum of h(x) * (1 - h(x+1)) over all shifts."""
        # h is defined on [0, 2] with N points
        # Shift by 1 unit = N/2 steps
        shift = N // 2
        h_shifted = np.roll(h, -shift)
        overlap = h * (1 - h_shifted)
        return np.max(overlap)
    
    current_score = compute_overlap(best_h)
    best_score = current_score
    
    # Use L-BFGS-B optimization to minimize the max overlap
    def objective(h_flat):
        h = np.clip(h_flat, 0, 1)
        return compute_overlap(h)
    
    # Try multiple random initializations and local optimizations
    n_restarts = 30
    rng = np.random.RandomState(42)
    
    for i in range(n_restarts):
        if time.time() > deadline - 5:
            break
            
        if i == 0:
            h_init = best_h.copy()
        else:
            # Random smooth initialization
            freq = rng.choice([1, 2, 3])
            phase = rng.uniform(0, 2 * np.pi)
            h_init = 0.5 + 0.4 * np.cos(2 * np.pi * freq * x + phase)
            h_init = np.clip(h_init, 0, 1)
            h_init = h_init / h_init.sum() * (N / 2)
            h_init = np.clip(h_init, 0, 1)
        
        try:
            result = minimize(
                objective,
                h_init,
                method='L-BFGS-B',
                bounds=[(0, 1)] * N,
                options={'maxiter': 100, 'ftol': 1e-10}
            )
            
            h_opt = np.clip(result.x, 0, 1)
            # Ensure sum constraint is approximately satisfied
            # The constraint sum = N/2 is important
            # Adjust if needed
            if abs(h_opt.sum() - N/2) > 1e-6:
                h_opt = h_opt + (N/2 - h_opt.sum()) / N
                h_opt = np.clip(h_opt, 0, 1)
            
            score = compute_overlap(h_opt)
            if score < best_score:
                best_score = score
                best_h = h_opt.copy()
                
        except Exception:
            continue
    
    # Final check and return
    final_score = compute_overlap(best_h)
    # Ensure the sum is exactly N/2 by adjusting slightly
    diff = N/2 - best_h.sum()
    if abs(diff) > 1e-10:
        # Distribute the difference evenly
        best_h = best_h + diff / N
        best_h = np.clip(best_h, 0, 1)
        # Re-normalize if clipping changed the sum
        if abs(best_h.sum() - N/2) > 1e-10:
            best_h = best_h / best_h.sum() * (N/2)
            best_h = np.clip(best_h, 0, 1)
    
    return best_h
# EVOLVE-BLOCK-END
