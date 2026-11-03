import numpy as np
import time
from scipy.optimize import minimize

N = 100
DEADLINE = 140.0  # safety margin before 150s kill

def construct():
    start_time = time.time()
    
    dw = 2.0 / N
    target = N / 2.0
    
    def compute_overlaps_vectorized(h):
        """Vectorized computation of all overlaps using broadcasting."""
        idx = np.arange(N)
        M = h[(idx[:, None] + idx[None, :]) % N]  # M[i,k] = h[(i+k)%N]
        mins = np.minimum(h[:, None], 1.0 - M)
        overlaps = np.sum(mins, axis=0) * dw
        return overlaps
    
    def objective(x):
        """Objective function for optimization: max overlap."""
        h = np.clip(x, 0, 1)
        # Project to satisfy sum constraint
        diff = target - h.sum()
        if abs(diff) > 1e-10:
            mask = (h > 1e-10) & (h < 1 - 1e-10)
            if mask.sum() > 0:
                h[mask] += diff / mask.sum()
                h = np.clip(h, 0, 1)
        overlaps = compute_overlaps_vectorized(h)
        return overlaps.max()
    
    def objective_grad(x):
        """Gradient of the objective (approximate via finite differences for speed)."""
        h = np.clip(x, 0, 1)
        diff = target - h.sum()
        if abs(diff) > 1e-10:
            mask = (h > 1e-10) & (h < 1 - 1e-10)
            if mask.sum() > 0:
                h[mask] += diff / mask.sum()
                h = np.clip(h, 0, 1)
        
        # Compute overlaps
        idx = np.arange(N)
        M = h[(idx[:, None] + idx[None, :]) % N]
        mins = np.minimum(h[:, None], 1.0 - M)
        overlaps = np.sum(mins, axis=0) * dw
        
        # Find the shift with maximum overlap
        k_max = np.argmax(overlaps)
        
        # Gradient with respect to h[i] for the max overlap shift
        # overlap_k = dw * sum_i min(h[i], 1 - h[(i+k)%N])
        # d/dh[i] min(h[i], 1-h[(i+k)%N]):
        #   if h[i] < 1-h[(i+k)%N]: derivative is 1
        #   if h[i] > 1-h[(i+k)%N]: derivative is 0
        #   if equal: subgradient, use 0.5
        
        h_shifted = np.roll(h, -k_max)
        grad = np.zeros(N)
        for i in range(N):
            a = h[i]
            b = 1.0 - h_shifted[i]
            if a < b:
                grad[i] = 1.0
            elif a > b:
                grad[i] = 0.0
            else:
                grad[i] = 0.5
        
        # Also need to account for h appearing in h[(i+k)%N] terms
        # For each i, h[i] appears in term j where (j+k)%N = i, i.e., j = (i-k)%N
        for i in range(N):
            j = (i - k_max) % N
            a = h[j]
            b = 1.0 - h[i]
            if a < b:
                # min(a,b) = a, derivative w.r.t. b is 0
                pass
            elif a > b:
                # min(a,b) = b = 1-h[i], derivative w.r.t. h[i] is -1
                grad[i] += -1.0
            else:
                grad[i] += -0.5
        
        grad *= dw
        
        # Add projection gradient (approximate)
        diff = target - np.clip(x, 0, 1).sum()
        if abs(diff) > 1e-10:
            mask = (np.clip(x, 0, 1) > 1e-10) & (np.clip(x, 0, 1) < 1 - 1e-10)
            if mask.sum() > 0:
                grad[mask] += diff / mask.sum()
        
        return grad
    
    # Start with uniform solution
    best_h = np.full(N, 0.5)
    best_score = compute_overlaps_vectorized(best_h).max()
    
    # Try alternating patterns with optimized amplitude
    # h[i] = 0.5 + A * (-1)^i
    for A in np.linspace(0, 0.5, 50):
        if time.time() > DEADLINE:
            break
        h = 0.5 + A * (-1) ** np.arange(N)
        h = np.clip(h, 0, 1)
        # Adjust to satisfy sum constraint
        diff = target - h.sum()
        if abs(diff) > 1e-10:
            mask = (h > 1e-10) & (h < 1 - 1e-10)
            if mask.sum() > 0:
                h[mask] += diff / mask.sum()
                h = np.clip(h, 0, 1)
        score = compute_overlaps_vectorized(h).max()
        if score < best_score:
            best_score = score
            best_h = h.copy()
    
    # Try various periodic patterns
    for period in [2, 3, 4, 5, 7, 10]:
        if time.time() > DEADLINE:
            break
        for A in np.linspace(0.1, 0.5, 20):
            if time.time() > DEADLINE:
                break
            # Create periodic pattern
            base = np.zeros(period)
            base[:period//2] = 1.0
            h = np.tile(base, N // period + 1)[:N]
            # Scale to have mean 0.5
            h = 0.5 * h + 0.5 * A
            h = np.clip(h, 0, 1)
            diff = target - h.sum()
            if abs(diff) > 1e-10:
                mask = (h > 1e-10) & (h < 1 - 1e-10)
                if mask.sum() > 0:
                    h[mask] += diff / mask.sum()
                    h = np.clip(h, 0, 1)
            score = compute_overlaps_vectorized(h).max()
            if score < best_score:
                best_score = score
                best_h = h.copy()
    
    # Use Nelder-Mead optimization on the best candidate
    if time.time() < DEADLINE - 30:
        x0 = best_h.copy()
        # Perturb slightly to escape local minima
        rng = np.random.default_rng(42)
        for trial in range(5):
            if time.time() > DEADLINE - 10:
                break
            x0_trial = x0 + rng.normal(0, 0.05, N)
            x0_trial = np.clip(x0_trial, 0, 1)
            
            result = minimize(objective, x0_trial, method='Nelder-Mead', 
                             options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-8})
            
            h_opt = np.clip(result.x, 0, 1)
            diff = target - h_opt.sum()
            if abs(diff) > 1e-10:
                mask = (h_opt > 1e-10) & (h_opt < 1 - 1e-10)
                if mask.sum() > 0:
                    h_opt[mask] += diff / mask.sum()
                    h_opt = np.clip(h_opt, 0, 1)
            
            score = compute_overlaps_vectorized(h_opt).max()
            if score < best_score:
                best_score = score
                best_h = h_opt.copy()
    
    # Coordinate descent refinement
    if time.time() < DEADLINE - 10:
        for _ in range(200):
            if time.time() > DEADLINE - 5:
                break
            improved = False
            for i in range(N):
                # Try varying h[i] while keeping sum constant
                # Adjust another element to maintain sum
                for j in range(N):
                    if i == j:
                        continue
                    # Try moving delta from j to i
                    for delta in [0.01, 0.05, 0.1]:
                        h_trial = best_h.copy()
                        h_trial[i] += delta
                        h_trial[j] -= delta
                        h_trial = np.clip(h_trial, 0, 1)
                        if h_trial.sum() != target:
                            # Re-project
                            diff = target - h_trial.sum()
                            mask = (h_trial > 1e-10) & (h_trial < 1 - 1e-10)
                            if mask.sum() > 0:
                                h_trial[mask] += diff / mask.sum()
                                h_trial = np.clip(h_trial, 0, 1)
                        score = compute_overlaps_vectorized(h_trial).max()
                        if score < best_score - 1e-10:
                            best_score = score
                            best_h = h_trial.copy()
                            improved = True
                            break
                    if improved:
                        break
            if not improved:
                break
    
    # Final projection to ensure exact sum
    diff = target - best_h.sum()
    if abs(diff) > 1e-12:
        mask = (best_h > 1e-15) & (best_h < 1 - 1e-15)
        if mask.sum() > 0:
            best_h[mask] += diff / mask.sum()
            best_h = np.clip(best_h, 0, 1)
    
    return best_h
