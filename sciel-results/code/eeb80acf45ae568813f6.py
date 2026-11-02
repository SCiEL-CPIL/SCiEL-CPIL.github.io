import numpy as np
import time

N = 100
W = 2.0 / N

def _overlap_max(h):
    """Compute the maximum overlap of h with shifted (1-h)."""
    # overlap_k = W * sum_j h[j] * (1 - h[(j+k)%N])
    # = W * (sum_j h[j] - sum_j h[j]*h[(j+k)%N])
    # = W * (N/2 - autocorr[k])
    H = np.fft.fft(h)
    autocorr = np.fft.ifft(H * np.conj(H)).real
    overlaps = (N / 2.0 - autocorr) * W
    return np.max(overlaps)

def _local_search(h, n_iterations=2000, deadline=None):
    """Local search: try flipping or adjusting individual elements."""
    best = h.copy()
    best_val = _overlap_max(best)
    
    for i in range(n_iterations):
        if deadline is not None and time.time() > deadline:
            break
        
        # Try a random perturbation
        idx = np.random.randint(N)
        old_val = best[idx]
        
        # Try several candidate values
        candidates = [0.0, 1.0, 0.25, 0.5, 0.75, 
                      max(0.0, old_val - 0.1), min(1.0, old_val + 0.1)]
        candidates = [c for c in candidates if abs(c - old_val) > 1e-6]
        
        if not candidates:
            continue
            
        for c in candidates:
            trial = best.copy()
            trial[idx] = c
            
            # Project to maintain sum = N/2
            diff = N / 2.0 - np.sum(trial)
            if abs(diff) > 1e-10:
                # Distribute the difference
                trial = trial + diff / N
                trial = np.clip(trial, 0, 1)
                # Re-project with bisection
                lo, hi = trial.min() - 2.0, trial.max() + 2.0
                for _ in range(30):
                    mid = (lo + hi) / 2.0
                    s = np.sum(np.clip(trial - mid, 0.0, 1.0))
                    if s > N / 2.0:
                        lo = mid
                    else:
                        hi = mid
                tau = (lo + hi) / 2.0
                trial = np.clip(trial - tau, 0.0, 1.0)
            
            val = _overlap_max(trial)
            if val < best_val - 1e-12:
                best = trial
                best_val = val
                break
    
    return best, best_val

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    best_h = None
    best_val = np.inf
    
    # Multiple restarts with different initializations
    n_restarts = 50
    
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Different initialization strategies
        if restart % 3 == 0:
            # Random binary-like initialization
            h = np.random.choice([0.0, 1.0], size=N)
        elif restart % 3 == 1:
            # Random continuous initialization
            h = np.random.uniform(0, 1, size=N)
        else:
            # Structured initialization: alternating patterns
            pattern = np.zeros(N)
            period = np.random.randint(2, N // 2 + 1)
            fill = np.random.uniform(0.3, 0.7)
            for j in range(N):
                if (j // period) % 2 == 0:
                    pattern[j] = fill
                else:
                    pattern[j] = 1.0 - fill
            h = pattern + np.random.normal(0, 0.05, N)
            h = np.clip(h, 0, 1)
        
        # Project to feasible set
        diff = N / 2.0 - np.sum(h)
        h = h + diff / N
        h = np.clip(h, 0, 1)
        
        # Bisection projection
        lo, hi = h.min() - 2.0, h.max() + 2.0
        for _ in range(40):
            mid = (lo + hi) / 2.0
            s = np.sum(np.clip(h - mid, 0.0, 1.0))
            if s > N / 2.0:
                lo = mid
            else:
                hi = mid
        tau = (lo + hi) / 2.0
        h = np.clip(h - tau, 0.0, 1.0)
        
        # Simulated annealing
        T = 0.5  # Initial temperature
        cooling = 0.95
        current_val = _overlap_max(h)
        
        sa_steps = 500
        for step in range(sa_steps):
            if time.time() > deadline:
                break
            
            # Random perturbation
            idx = np.random.randint(N)
            old_val = h[idx]
            new_val = np.clip(old_val + np.random.normal(0, 0.2), 0, 1)
            
            trial = h.copy()
            trial[idx] = new_val
            
            # Project to maintain sum
            diff = N / 2.0 - np.sum(trial)
            trial = trial + diff / N
            trial = np.clip(trial, 0, 1)
            
            # Bisection projection
            lo, hi = trial.min() - 2.0, trial.max() + 2.0
            for _ in range(25):
                mid = (lo + hi) / 2.0
                s = np.sum(np.clip(trial - mid, 0.0, 1.0))
                if s > N / 2.0:
                    lo = mid
                else:
                    hi = mid
            tau = (lo + hi) / 2.0
            trial = np.clip(trial - tau, 0.0, 1.0)
            
            new_val_obj = _overlap_max(trial)
            delta = new_val_obj - current_val
            
            # Accept if better, or with probability exp(-delta/T)
            if delta < 0 or np.random.random() < np.exp(-delta / max(T, 1e-10)):
                h = trial
                current_val = new_val_obj
            
            T *= cooling
        
        # Local search refinement
        h, val = _local_search(h, n_iterations=500, deadline=deadline)
        
        if val < best_val:
            best_val = val
            best_h = h.copy()
        
        # If we've found a very good solution, do more refinement
        if best_val < 0.35 and time.time() < deadline - 30:
            h, val = _local_search(best_h, n_iterations=2000, deadline=deadline)
            if val < best_val:
                best_val = val
                best_h = h.copy()
    
    # Final projection to ensure exact sum
    if best_h is not None:
        diff = N / 2.0 - np.sum(best_h)
        best_h = best_h + diff / N
        best_h = np.clip(best_h, 0, 1)
        
        # One final bisection projection
        lo, hi = best_h.min() - 2.0, best_h.max() + 2.0
        for _ in range(50):
            mid = (lo + hi) / 2.0
            s = np.sum(np.clip(best_h - mid, 0.0, 1.0))
            if s > N / 2.0:
                lo = mid
            else:
                hi = mid
        tau = (lo + hi) / 2.0
        best_h = np.clip(best_h - tau, 0.0, 1.0)
    
    return best_h
