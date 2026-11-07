import numpy as np
import time
from scipy.optimize import minimize

N = 50
w = 1.0 / 100.0  # spacing on [-0.5, 0.5] with m+2=102 points

def compute_value(h):
    """
    Standard computation of the objective value.
    h: array of N nonnegative step heights.
    """
    h = np.asarray(h, dtype=float)
    if not np.all(np.isfinite(h)) or np.any(h < 0):
        return 0.0
    g = np.convolve(h, h)
    m = len(g)  # 2*N - 1 = 99
    # y = [0, g[0], ..., g[m-1], 0]
    # l2 = sum_{i=0}^{m} (w/3) * (y[i]^2 + y[i]*y[i+1] + y[i+1]^2)
    # Let's derive a faster form for l2.
    # y[0]=0, y[m+1]=0, y[1..m] = g[0..m-1]
    # Sum = (w/3) * [ (y[0]^2 + y[0]y[1] + y[1]^2) + ... + (y[m]^2 + y[m]y[m+1] + y[m+1]^2) ]
    # = (w/3) * [ y[0]^2 + 2*sum(y[1..m]^2) + y[m+1]^2 + sum_{i=0}^{m} y[i]y[i+1] ]
    # Since y[0]=y[m+1]=0:
    # = (w/3) * [ 2*sum(g^2) + sum_{i=0}^{m-1} g[i]g[i+1] ]
    # Note: sum_{i=0}^{m} y[i]y[i+1] = y[0]y[1] + ... + y[m-1]y[m] + y[m]y[m+1]
    # = 0 + g[0]g[1] + ... + g[m-2]g[m-1] + 0
    # = sum_{i=0}^{m-2} g[i]g[i+1]
    
    g_sq_sum = np.sum(g**2)
    cross_sum = np.sum(g[:-1] * g[1:]) if m > 1 else 0.0
    l2 = (w / 3.0) * (2.0 * g_sq_sum + cross_sum)
    
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    if norm_inf == 0 or norm_1 == 0:
        return 0.0
    return l2 / (norm_1 * norm_inf)

def neg_value(log_h):
    h = np.exp(log_h)
    return -compute_value(h)

def neg_value_direct(h):
    return -compute_value(h)

def construct():
    deadline = time.time() + 145.0  # 5s safety margin
    best_h = np.ones(N)
    best_val = compute_value(best_h)
    
    rng = np.random.RandomState(42)

    def polish(start_h, maxiter=2000):
        """Run L-BFGS-B in log-space to polish a solution."""
        nonlocal best_h, best_val
        start_log_h = np.log(np.maximum(start_h, 1e-8))
        try:
            res = minimize(
                neg_value,
                start_log_h,
                method='L-BFGS-B',
                options={'maxiter': maxiter, 'maxfun': maxiter * 5, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            h_cand = np.exp(res.x)
            v_cand = compute_value(h_cand)
            if v_cand > best_val:
                best_val = v_cand
                best_h = h_cand.copy()
        except Exception:
            pass

    def polish_direct(start_h, maxiter=2000):
        """Run L-BFGS-B in direct space with bounds."""
        nonlocal best_h, best_val
        try:
            res = minimize(
                neg_value_direct,
                start_h,
                method='L-BFGS-B',
                bounds=[(0, None)] * N,
                options={'maxiter': maxiter, 'maxfun': maxiter * 5, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            h_cand = np.maximum(res.x, 0)
            v_cand = compute_value(h_cand)
            if v_cand > best_val:
                best_val = v_cand
                best_h = h_cand.copy()
        except Exception:
            pass

    # 1. Initial polish from uniform seed
    polish(best_h, maxiter=3000)
    
    # 2. Simulated Annealing with periodic polishing
    # Work in log-space for SA to ensure positivity
    current_log_h = np.log(best_h)
    current_val = best_val
    
    # SA Parameters
    T_start = 0.5
    T_end = 0.001
    alpha = 0.995  # cooling rate
    steps_per_temp = 50
    
    # We will run SA for a significant portion of the time, 
    # interspersed with L-BFGS-B polishing.
    
    # Define diverse starting points to re-seed SA if stuck or for initial multi-start
    starts = [
        np.zeros(N), # Uniform
        np.sin(np.pi * np.arange(N) / N), # Sine bump
        np.cos(2*np.pi*np.arange(N)/N), # Cosine
        1 - 2*np.abs(np.arange(N)/N - 0.5), # Triangle
        np.exp(-((np.arange(N)-N//2)**2)/20.0), # Gaussian peak
        np.where(np.arange(N) < N//2, 0.5, 1.5), # Step
    ]
    
    # Normalize starts to have similar scale
    for i, s in enumerate(starts):
        s = s - np.mean(s)
        if np.std(s) > 0:
            s = s / np.std(s) * 0.5
        starts[i] = s + 0.5 # Ensure positive-ish log values? 
        # Actually, let's just use them as perturbations to the current best or as new seeds
    
    # We will iterate through a few "phases". 
    # Phase 1: SA from the best found so far.
    # Phase 2: SA from diverse starts, then polish.
    
    phase = 0
    max_phases = 10
    
    while time.time() < deadline - 30.0:
        if phase == 0:
            # Continue from current best
            pass
        else:
            # Restart from a diverse point
            idx = (phase - 1) % len(starts)
            start_log_h = np.log(np.maximum(best_h, 1e-8)) + 0.5 * starts[idx]
            current_log_h = start_log_h
            current_val = compute_value(np.exp(current_log_h))
            if current_val < best_val * 0.5:
                 # If the restart is too bad, maybe just perturb the best instead
                 current_log_h = np.log(np.maximum(best_h, 1e-8)) + 0.2 * rng.randn(N)
                 current_val = compute_value(np.exp(current_log_h))
        
        T = T_start
        # Run SA for a fixed number of steps or until time is up
        sa_steps = 2000
        for k in range(sa_steps):
            if time.time() > deadline - 30.0:
                break
            
            # Perturb
            step_size = 0.1 * (1 + 0.5 * np.cos(np.pi * k / sa_steps)) # Decay step size
            new_log_h = current_log_h + step_size * rng.randn(N)
            new_h = np.exp(new_log_h)
            new_val = compute_value(new_h)
            
            # Acceptance criterion
            delta = new_val - current_val
            if delta > 0 or (T > 0 and np.log(rng.uniform()) < delta / T):
                current_log_h = new_log_h
                current_val = new_val
                if current_val > best_val:
                    best_val = current_val
                    best_h = new_h.copy()
            
            T *= alpha
            if T < T_end:
                break
        
        # Polish the best found in this SA phase
        if time.time() < deadline - 20.0:
            polish(best_h, maxiter=2000)
        
        phase += 1
        if phase > max_phases:
            break

    # 3. Final Polish
    if time.time() < deadline - 10.0:
        polish_direct(best_h, maxiter=3000)
        polish(best_h, maxiter=3000)
    
    return list(np.asarray(best_h, dtype=float))
