import numpy as np
from scipy.optimize import minimize
import time

N = 100

def objective(h):
    corr = np.correlate(h, 1.0 - h, mode='full')
    return np.max(corr)

def project_to_feasible(h):
    """Project h to satisfy 0<=h<=1 and sum(h)=N/2."""
    h = np.clip(h, 0.0, 1.0)
    for _ in range(100):
        diff = N/2.0 - np.sum(h)
        if abs(diff) < 1e-12:
            break
        # Add diff to elements not at bounds
        mask = (h > 1e-9) & (h < 1.0 - 1e-9)
        if np.sum(mask) == 0:
            # All at bounds, can't fix
            break
        h[mask] += diff / np.sum(mask)
        h = np.clip(h, 0.0, 1.0)
    return h

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_h = np.full(N, 0.5)
    best_val = objective(best_h)
    
    bounds = [(0.0, 1.0) for _ in range(N)]
    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - N / 2.0, 'jac': lambda x: np.ones(N)}
    ]
    
    # Multiple starting points
    rng = np.random.RandomState(42)
    starts = [np.full(N, 0.5)]
    
    # Random perturbations
    for _ in range(5):
        h_pert = np.full(N, 0.5) + rng.randn(N) * 0.1
        starts.append(project_to_feasible(h_pert))
    
    # Structured starts
    # Sine waves with different frequencies
    for freq in [1, 2, 3]:
        h_sine = 0.5 + 0.4 * np.sin(2 * np.pi * freq * np.arange(N) / N)
        starts.append(project_to_feasible(h_sine))
    
    # Linear ramp
    h_ramp = np.linspace(0.0, 1.0, N)
    starts.append(project_to_feasible(h_ramp))
    
    # Two-level: first 50 are 1, rest 0
    h_bin = np.zeros(N)
    h_bin[:50] = 1.0
    starts.append(h_bin.copy())
    
    # Alternating 0.25 and 0.75
    h_alt = np.where(np.arange(N) % 2 == 0, 0.25, 0.75)
    starts.append(h_alt.copy())
    
    for h0 in starts:
        if time.time() > deadline - 10:
            break
        
        try:
            res = minimize(
                objective,
                h0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 800, 'ftol': 1e-10, 'disp': False}
            )
            
            h_res = project_to_feasible(res.x)
            val = objective(h_res)
            if val < best_val:
                best_val = val
                best_h = h_res.copy()
                
        except Exception:
            pass
    
    # Final projection
    best_h = project_to_feasible(best_h)
    
    return best_h
