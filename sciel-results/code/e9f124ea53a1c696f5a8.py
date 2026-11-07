"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _correlation_value(h):
    """Compute the evaluator's value: max(corr) / N * 2"""
    corr = np.correlate(h, 1 - h, mode="full")
    return np.max(corr) / N * 2

def _project_to_feasible(h):
    """Project h to [0,1] and sum=N/2 using bisection on a global shift"""
    target = N / 2.0
    
    lo = -np.max(h) - 1.0
    hi = 1.0 - np.min(h) + 1.0
    
    for _ in range(60):
        mid = (lo + hi) / 2.0
        val = np.sum(np.clip(h + mid, 0.0, 1.0)) - target
        if val > 0:
            hi = mid
        else:
            lo = mid
            
    s = (lo + hi) / 2.0
    h_proj = np.clip(h + s, 0.0, 1.0)
    
    diff = np.sum(h_proj) - target
    if abs(diff) > 1e-14:
        if diff > 0:
            idx = np.argsort(h_proj)[::-1]
        else:
            idx = np.argsort(h_proj)
        for i in idx:
            if h_proj[i] > 0.001 or diff < 0:
                if diff > 0:
                    h_proj[i] = max(0.0, h_proj[i] - diff)
                    diff = 0.0
                else:
                    h_proj[i] = min(1.0, h_proj[i] - diff)
                    diff = 0.0
    
    return h_proj

def _soft_max(x, beta=100.0):
    """Numerically stable soft-max (log-sum-exp / beta)."""
    x = np.asarray(x, dtype=float)
    x_max = np.max(x)
    if not np.isfinite(x_max):
        return 0.0
    return (np.log(np.sum(np.exp(x - x_max))) + x_max) / beta

def construct():
    """
    Construct 100 step heights h in [0,1] with sum(h)=50,
    minimizing max(np.correlate(h, 1-h, mode='full')) / N * 2.
    
    Strategy:
    - Use a smooth surrogate (soft-max via log-sum-exp) for the max correlation objective.
    - Use SLSQP with bounds and equality constraint for sum(h)=50.
    - Try multiple starting points to avoid local minima.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin
    
    def objective(h, beta=100.0):
        """Smooth objective using soft-max of correlation values."""
        h = np.asarray(h, dtype=float)
        if not np.all(np.isfinite(h)):
            return 1e10
        corr = np.correlate(h, 1 - h, mode="full")
        return _soft_max(corr, beta) / N * 2
    
    def constraints(h):
        """Sum constraint: sum(h) = 50."""
        return np.sum(h) - 50.0
    
    bounds = [(0.0, 1.0)] * N
    cons = [{'type': 'eq', 'fun': constraints}]
    
    best_h = None
    best_val = np.inf
    
    # Multiple starting points
    start_points = []
    
    # 1. Uniform
    start_points.append(np.full(N, 0.5))
    
    # 2. Alternating 0.3, 0.7
    h_alt = np.zeros(N)
    for i in range(N):
        h_alt[i] = 0.3 if i % 2 == 0 else 0.7
    start_points.append(h_alt)
    
    # 3. Sinusoidal patterns
    for f in [0.1, 0.2, 0.5, 1.0, 2.0]:
        h_sin = 0.5 + 0.4 * np.sin(2 * np.pi * f * np.arange(N))
        h_sin = np.clip(h_sin, 0.0, 1.0)
        h_sin = _project_to_feasible(h_sin)
        start_points.append(h_sin)
    
    # 4. Random starts
    np.random.seed(42)
    for _ in range(5):
        h_rand = np.random.uniform(0.0, 1.0, N)
        h_rand = _project_to_feasible(h_rand)
        start_points.append(h_rand)
    
    # Optimize from each starting point
    for sp in start_points:
        if time.time() > deadline - 10.0:
            break
        
        try:
            res = minimize(
                objective,
                sp,
                method='SLSQP',
                bounds=bounds,
                constraints=cons,
                options={'maxiter': 1000, 'ftol': 1e-12}
            )
            
            if res.success or res.fun < best_val:
                h_opt = np.clip(res.x, 0.0, 1.0)
                h_opt = _project_to_feasible(h_opt)
                val = _correlation_value(h_opt)
                
                if val < best_val:
                    best_val = val
                    best_h = h_opt.copy()
        except Exception:
            continue
    
    # If we have a good result, try to polish with higher beta
    if best_h is not None and time.time() < deadline - 5.0:
        for beta in [200.0, 500.0]:
            if time.time() > deadline - 5.0:
                break
            
            def objective_high_beta(h, beta=beta):
                h = np.asarray(h, dtype=float)
                if not np.all(np.isfinite(h)):
                    return 1e10
                corr = np.correlate(h, 1 - h, mode="full")
                return _soft_max(corr, beta) / N * 2
            
            try:
                res = minimize(
                    objective_high_beta,
                    best_h,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=cons,
                    options={'maxiter': 500, 'ftol': 1e-14}
                )
                
                if res.success or res.fun < _correlation_value(best_h):
                    h_opt = np.clip(res.x, 0.0, 1.0)
                    h_opt = _project_to_feasible(h_opt)
                    val = _correlation_value(h_opt)
                    
                    if val < best_val:
                        best_val = val
                        best_h = h_opt.copy()
            except Exception:
                continue
    
    # Final projection to ensure constraints are met
    if best_h is not None:
        best_h = _project_to_feasible(best_h)
    else:
        # Fallback
        best_h = np.full(N, 0.5)
    
    return best_h
# EVOLVE-BLOCK-END
