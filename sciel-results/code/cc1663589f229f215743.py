import numpy as np
import time
from scipy.optimize import differential_evolution, minimize
from functools import partial

N = 14
D = 3
NUM_PAIRS = N * (N - 1) // 2

def _pairwise_sq_dists(points):
    """
    Returns a flat array of squared pairwise distances for N points.
    """
    # points shape (N, D)
    # diff shape (N, N, D)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    sq_dists = np.sum(diff**2, axis=2)
    # Extract upper triangle
    iu = np.triu_indices(N, k=1)
    return sq_dists[iu]

def _raw_score(points):
    """
    Calculate the actual score: (max_dist / min_dist)^2.
    Returns inf if degenerate.
    """
    d2 = _pairwise_sq_dists(points)
    d_min = np.min(d2)
    d_max = np.max(d2)
    if d_min < 1e-12:
        return 1e10
    return (d_max / d_min)

def _smooth_objective(x, tau):
    """
    Smooth objective: minimizes log((max_d/min_d)^2) = 2*(soft_max(log d2) - soft_min(log d2)).
    Uses numerically stable log-sum-exp.
    """
    X = x.reshape(N, D)
    d2 = _pairwise_sq_dists(X)
    
    # Ensure positive values for log
    d2_safe = np.maximum(d2, 1e-15)
    log_d2 = np.log(d2_safe)
    
    if tau <= 0:
        tau = 1e-10
        
    # Soft max
    max_log = np.max(log_d2)
    term_max = tau * (log_d2 - max_log)
    term_max = np.clip(term_max, -50, 50)
    sum_exp_max = np.sum(np.exp(term_max))
    soft_max_val = max_log + (1.0 / tau) * np.log(sum_exp_max)
    
    # Soft min
    min_log = np.min(log_d2)
    term_min = -tau * (log_d2 - min_log)
    term_min = np.clip(term_min, -50, 50)
    sum_exp_min = np.sum(np.exp(term_min))
    soft_min_val = min_log - (1.0 / tau) * np.log(sum_exp_min)
    
    log_ratio = soft_max_val - soft_min_val
    return 2.0 * log_ratio

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill
    
    best_score = 1e10
    best_X = None
    
    def _evaluate_and_update(x_flat):
        nonlocal best_score, best_X
        if x_flat is None:
            return
        X = x_flat.reshape(N, D)
        # Check for NaN/Inf
        if not np.all(np.isfinite(X)):
            return
        score = _raw_score(X)
        if score < best_score:
            best_score = score
            best_X = X.copy()

    # 1. Differential Evolution for Global Search
    # The objective is scale invariant, so we can search in a bounded box.
    # A box of [-2, 2] should be sufficient for the optimal configuration.
    bounds = [(-2.0, 2.0)] * (N * D)
    
    # Use a moderate tau for DE to keep the landscape smooth enough for DE's sampling
    # but sharp enough to guide towards the minimum.
    tau_de = 50.0
    obj_de = partial(_smooth_objective, tau=tau_de)
    
    try:
        # popsize=15 is default, maxiter=100 is reasonable.
        # tol and atol control convergence.
        res_de = differential_evolution(
            obj_de,
            bounds,
            maxiter=100,
            popsize=20,
            tol=1e-6,
            seed=42,
            polish=False  # We will do our own polish
        )
        if res_de.x is not None:
            _evaluate_and_update(res_de.x)
    except Exception:
        pass
    
    # 2. Diverse Initial Points for Local Search
    # Even if DE found a good point, local search from diverse starts can sometimes find better local minima
    # or refine the DE result if it got stuck in a suboptimal basin.
    
    inits = []
    
    # Add the DE result as an initial point if it's valid
    if res_de.x is not None and np.all(np.isfinite(res_de.x)):
        inits.append(res_de.x.copy())
    
    # Fibonacci sphere
    golden_angle = np.pi * (3. - np.sqrt(5.))
    X_fib = np.zeros((N, 3))
    for i in range(N):
        z = 1. - (i / (N - 1.)) * 2.
        radius = np.sqrt(max(0, 1 - z**2))
        theta = golden_angle * i
        x = np.cos(theta) * radius
        y = np.sin(theta) * radius
        X_fib[i] = [x, y, z]
    inits.append(X_fib.flatten())
    
    # Random points on sphere
    np.random.seed(123)
    for _ in range(5):
        pts = np.random.randn(N, 3)
        norms = np.linalg.norm(pts, axis=1)
        pts /= norms[:, np.newaxis]
        inits.append(pts.flatten())
        
    # Random points in ball
    for _ in range(5):
        pts = np.random.randn(N, 3)
        max_norm = np.max(np.linalg.norm(pts, axis=1))
        if max_norm > 0:
            pts /= max_norm
        inits.append(pts.flatten())

    # 3. Local Refinement Pipeline
    # For each initial point, run a multi-stage annealing optimization.
    
    tau_schedule = [10, 50, 200, 1000]
    
    for x_init in inits:
        if time.time() > deadline - 10:
            break
            
        x_current = x_init.copy()
        
        # Center the points to help with conditioning? 
        # The objective is translation invariant. Centering can help gradient methods.
        X_c = x_current.reshape(N, D)
        x_current = (X_c - np.mean(X_c, axis=0)).flatten()
        
        # Stage 1-4: L-BFGS-B with increasing tau
        for tau in tau_schedule:
            if time.time() > deadline - 5:
                break
                
            try:
                obj = partial(_smooth_objective, tau=tau)
                res = minimize(
                    obj,
                    x_current,
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                if res.x is not None:
                    x_current = res.x
            except Exception:
                pass
        
        # Evaluate after L-BFGS-B
        _evaluate_and_update(x_current)
        
        # Stage 5: Nelder-Mead on raw objective for final polish
        # NM is robust to non-smoothness and can handle the discrete max/min switch.
        if time.time() < deadline - 2:
            try:
                res_nm = minimize(
                    lambda x: _raw_score(x.reshape(N, D)),
                    x_current,
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
                )
                if res_nm.x is not None:
                    _evaluate_and_update(res_nm.x)
            except Exception:
                pass

    if best_X is None:
        # Fallback
        return np.zeros((N, D))
        
    return best_X
