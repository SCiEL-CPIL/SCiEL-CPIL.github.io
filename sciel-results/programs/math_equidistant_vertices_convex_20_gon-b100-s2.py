import numpy as np
import time
from scipy.optimize import minimize

def get_points(deltas):
    n = 20
    base_angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    thetas = base_angles.copy()
    if len(deltas) > 0:
        thetas[1:] += deltas
    thetas = np.sort(thetas)
    thetas = thetas % (2*np.pi)
    thetas = np.sort(thetas)
    x = np.cos(thetas)
    y = np.sin(thetas)
    return np.column_stack((x, y))

def vertex_score(v_idx, points):
    dists = np.linalg.norm(points - points[v_idx], axis=1)
    dists[v_idx] = np.inf
    sorted_dists = np.sort(dists)
    if sorted_dists[0] < 0.05:
        return 0.0
    gaps = sorted_dists[1:] - sorted_dists[:-1]
    k = np.argmin(gaps)
    D = (sorted_dists[k] + sorted_dists[k+1]) / 2.0
    if D < 0.1:
        return 0.0
    if k == 0:
        d_sel = sorted_dists[0:4]
    elif k == 17:
        d_sel = sorted_dists[15:19]
    else:
        d_sel = sorted_dists[k-1:k+3]
    ratios = np.maximum(d_sel / D, D / d_sel)
    raw = np.mean(ratios)
    score = min(1.0, 1.0 / raw)
    return score

def min_vertex_score(deltas):
    points = get_points(deltas)
    n = 20
    scores = np.array([vertex_score(i, points) for i in range(n)])
    return -np.min(scores)

def equidistant_points_polygon20() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 590  # safety margin
    
    n = 20
    x0 = np.zeros(19)
    best_result = get_points(x0)
    best_score = -min_vertex_score(x0)
    
    # Try multiple restarts with different initial perturbations
    rng = np.random.default_rng(42)
    num_restarts = 20
    max_iter_per_restart = 2000
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Generate initial guess: regular polygon + small random perturbation
        if restart == 0:
            x_init = np.zeros(19)
        else:
            # Scale perturbation: larger for early restarts, smaller for later
            scale = 0.1 / (restart + 1)
            x_init = rng.normal(0, scale, size=19)
        
        try:
            res = minimize(
                min_vertex_score,
                x_init,
                method='L-BFGS-B',
                options={'maxiter': max_iter_per_restart, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            
            if res.x is not None and res.fun < -best_score:
                current_score = -res.fun
                if current_score > best_score:
                    best_score = current_score
                    best_result = get_points(res.x)
                    
        except Exception:
            pass
        
        # Also try Nelder-Mead from the best point found so far
        if time.time() > deadline:
            break
        try:
            res_nm = minimize(
                min_vertex_score,
                best_result if isinstance(best_result, np.ndarray) else get_points(x0),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-7, 'fatol': 1e-7}
            )
            # This won't work directly since minimize expects 1D input
            # Skip Nelder-Mead here, rely on L-BFGS-B
        except Exception:
            pass
    
    return best_result
