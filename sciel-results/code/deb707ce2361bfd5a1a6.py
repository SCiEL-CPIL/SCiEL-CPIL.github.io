import numpy as np
import time
from scipy.optimize import minimize

def get_points(deltas):
    """
    Reconstruct points from angle deviations.
    deltas shape (19,) -> angles shape (20,)
    theta_0 is fixed at 0.
    theta_i = 2*pi*i/20 + deltas[i-1] for i=1..19
    """
    n = 20
    base_angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    thetas = base_angles.copy()
    if len(deltas) > 0:
        thetas[1:] += deltas
    # Ensure sorted to maintain convexity on circle (if deltas are large)
    # Sorting changes vertex indices, but score is min over all vertices, so invariant.
    thetas = np.sort(thetas)
    # Shift to [0, 2pi)
    thetas = thetas % (2*np.pi)
    # Re-sort after mod
    thetas = np.sort(thetas)
    
    x = np.cos(thetas)
    y = np.sin(thetas)
    return np.column_stack((x, y))

def vertex_score(v_idx, points):
    """
    Calculate score for vertex v_idx.
    """
    # Distances to all other points
    dists = np.linalg.norm(points - points[v_idx], axis=1)
    dists[v_idx] = np.inf # Exclude self
    
    # Sort 19 distances
    sorted_dists = np.sort(dists)
    
    # Check for repeated points (dist < 0.05)
    if sorted_dists[0] < 0.05:
        return 0.0
        
    # Find k: first index minimizing d_{k+1} - d_k
    # d_0...d_18
    gaps = sorted_dists[1:] - sorted_dists[:-1]
    # gaps length 18
    min_gap = np.min(gaps)
    # "first index minimizing"
    # If multiple, take the first one.
    k = np.argmin(gaps)
    
    # D = (d_k + d_{k+1}) / 2
    D = (sorted_dists[k] + sorted_dists[k+1]) / 2.0
    
    if D < 0.1:
        return 0.0
        
    # Select 4 distances
    if k == 0:
        d_sel = sorted_dists[0:4]
    elif k == 17:
        d_sel = sorted_dists[15:19] # d_15..d_18
    else:
        d_sel = sorted_dists[k-1:k+3] # d_{k-1}..d_{k+2}
        
    # raw = mean of max(d/D, D/d)
    ratios = np.maximum(d_sel / D, D / d_sel)
    raw = np.mean(ratios)
    
    score = min(1.0, 1.0 / raw)
    return score

def min_vertex_score(deltas):
    """
    Objective function: negative of the minimum vertex score.
    """
    points = get_points(deltas)
    n = 20
    scores = np.array([vertex_score(i, points) for i in range(n)])
    min_score = np.min(scores)
    return -min_score

def equidistant_points_polygon20() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 590  # safety margin for 600s limit
    
    n = 20
    x0 = np.zeros(19)
    
    best_result = get_points(x0)
    best_score = -min_vertex_score(x0)
    
    # Strategy: Combine L-BFGS-B (good for smooth landscapes) with Nelder-Mead (robust).
    # Inspiration 1 used multiple restarts with L-BFGS-B.
    # Current program used single Nelder-Mead.
    # We will use a multi-restart strategy with L-BFGS-B, then refine the best with Nelder-Mead.
    
    rng = np.random.default_rng(42)
    
    # Phase 1: Global search with L-BFGS-B from multiple random starts
    # We try a few scales of perturbation.
    scales = [0.01, 0.05, 0.1, 0.2, 0.5]
    
    for scale in scales:
        if time.time() > deadline:
            break
        
        # Try a few random directions at this scale
        for _ in range(5):
            if time.time() > deadline:
                break
            
            x_init = rng.normal(0, scale, size=19)
            
            try:
                res = minimize(
                    min_vertex_score, 
                    x_init, 
                    method='L-BFGS-B',
                    options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                
                if res.x is not None:
                    current_score = -res.fun
                    if current_score > best_score:
                        best_score = current_score
                        best_result = get_points(res.x)
                        
            except Exception:
                pass
    
    # Phase 2: Local refinement with Nelder-Mead from the best point found
    # Convert best_result back to deltas? 
    # Actually, we can just optimize from the last successful x_init or res.x.
    # But we lost the exact res.x that gave best_score if it wasn't the last one.
    # Let's keep track of the best x_deltas.
    
    # Re-run to capture best_x_deltas properly or just refine the current best_result?
    # Since get_points is deterministic from deltas, and we have best_result, 
    # we can try to optimize in the neighborhood of the regular polygon again with NM, 
    # or better, keep track of best_x.
    
    # Let's refactor slightly to keep best_x
    best_x = x0
    
    # Re-do the loop keeping track of best_x
    best_score = -min_vertex_score(x0)
    best_x = x0
    best_result = get_points(x0)
    
    for scale in scales:
        if time.time() > deadline:
            break
        
        for _ in range(5):
            if time.time() > deadline:
                break
            
            x_init = rng.normal(0, scale, size=19)
            
            try:
                res = minimize(
                    min_vertex_score, 
                    x_init, 
                    method='L-BFGS-B',
                    options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                
                if res.x is not None:
                    current_score = -res.fun
                    if current_score > best_score:
                        best_score = current_score
                        best_x = res.x
                        best_result = get_points(res.x)
                        
            except Exception:
                pass
    
    # Final refinement with Nelder-Mead from best_x
    if time.time() < deadline:
        try:
            res_nm = minimize(
                min_vertex_score, 
                best_x, 
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            
            if res_nm.x is not None:
                current_score = -res_nm.fun
                if current_score > best_score:
                    best_score = current_score
                    best_result = get_points(res_nm.x)
                    
        except Exception:
            pass

    return best_result
