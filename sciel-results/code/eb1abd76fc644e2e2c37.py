"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull

def _score(points):
    """
    Calculate the score: min triangle area / convex hull area.
    points: np.array of shape (N, 2)
    """
    n = len(points)
    if n < 3:
        return 0.0
    
    # Calculate min triangle area efficiently
    min_area = float('inf')
    for i in range(n - 2):
        pi = points[i]
        diffs = points[i+1:] - pi
        m = len(diffs)
        if m < 2:
            continue
        
        # Compute cross products for all pairs in diffs
        cross = diffs[:, 0][:, None] * diffs[:, 1][None, :] - diffs[:, 1][:, None] * diffs[:, 0][None, :]
        
        # We only care about a < b
        mask = np.triu(np.ones((m, m), dtype=bool), k=1)
        if not np.any(mask):
            continue
            
        areas = 0.5 * np.abs(cross[mask])
        current_min = np.min(areas)
        if current_min < min_area:
            min_area = current_min
            
    # Calculate convex hull area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        return 0.0
        
    if hull_area < 1e-10:
        return 0.0
        
    return min_area / hull_area

def _neg_score(points_flat):
    """Negative score for minimization."""
    points = points_flat.reshape(-1, 2)
    return -_score(points)

def _soft_min_score(points, beta=100.0):
    """
    Smooth surrogate using soft-min for the minimum triangle area.
    This makes the objective differentiable, allowing gradient-based optimization.
    """
    n = len(points)
    if n < 3:
        return 0.0
    
    # Calculate all triangle areas
    areas = []
    for i in range(n - 2):
        pi = points[i]
        diffs = points[i+1:] - pi
        m = len(diffs)
        if m < 2:
            continue
        
        cross = diffs[:, 0][:, None] * diffs[:, 1][None, :] - diffs[:, 1][:, None] * diffs[:, 0][None, :]
        mask = np.triu(np.ones((m, m), dtype=bool), k=1)
        if not np.any(mask):
            continue
            
        areas.append(0.5 * np.abs(cross[mask]))
    
    if not areas:
        return 0.0
        
    all_areas = np.concatenate(areas)
    
    # Soft-min approximation
    min_val = np.min(all_areas)
    max_val = np.max(all_areas)
    if max_val - min_val < 1e-10:
        soft_min = min_val
    else:
        # Use log-sum-exp trick for numerical stability
        shifted = all_areas - max_val
        log_sum = np.log(np.sum(np.exp(-beta * shifted))) - np.log(len(all_areas))
        soft_min = max_val + log_sum / (-beta)
        
    # Calculate convex hull area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        return 0.0
        
    if hull_area < 1e-10:
        return 0.0
        
    return soft_min / hull_area

def _neg_soft_score(points_flat, beta=100.0):
    """Negative soft score for minimization."""
    points = points_flat.reshape(-1, 2)
    return -_soft_min_score(points, beta)

def _refine_local(points, deadline):
    """
    Local refinement using Nelder-Mead on the exact objective.
    """
    best_score = _score(points)
    best_points = points.copy()
    
    try:
        res = minimize(_neg_score, best_points.flatten(), method='Nelder-Mead', 
                       options={'maxiter': 2000, 'xatol': 1e-7, 'fatol': 1e-9, 'adaptive': True})
        
        if time.time() < deadline:
            p_opt = res.x.reshape(-1, 2)
            s_opt = _score(p_opt)
            if s_opt > best_score:
                best_score = s_opt
                best_points = p_opt
    except Exception:
        pass
    
    # Try a few small perturbations and refine
    for _ in range(3):
        if time.time() > deadline - 5.0:
            break
        perturbed = best_points + 0.01 * np.random.randn(N, 2)
        try:
            res = minimize(_neg_score, perturbed.flatten(), method='Nelder-Mead', 
                           options={'maxiter': 1000, 'xatol': 1e-7, 'fatol': 1e-9, 'adaptive': True})
            if time.time() < deadline:
                p_opt = res.x.reshape(-1, 2)
                s_opt = _score(p_opt)
                if s_opt > best_score:
                    best_score = s_opt
                    best_points = p_opt
        except Exception:
            pass
    
    return best_points, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin before 150s limit
    
    # Initial guess: Regular 14-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    x0 = np.column_stack([np.cos(angles), np.sin(angles)])
    
    best_score = _score(x0)
    best_points = x0.copy()
    
    # Phase 1: Differential Evolution for global search
    if time.time() < deadline - 60.0:
        try:
            # Bounds for points: roughly in [-1.5, 1.5]^2
            bounds = [(-1.5, 1.5)] * (2 * N)
            
            def _de_objective(x_flat):
                if time.time() > deadline - 60.0:
                    return 1e10
                points = x_flat.reshape(-1, 2)
                return -_score(points)
            
            de_result = differential_evolution(_de_objective, bounds, maxiter=100, 
                                                popsize=15, tol=1e-8, seed=42, 
                                                polish=False, init='latinhypercube')
            
            if time.time() < deadline - 30.0:
                p_de = de_result.x.reshape(-1, 2)
                s_de = _score(p_de)
                if s_de > best_score:
                    best_score = s_de
                    best_points = p_de
        except Exception:
            pass
    
    # Phase 2: Local refinement from multiple starting points
    configs = [
        best_points,
        x0,
        x0 + 0.1 * np.random.randn(N, 2),
        x0 + 0.05 * np.random.randn(N, 2),
        # Two rings: 7 outer, 7 inner
        np.vstack([
            np.column_stack([np.cos(np.linspace(0, 2*np.pi, 7, endpoint=False)), 
                             np.sin(np.linspace(0, 2*np.pi, 7, endpoint=False))]),
            np.column_stack([0.5*np.cos(np.linspace(0, 2*np.pi, 7, endpoint=False) + np.pi/7), 
                             0.5*np.sin(np.linspace(0, 2*np.pi, 7, endpoint=False) + np.pi/7)])
        ]),
        # Ellipse
        np.column_stack([1.2 * np.cos(angles), 0.8 * np.sin(angles)])
    ]
    
    for x_init in configs:
        if time.time() > deadline - 20.0:
            break
        
        p_refined, s_refined = _refine_local(x_init, deadline)
        if s_refined > best_score:
            best_score = s_refined
            best_points = p_refined
    
    # Final refinement on the best solution found
    if time.time() < deadline - 10.0:
        p_final, s_final = _refine_local(best_points, deadline)
        if s_final > best_score:
            best_score = s_final
            best_points = p_final
    
    return best_points
# EVOLVE-BLOCK-END
