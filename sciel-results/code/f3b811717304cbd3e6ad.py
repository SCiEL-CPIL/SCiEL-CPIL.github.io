import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 13

def _all_triangle_areas(points):
    """Vectorized computation of all C(n,3) triangle areas."""
    n = len(points)
    if n < 3:
        return np.array([0.0])
    idx = np.arange(n)
    i_idx, j_idx, k_idx = np.meshgrid(idx, idx, idx, indexing='ij')
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    i_idx, j_idx, k_idx = i_idx[mask], j_idx[mask], k_idx[mask]
    pi = points[i_idx]
    pj = points[j_idx]
    pk = points[k_idx]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    return areas

def _hull_area(points):
    """Compute area of the convex hull."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 1.0

def _exact_score(points):
    """Calculate the exact score: min triangle area / hull area."""
    areas = _all_triangle_areas(points)
    min_area = np.min(areas)
    hull_area = _hull_area(points)
    if hull_area <= 0:
        return 0.0
    return min_area / hull_area

def _soft_objective(params, beta=50.0):
    """Smoothed objective for L-BFGS-B. Minimizes negative soft-min area / hull area."""
    points = params.reshape(-1, 2)
    areas = _all_triangle_areas(points)
    # Soft min approximation
    scaled_areas = -beta * areas
    max_val = np.max(scaled_areas)
    if max_val == -np.inf:
        return 1e10
    log_sum_exp = max_val + np.log(np.sum(np.exp(scaled_areas - max_val)))
    soft_min = -log_sum_exp / beta
    hull_area = _hull_area(points)
    if hull_area <= 0:
        return 1e10
    return -soft_min / hull_area

def _refine_local(points, t0, deadline):
    """Refine a single point configuration using L-BFGS-B and Nelder-Mead."""
    best_points = points
    best_score = _exact_score(points)
    
    # Step 1: L-BFGS-B on soft objective
    if time.time() - t0 < deadline - 5:
        try:
            x0 = best_points.flatten()
            # Use a moderate beta for smoothness
            res1 = minimize(_soft_objective, x0, args=(50.0,), method='L-BFGS-B',
                            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
            cand = res1.x.reshape(-1, 2)
            score1 = _exact_score(cand)
            if score1 > best_score:
                best_score = score1
                best_points = cand
        except Exception:
            pass

    # Step 2: Nelder-Mead on exact objective for final polish
    if time.time() - t0 < deadline - 2:
        try:
            x0 = best_points.flatten()
            res2 = minimize(lambda x: -_exact_score(x.reshape(-1, 2)), x0, method='Nelder-Mead',
                            options={'maxiter': 500, 'fatol': 1e-10, 'xatol': 1e-10, 'adaptive': True})
            cand = res2.x.reshape(-1, 2)
            score2 = _exact_score(cand)
            if score2 > best_score:
                best_score = score2
                best_points = cand
        except Exception:
            pass

    return best_points, best_score

def construct():
    n = N
    t0 = time.time()
    deadline = 140.0  # Safety margin from 150s limit
    
    best_points = None
    best_score = -np.inf
    
    # Strategy: Multi-start local refinement
    # 1. Start with regular 13-gon
    # 2. Add random perturbations and refine each
    
    # Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    base_points = np.column_stack((np.cos(angles), np.sin(angles)))
    
    # List of initial candidates
    candidates = [base_points]
    
    # Add some random perturbations of the base points
    # We can also try points slightly inside/outside the circle
    rng = np.random.default_rng(42)
    for i in range(10):
        # Random perturbation
        noise = rng.normal(0, 0.1, size=(n, 2))
        cand = base_points + noise
        candidates.append(cand)
        
        # Slight scaling
        scale = rng.uniform(0.8, 1.2)
        cand_scaled = base_points * scale
        candidates.append(cand_scaled)
        
        # Random points in unit disk (to explore different hull shapes)
        # This might find a better hull shape
        # r = rng.uniform(0, 1, n)
        # theta = rng.uniform(0, 2*np.pi, n)
        # cand_disk = np.column_stack((r * np.cos(theta), r * np.sin(theta)))
        # candidates.append(cand_disk)

    # Refine each candidate
    for i, cand in enumerate(candidates):
        if time.time() - t0 > deadline - 10:
            break
        
        # Check if we have enough time for one refinement
        # Each refinement takes roughly 1-5 seconds depending on convergence
        refined_pts, score = _refine_local(cand, t0, deadline)
        
        if score > best_score:
            best_score = score
            best_points = refined_pts

    # Fallback: If no valid points were found (should not happen)
    if best_points is None:
        best_points = base_points
        best_score = _exact_score(best_points)

    return best_points
