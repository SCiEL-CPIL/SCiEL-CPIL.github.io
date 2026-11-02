"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize, differential_evolution
import time
import itertools


def _all_triangle_areas(pts):
    """Compute all triangle areas among triples of points, vectorized."""
    n = len(pts)
    if n < 3:
        return np.array([])
    
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            v1 = pts[j] - pts[i]
            v2 = pts[j+1:] - pts[i]
            cross = np.abs(v1[0] * v2[:, 1] - v1[1] * v2[:, 0])
            areas.append(0.5 * cross)
    
    if len(areas) == 0:
        return np.array([])
    return np.concatenate(areas)


def _min_triangle_area(pts):
    """Compute the minimum triangle area among all triples of points."""
    areas = _all_triangle_areas(pts)
    if len(areas) == 0:
        return 0.0
    return np.min(areas)


def _hull_area(pts):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0


def _score_points(pts):
    """Compute min triangle area / convex hull area."""
    n = len(pts)
    if n < 3:
        return 0.0
    hull_area = _hull_area(pts)
    if hull_area <= 1e-12:
        return 0.0
    min_area = _min_triangle_area(pts)
    return min_area / hull_area


def _soft_min_triangle_area(pts, temperature=0.01):
    """Smooth approximation of min triangle area using soft-min."""
    areas = _all_triangle_areas(pts)
    if len(areas) == 0:
        return 0.0
    min_val = np.min(areas)
    exp_terms = np.exp(-(areas - min_val) / temperature)
    soft_min = min_val - temperature * np.log(np.sum(exp_terms))
    return soft_min


def _neg_smooth_score(params, temperature=0.01):
    """Negative smooth score for minimization."""
    pts = params.reshape(-1, 2)
    hull_area = _hull_area(pts)
    if hull_area <= 1e-12:
        return 0.0
    soft_min = _soft_min_triangle_area(pts, temperature)
    return -soft_min / hull_area


def _neg_score(params):
    """Negative score for minimization (hard min)."""
    pts = params.reshape(-1, 2)
    return -_score_points(pts)


def _normalize_hull(pts):
    """Normalize points so that convex hull area is 1."""
    hull_area = _hull_area(pts)
    if hull_area <= 1e-12:
        return pts
    # Scale by sqrt(1/hull_area) to make hull area = 1
    scale = 1.0 / np.sqrt(hull_area)
    return pts * scale


def _normalized_score(pts):
    """Score with hull area normalized to 1."""
    norm_pts = _normalize_hull(pts)
    min_area = _min_triangle_area(norm_pts)
    return min_area


def _neg_normalized_score(params):
    """Negative normalized score for minimization."""
    pts = params.reshape(-1, 2)
    return -_normalized_score(pts)


def _neg_normalized_smooth_score(params, temperature=0.01):
    """Negative normalized smooth score for minimization."""
    pts = params.reshape(-1, 2)
    norm_pts = _normalize_hull(pts)
    soft_min = _soft_min_triangle_area(norm_pts, temperature)
    return -soft_min


def _generate_initializations():
    """Generate diverse initial configurations."""
    inits = []
    
    # Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    pts = np.column_stack([np.cos(angles), np.sin(angles)])
    inits.append(pts.copy())
    
    # 12-gon + center
    angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    pts = np.column_stack([np.cos(angles), np.sin(angles)])
    center = np.array([[0.0, 0.0]])
    pts = np.vstack([pts, center])
    inits.append(pts.copy())
    
    # Two concentric polygons (6+7)
    angles1 = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    pts1 = np.column_stack([0.5 * np.cos(angles1), 0.5 * np.sin(angles1)])
    angles2 = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    pts2 = np.column_stack([np.cos(angles2), np.sin(angles2)])
    inits.append(np.vstack([pts1, pts2]))
    
    # Three rings (4+4+5)
    angles1 = np.linspace(0, 2 * np.pi, 4, endpoint=False)
    pts1 = np.column_stack([0.33 * np.cos(angles1), 0.33 * np.sin(angles1)])
    angles2 = np.linspace(0, 2 * np.pi, 4, endpoint=False) + np.pi/4
    pts2 = np.column_stack([0.66 * np.cos(angles2), 0.66 * np.sin(angles2)])
    angles3 = np.linspace(0, 2 * np.pi, 5, endpoint=False)
    pts3 = np.column_stack([np.cos(angles3), np.sin(angles3)])
    inits.append(np.vstack([pts1, pts2, pts3]))
    
    # Grid-like (4x4 minus one)
    grid_pts = []
    for i in range(4):
        for j in range(4):
            if i == 3 and j == 3:
                continue
            grid_pts.append([i / 3.0, j / 3.0])
    inits.append(np.array(grid_pts[:N]))
    
    # Random perturbations of regular polygon
    np.random.seed(42)
    for seed in range(5):
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        radii = 1.0 + np.random.normal(0, 0.1, N)
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        inits.append(pts.copy())
    
    return inits


def _powell_refine(pts, deadline):
    """Refine points using Powell method on the hard objective."""
    best_pts = pts.copy()
    best_score = _score_points(best_pts)
    
    # Powell on the normalized objective
    try:
        result = minimize(
            _neg_normalized_score,
            best_pts.flatten(),
            method='Powell',
            options={'maxiter': 500, 'xtol': 1e-8, 'ftol': 1e-10}
        )
        if result.fun is not None:
            candidate_pts = result.x.reshape(-1, 2)
            candidate_score = _score_points(candidate_pts)
            if candidate_score > best_score:
                best_score = candidate_score
                best_pts = candidate_pts.copy()
    except Exception:
        pass
    
    return best_pts, best_score


def _local_search(pts, deadline):
    """Local search with random perturbations and Powell refinement."""
    best_pts = pts.copy()
    best_score = _score_points(best_pts)
    
    n_iter = 0
    while time.time() < deadline - 5 and n_iter < 50:
        n_iter += 1
        # Random perturbation
        perturbation_scale = 0.05 * (1 - n_iter / 100)
        new_pts = best_pts + np.random.normal(0, perturbation_scale, size=best_pts.shape)
        
        # Refine with Powell
        refined_pts, refined_score = _powell_refine(new_pts, deadline)
        
        if refined_score > best_score:
            best_score = refined_score
            best_pts = refined_pts
    
    return best_pts, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    best_score = 0.0
    best_pts = None
    
    # Generate diverse initializations
    inits = _generate_initializations()
    
    # Evaluate all initializations
    for pts in inits:
        score = _score_points(pts)
        if score > best_score:
            best_score = score
            best_pts = pts.copy()
    
    # Strategy 1: L-BFGS-B with soft-min annealing on normalized objective
    temperatures = [0.05, 0.02, 0.01, 0.005, 0.002]
    
    for temp in temperatures:
        if time.time() > deadline - 30:
            break
        
        # Try from multiple starting points
        n_starts = min(3, len(inits))
        for i in range(n_starts):
            if time.time() > deadline - 15:
                break
            
            start = inits[i].flatten().copy()
            
            try:
                result = minimize(
                    lambda x: _neg_normalized_smooth_score(x, temperature=temp),
                    start,
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                
                if result.fun is not None:
                    candidate_pts = result.x.reshape(-1, 2)
                    candidate_score = _score_points(candidate_pts)
                    if candidate_score > best_score:
                        best_score = candidate_score
                        best_pts = candidate_pts.copy()
                        
            except Exception:
                pass
    
    # Strategy 2: Powell refinement on hard objective
    if time.time() < deadline - 20:
        refined_pts, refined_score = _powell_refine(best_pts, deadline)
        if refined_score > best_score:
            best_score = refined_score
            best_pts = refined_pts
    
    # Strategy 3: Local search with random perturbations
    if time.time() < deadline - 10:
        search_pts, search_score = _local_search(best_pts, deadline)
        if search_score > best_score:
            best_score = search_score
            best_pts = search_pts
    
    # Strategy 4: Differential evolution for global search
    if time.time() < deadline - 30:
        bounds = [(0, 1), (0, 1)] * N
        
        try:
            result = differential_evolution(
                _neg_score,
                bounds=bounds,
                maxiter=50,
                popsize=12,
                tol=1e-4,
                seed=42,
                init='sobol',
                polish=False,
                x0=best_pts.flatten()
            )
            
            if result.fun is not None:
                candidate_pts = result.x.reshape(-1, 2)
                candidate_score = _score_points(candidate_pts)
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_pts = candidate_pts.copy()
                    
        except Exception:
            pass
    
    # Final polish: Powell refinement
    if time.time() < deadline - 3:
        refined_pts, refined_score = _powell_refine(best_pts, deadline)
        if refined_score > best_score:
            best_score = refined_score
            best_pts = refined_pts
    
    # Return the best points found
    if best_pts is not None and len(best_pts) == N:
        return best_pts
    else:
        # Fallback: return first initialization
        return inits[0]
# EVOLVE-BLOCK-END

