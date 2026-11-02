"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
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


def _normalize_to_unit_hull(pts):
    """Scale points so that convex hull area = 1."""
    ha = _hull_area(pts)
    if ha > 1e-12:
        scale = 1.0 / np.sqrt(ha)
        return pts * scale
    return pts


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


def _neg_smooth_score(params, temperature=0.01):
    """
    Negative smooth score for minimization with normalized hull area.
    We fix hull area to 1 by normalizing, then minimize -soft_min(areas).
    This removes the scale invariance flat direction.
    """
    pts = params.reshape(-1, 2)
    
    # Normalize to unit hull area to remove scale invariance
    ha = _hull_area(pts)
    if ha <= 1e-12:
        return 1e6  # Large penalty for degenerate
    
    scale = 1.0 / np.sqrt(ha)
    pts_norm = pts * scale
    
    areas = _all_triangle_areas(pts_norm)
    if len(areas) == 0:
        return 1e6
    
    # Soft min approximation
    min_val = np.min(areas)
    # Use log-sum-exp trick for numerical stability
    shifted = areas - min_val
    log_sum = np.log(np.sum(np.exp(-shifted / temperature)))
    soft_min = min_val - temperature * log_sum
    
    # We want to maximize soft_min, so minimize -soft_min
    return -soft_min


def _neg_hard_score(params):
    """
    Negative hard score for minimization with normalized hull area.
    """
    pts = params.reshape(-1, 2)
    ha = _hull_area(pts)
    if ha <= 1e-12:
        return 1e6
    
    scale = 1.0 / np.sqrt(ha)
    pts_norm = pts * scale
    
    areas = _all_triangle_areas(pts_norm)
    if len(areas) == 0:
        return 1e6
    
    return -np.min(areas)


def _generate_initializations():
    """Generate diverse initial configurations, all normalized to unit hull area."""
    inits = []
    
    # Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    pts = np.column_stack([np.cos(angles), np.sin(angles)])
    inits.append(_normalize_to_unit_hull(pts))
    
    # 12-gon + center
    angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    pts = np.column_stack([np.cos(angles), np.sin(angles)])
    center = np.array([[0.0, 0.0]])
    pts = np.vstack([pts, center])
    inits.append(_normalize_to_unit_hull(pts))
    
    # Two concentric polygons (6+7)
    angles1 = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    pts1 = np.column_stack([0.5 * np.cos(angles1), 0.5 * np.sin(angles1)])
    angles2 = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    pts2 = np.column_stack([np.cos(angles2), np.sin(angles2)])
    inits.append(_normalize_to_unit_hull(np.vstack([pts1, pts2])))
    
    # Three rings (4+4+5)
    angles1 = np.linspace(0, 2 * np.pi, 4, endpoint=False)
    pts1 = np.column_stack([0.33 * np.cos(angles1), 0.33 * np.sin(angles1)])
    angles2 = np.linspace(0, 2 * np.pi, 4, endpoint=False) + np.pi/4
    pts2 = np.column_stack([0.66 * np.cos(angles2), 0.66 * np.sin(angles2)])
    angles3 = np.linspace(0, 2 * np.pi, 5, endpoint=False)
    pts3 = np.column_stack([np.cos(angles3), np.sin(angles3)])
    inits.append(_normalize_to_unit_hull(np.vstack([pts1, pts2, pts3])))
    
    # Random perturbations of regular polygon
    np.random.seed(42)
    for seed in range(8):
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        radii = 1.0 + np.random.normal(0, 0.15, N)
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        inits.append(_normalize_to_unit_hull(pts))
    
    # Random points in unit disk
    for seed in range(5):
        np.random.seed(100 + seed)
        angles = np.random.uniform(0, 2 * np.pi, N)
        radii = np.sqrt(np.random.uniform(0.3, 1.0, N))
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        inits.append(_normalize_to_unit_hull(pts))
    
    return inits


def _optimize_from_start(pts, deadline):
    """
    Multi-stage optimization from a single starting point:
    1. L-BFGS-B with soft-min annealing (global exploration)
    2. Powell for robust local refinement on hard objective
    """
    best_score = _score_points(pts)
    best_pts = pts.copy()
    
    # Phase 1: L-BFGS-B with soft-min annealing
    temperatures = [0.1, 0.05, 0.02, 0.01, 0.005]
    for temp in temperatures:
        if time.time() > deadline - 10:
            break
        
        try:
            result = minimize(
                lambda x: _neg_smooth_score(x, temperature=temp),
                best_pts.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if result.fun is not None and result.fun < 1e5:
                candidate_pts = result.x.reshape(-1, 2)
                candidate_score = _score_points(candidate_pts)
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_pts = candidate_pts.copy()
                    
        except Exception:
            pass
    
    # Phase 2: Powell for robust local refinement on hard objective
    if time.time() < deadline - 5:
        try:
            result = minimize(
                _neg_hard_score,
                best_pts.flatten(),
                method='Powell',
                options={'maxiter': 300, 'xtol': 1e-8, 'ftol': 1e-10}
            )
            
            if result.fun is not None and result.fun < 1e5:
                candidate_pts = result.x.reshape(-1, 2)
                candidate_score = _score_points(candidate_pts)
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_pts = candidate_pts.copy()
                    
        except Exception:
            pass
    
    # Phase 3: Final Nelder-Mead polish
    if time.time() < deadline - 2:
        try:
            result = minimize(
                _neg_hard_score,
                best_pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-9, 'fatol': 1e-10}
            )
            
            if result.fun is not None and result.fun < 1e5:
                candidate_pts = result.x.reshape(-1, 2)
                candidate_score = _score_points(candidate_pts)
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_pts = candidate_pts.copy()
                    
        except Exception:
            pass
    
    return best_pts, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
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
    
    # Multi-stage optimization from top starting points
    if best_pts is not None:
        # Sort initializations by score and try top few
        scored_inits = []
        for pts in inits:
            s = _score_points(pts)
            scored_inits.append((s, pts))
        scored_inits.sort(key=lambda x: -x[0])
        
        # Try optimization from top 4 initializations
        for idx in range(min(4, len(scored_inits))):
            if time.time() > deadline - 20:
                break
            
            init_pts = scored_inits[idx][1]
            try:
                optimized_pts, optimized_score = _optimize_from_start(init_pts, deadline - 5)
                if optimized_score > best_score:
                    best_score = optimized_score
                    best_pts = optimized_pts.copy()
            except Exception:
                pass
    
    # Final refinement on the best solution
    if best_pts is not None and time.time() < deadline - 5:
        try:
            result = minimize(
                _neg_hard_score,
                best_pts.flatten(),
                method='Powell',
                options={'maxiter': 200, 'xtol': 1e-9, 'ftol': 1e-11}
            )
            if result.fun is not None and result.fun < 1e5:
                candidate_pts = result.x.reshape(-1, 2)
                candidate_score = _score_points(candidate_pts)
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_pts = candidate_pts.copy()
        except Exception:
            pass
    
    # Return the best points found
    if best_pts is not None and len(best_pts) == N:
        return best_pts
    else:
        # Fallback: return first initialization
        return inits[0]
# EVOLVE-BLOCK-END
