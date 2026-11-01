"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
from itertools import combinations

N = 14

# EVOLVE-BLOCK-START
def _get_triangle_indices(n):
    """Precompute all (i,j,k) triple indices for n points."""
    idx = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                idx.append((i, j, k))
    return np.array(idx)

def _min_triangle_area(pts):
    """Compute minimum triangle area from point array (n,2)."""
    n = len(pts)
    idx = _get_triangle_indices(n)
    a = pts[idx[:, 0]]
    b = pts[idx[:, 1]]
    c = pts[idx[:, 2]]
    areas = 0.5 * np.abs((b[:,0]-a[:,0])*(c[:,1]-a[:,1]) - (c[:,0]-a[:,0])*(b[:,1]-a[:,1]))
    return np.min(areas)

def _hull_area(pts):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0

def _score(pts):
    """Score = min triangle area / convex hull area."""
    try:
        hull_area = _hull_area(pts)
        if hull_area < 1e-12:
            return 0.0
        min_area = _min_triangle_area(pts)
        return min_area / hull_area
    except Exception:
        return 0.0

def _soft_min_triangle_area(pts, beta=50.0):
    """Soft minimum of triangle areas for smoother optimization."""
    n = len(pts)
    idx = _get_triangle_indices(n)
    a = pts[idx[:, 0]]
    b = pts[idx[:, 1]]
    c = pts[idx[:, 2]]
    areas = 0.5 * np.abs((b[:,0]-a[:,0])*(c[:,1]-a[:,1]) - (c[:,0]-a[:,0])*(b[:,1]-a[:,1]))
    # Soft min: -beta * log(sum(exp(-beta * a)))
    m = np.min(areas)
    return m - np.log(np.sum(np.exp(-beta * (areas - m)))) / beta

def _neg_soft_score(pts, beta=50.0):
    """Negative soft score for minimization."""
    try:
        hull_area = _hull_area(pts)
        if hull_area < 1e-12:
            return 1.0
        soft_min = _soft_min_triangle_area(pts, beta)
        return -soft_min / hull_area
    except Exception:
        return 1.0

def _normalize(pts):
    """Translate to centroid at origin, scale so hull area = 1."""
    pts = pts.copy()
    pts -= np.mean(pts, axis=0)
    hull_area = _hull_area(pts)
    if hull_area > 1e-12:
        pts *= np.sqrt(1.0 / hull_area)
    return pts

def _regular_polygon(n, radius=1.0, angle_offset=0.0):
    """Generate n equally spaced points on a circle."""
    angles = np.linspace(angle_offset, 2*np.pi + angle_offset, n, endpoint=False)
    return np.column_stack((radius * np.cos(angles), radius * np.sin(angles)))

def _generate_parameterized_points(params):
    """
    Generate 14 points from parameters.
    params = [r_inner, angle_offset_inner, r_outer, angle_offset_outer]
    7 points on inner ring at radius r_inner with offset angle_offset_inner
    7 points on outer ring at radius r_outer with offset angle_offset_outer
    """
    r_in = params[0]
    a_in = params[1]
    r_out = params[2]
    a_out = params[3]
    
    inner = _regular_polygon(7, r_in, a_in)
    outer = _regular_polygon(7, r_out, a_out)
    return np.vstack([inner, outer])

def _score_parameterized(params):
    """Score for parameterized configuration."""
    pts = _generate_parameterized_points(params)
    return _score(pts)

def _neg_score_parameterized(params):
    """Negative score for minimization."""
    return -_score_parameterized(params)

def _neg_soft_score_parameterized(params, beta=50.0):
    """Negative soft score for minimization."""
    pts = _generate_parameterized_points(params)
    return _neg_soft_score(pts, beta)

def construct():
    deadline = time.time() + 130  # Safety margin before 150s kill
    np.random.seed(42)
    
    best_score = 0.0
    best_pts = None
    
    # Start with regular 14-gon
    init_pts = _regular_polygon(N, 1.0, 0.0)
    best_pts = init_pts.copy()
    best_score = _score(best_pts)
    
    # Phase 1: Optimize two-ring parameterization with Nelder-Mead
    # Parameters: [r_inner, angle_offset_inner, r_outer, angle_offset_outer]
    # We fix outer ring radius to 1.0 and optimize relative inner radius and offsets
    
    def _neg_score_param_full(p):
        """Full parameter optimization: r_in, offset_in, r_out, offset_out."""
        r_in = max(0.01, p[0])
        a_in = p[1]
        r_out = max(0.5, p[2])
        a_out = p[3]
        pts = _generate_parameterized_points([r_in, a_in, r_out, a_out])
        s = _score(pts)
        return -s if s > 0 else 1.0
    
    # Initial guess: inner ring at r=0.5, outer at r=1.0, offset by pi/7
    p0 = [0.5, np.pi/7, 1.0, 0.0]
    
    try:
        result = minimize(
            _neg_score_param_full,
            p0,
            method='Nelder-Mead',
            options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10}
        )
        if result.fun < -best_score:
            p_best = result.x
            best_pts = _generate_parameterized_points([
                max(0.01, p_best[0]), p_best[1], max(0.5, p_best[2]), p_best[3]
            ])
            best_score = -result.fun
    except Exception:
        pass
    
    # Phase 2: Try different initial offsets for the two rings
    for offset_try in range(5):
        if time.time() > deadline:
            break
        
        # Vary the relative angle offset between rings
        rel_offset = (offset_try + 1) * np.pi / 14.0
        
        p0_try = [0.4 + 0.1 * offset_try, rel_offset, 1.0, 0.0]
        
        try:
            result = minimize(
                _neg_score_param_full,
                p0_try,
                method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            if result.fun < -best_score:
                p_best = result.x
                best_pts = _generate_parameterized_points([
                    max(0.01, p_best[0]), p_best[1], max(0.5, p_best[2]), p_best[3]
                ])
                best_score = -result.fun
        except Exception:
            pass
    
    # Phase 3: Full local optimization on all 14 points using the best as start
    # Fix p0 at origin, p1 at (1,0) to remove translation/rotation
    def _normalize_for_opt(pts):
        pts = pts.copy()
        pts = pts - pts[0]
        p1 = pts[1]
        angle = np.arctan2(p1[1], p1[0])
        c, s = np.cos(-angle), np.sin(-angle)
        R = np.array([[c, -s], [s, c]])
        pts = pts @ R.T
        dist = np.linalg.norm(p1)
        if dist > 1e-12:
            pts = pts / dist
        return pts
    
    def _from_opt_vars(x):
        pts = np.zeros((N, 2))
        pts[0] = [0.0, 0.0]
        pts[1] = [1.0, 0.0]
        for i in range(N - 2):
            pts[i + 2, 0] = x[2 * i]
            pts[i + 2, 1] = x[2 * i + 1]
        return pts
    
    def _objective_norm(x):
        pts = _from_opt_vars(x)
        return -_score(pts)
    
    # Use best configuration as starting point for full optimization
    if best_pts is not None:
        norm_pts = _normalize_for_opt(best_pts)
        x0 = norm_pts[2:].flatten()
        
        try:
            result = minimize(
                _objective_norm,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8}
            )
            if result.fun < -best_score:
                new_pts = _from_opt_vars(result.x)
                new_score = _score(new_pts)
                if new_score > best_score:
                    best_pts = new_pts
                    best_score = new_score
        except Exception:
            pass
    
    # Phase 4: Try soft-min guided optimization for smoother landscape
    if time.time() < deadline:
        def _neg_soft_opt(x):
            pts = _from_opt_vars(x)
            return _neg_soft_score(pts, beta=30.0)
        
        if best_pts is not None:
            norm_pts = _normalize_for_opt(best_pts)
            x0 = norm_pts[2:].flatten()
            
            try:
                result = minimize(
                    _neg_soft_opt,
                    x0,
                    method='Nelder-Mead',
                    options={'maxiter': 300, 'xatol': 1e-6, 'fatol': 1e-8}
                )
                if result.fun < -best_score:
                    new_pts = _from_opt_vars(result.x)
                    new_score = _score(new_pts)
                    if new_score > best_score:
                        best_pts = new_pts
                        best_score = new_score
            except Exception:
                pass
    
    return best_pts
# EVOLVE-BLOCK-END
