"""
Heilbronn convex regions n=14: maximize min triangle area / convex hull area.

Strategy: Centrally Symmetric Convex Polygon parameterization.
  1. Points are vertices of a convex polygon with central symmetry.
  2. Parameterized by 7 edge lengths and 7 edge angles (14 dims total).
     This enforces convexity and symmetry, reducing search space.
  3. Optimization: Differential Evolution (global) + Nelder-Mead/Powell (local).
  4. Time-guarded to finish within 150s wall-clock.
"""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull
from itertools import combinations

N = 14

# Pre-compute all triangle index combinations (364 triples)
tri_indices = np.array(list(combinations(range(N), 3)))  # shape (364, 3)


# ---------- Core geometry (fast, vectorized) ----------

def _min_triangle_area(points):
    """Minimum triangle area over all triples, vectorized."""
    pi = points[tri_indices[:, 0]]
    pj = points[tri_indices[:, 1]]
    pk = points[tri_indices[:, 2]]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - \
            (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    return np.min(areas)


def _all_triangle_areas(points):
    """All triangle areas over all triples, vectorized."""
    pi = points[tri_indices[:, 0]]
    pj = points[tri_indices[:, 1]]
    pk = points[tri_indices[:, 2]]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - \
            (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    return 0.5 * np.abs(cross)


def _hull_area(points):
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0


def _score(points):
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    min_area = _min_triangle_area(points)
    return min_area / hull_area


def _objective_hard(flat):
    """Minimize hull_area / min_triangle_area (equivalent to maximizing score)."""
    pts = flat.reshape(N, 2)
    if np.any(~np.isfinite(pts)):
        return 1e10
    min_area = _min_triangle_area(pts)
    if min_area < 1e-12:
        return 1e10
    hull_area = _hull_area(pts)
    if hull_area < 1e-10:
        return 1e10
    return hull_area / min_area


def _objective_soft(flat, beta=50.0):
    """Soft objective using log-sum-exp for smooth minimum approximation."""
    pts = flat.reshape(N, 2)
    if np.any(~np.isfinite(pts)):
        return 1e10
    areas = _all_triangle_areas(pts)
    if np.min(areas) < 1e-12:
        return 1e10
    z = -beta * areas
    max_z = np.max(z)
    if max_z < -500:
        log_sum = max_z
    else:
        log_sum = max_z + np.log(np.sum(np.exp(z - max_z)))
    soft_min = -log_sum / beta
    if soft_min < 1e-12:
        return 1e10
    hull_area = _hull_area(pts)
    if hull_area < 1e-10:
        return 1e10
    return hull_area / soft_min


# ---------- Centrally Symmetric Convex Polygon Parameterization ----------
# 7 independent edges in the first octant.
# Edge i has angle theta_i in [0, pi/7] and length L_i > 0.
# The full polygon has 14 edges: v_0, ..., v_6, -v_0, ..., -v_6.
# Vertices are constructed by summing edge vectors from the origin.
# Since the polygon is centrally symmetric, it passes through the origin?
# No, we construct the vertices relative to the first vertex.

def _decode_symmetric(params):
    """
    params: [L_0, ..., L_6, theta_0, ..., theta_6]
    14 parameters total.
    Returns (N, 2) array of points.
    """
    L = params[:7]
    theta = params[7:14]
    
    # Ensure convexity: angles must be in [0, pi/7] and sum to pi/2?
    # Actually, for a centrally symmetric convex polygon with 14 vertices,
    # the 7 edges in the first half must span an angle of pi/2.
    # The angles of the edges relative to the previous edge are not directly theta.
    # Let's define theta_i as the absolute angle of edge i.
    # For the polygon to close and be convex, the angles must be increasing.
    # And the total turn for 7 edges is pi/2 (90 degrees) due to symmetry?
    # Yes, if we start at angle 0, after 7 edges we are at angle pi/2?
    # Let's just enforce 0 <= theta_0 < theta_1 < ... < theta_6 < pi/2.
    
    # To ensure strict convexity and closure, we can map parameters to angles.
    # Let alpha_i be the angle increment. Sum(alpha) = pi/2.
    # We can parameterize alpha_i via softmax or simple normalization.
    # But simpler: just constrain theta directly in the optimizer bounds.
    # If we use DE/NM, we can handle bounds.
    # However, we need to ensure the polygon closes.
    # Sum of 14 edge vectors must be 0.
    # v_0 + ... + v_6 + (-v_0) + ... + (-v_6) = 0. This is always true by construction.
    
    # We need to ensure the vertices are distinct and ordered.
    # If angles are sorted and lengths positive, the polygon is convex and star-shaped wrt origin?
    # Not necessarily. But if we start at origin and add vectors, we get a path.
    # For a convex polygon containing the origin, the vertices can be generated this way.
    
    # Let's generate the 14 edge vectors.
    edges = np.zeros((14, 2))
    for i in range(7):
        # Angle must be in [0, pi/2)
        # We will rely on the optimizer to keep them in range, or we clip here.
        # Clipping might cause non-smoothness. Better to use bounds in DE.
        # For local optimizers, we might need to map.
        # Let's assume params are already in valid range or we clip for safety.
        t = np.clip(theta[i], 0, np.pi/2 - 1e-6)
        l = np.clip(L[i], 1e-6, 10.0)
        
        edges[i] = np.array([l * np.cos(t), l * np.sin(t)])
        edges[i+7] = -edges[i]
        
    # Construct vertices
    # Start at origin? No, that would make one vertex at origin.
    # We want the polygon centered at origin.
    # Sum of vertices should be 0.
    # Let's compute the path starting from origin.
    verts = np.zeros((14, 2))
    current = np.array([0.0, 0.0])
    for i in range(14):
        verts[i] = current
        current = current + edges[i]
        
    # The path should close (current should be 0).
    # Now center the polygon.
    center = np.mean(verts, axis=0)
    verts = verts - center
    
    return verts


def _objective_symmetric(params, deadline):
    """Objective for symmetric polygon parameters."""
    if time.time() > deadline:
        return 1e10
    pts = _decode_symmetric(params)
    return _objective_hard(pts.flatten())


# ---------- Optimizer helpers ----------

def _nm(flat0, deadline, obj, max_iter=2000):
    if time.time() > deadline - 3:
        return None
    try:
        res = minimize(obj, flat0, method='Nelder-Mead',
                       options={'maxiter': max_iter, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True})
        return res
    except Exception:
        return None


def _powell(flat0, deadline, obj, max_iter=1000):
    if time.time() > deadline - 3:
        return None
    try:
        res = minimize(obj, flat0, method='Powell',
                       options={'maxiter': max_iter, 'xtol': 1e-11, 'ftol': 1e-13})
        return res
    except Exception:
        return None


def _lbfgs(flat0, deadline, obj, max_iter=3000):
    if time.time() > deadline - 5:
        return None
    try:
        res = minimize(obj, flat0, method='L-BFGS-B',
                       options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-10})
        return res
    except Exception:
        return None


def _normalize_unit_area(points):
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area > 1e-10:
            scale = np.sqrt(1.0 / hull_area)
            return points * scale
    except Exception:
        pass
    xs = points[:, 0]
    ys = points[:, 1]
    w = xs.max() - xs.min()
    h = ys.max() - ys.min()
    if w > 1e-10 and h > 1e-10:
        scale = 1.0 / np.sqrt(w * h)
        return points * scale
    return points


# ---------- Main ----------

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill
    
    best_score = 0.0
    best_pts = None
    
    def _consider(pts):
        nonlocal best_score, best_pts
        pts = np.asarray(pts, dtype=float)
        if pts.shape != (N, 2) or np.any(~np.isfinite(pts)):
            return
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_pts = pts.copy()
    
    # Phase 1: Differential Evolution in symmetric parameter space
    # Bounds: Lengths [0.1, 2.0], Angles [0, pi/2]
    # We need to ensure angles are sorted? 
    # If we don't enforce sorting, the polygon might self-intersect or be non-convex.
    # However, if we define edges by absolute angles, and the polygon is centrally symmetric,
    # convexity requires that the angles are increasing and span exactly pi/2.
    # To handle this, we can parameterize the *differences* or use a transformation.
    # Simpler: Let theta_i be the cumulative angle.
    # theta_0 < theta_1 < ... < theta_6.
    # We can enforce this by parameterizing deltas d_i > 0 such that sum(d_i) = pi/2.
    # Or just use DE with bounds and hope it finds a valid convex one? 
    # Risky. 
    # Let's use a transformation: 
    # Let u_i be free parameters. 
    # theta_i = cumulative sum of softmax(u) * (pi/2).
    # This ensures 0 < theta_i < pi/2 and strictly increasing.
    
    def _decode_symmetric_safe(params):
        L = params[:7]
        u = params[7:14]
        
        # Normalize lengths to be positive
        L = np.exp(L) # or clip
        
        # Transform u to angles
        # softmax
        u_max = np.max(u)
        u_exp = np.exp(u - u_max)
        u_sum = np.sum(u_exp)
        if u_sum == 0:
            probs = np.ones(7) / 7.0
        else:
            probs = u_exp / u_sum
            
        # Cumulative angles
        angles = np.cumsum(probs) * (np.pi / 2.0)
        
        edges = np.zeros((14, 2))
        for i in range(7):
            t = angles[i]
            l = L[i]
            edges[i] = np.array([l * np.cos(t), l * np.sin(t)])
            edges[i+7] = -edges[i]
            
        verts = np.zeros((14, 2))
        current = np.array([0.0, 0.0])
        for i in range(14):
            verts[i] = current
            current = current + edges[i]
            
        center = np.mean(verts, axis=0)
        verts = verts - center
        
        return verts

    def _objective_safe(params, deadline):
        if time.time() > deadline:
            return 1e10
        pts = _decode_symmetric_safe(params)
        return _objective_hard(pts.flatten())

    # Bounds for DE:
    # L: log-space, so [-2, 2] (0.1 to 7.4)
    # u: free, [-5, 5]
    bounds = [(-2, 2)] * 7 + [(-5, 5)] * 7
    
    try:
        res_de = differential_evolution(
            _objective_safe, bounds,
            args=(deadline,),
            seed=42, maxiter=100, tol=1e-8,
            popsize=20, mutation=(0.5, 1.0), recombination=0.7,
            polish=False, update_interval=1
        )
        if res_de is not None and np.isfinite(res_de.fun) and res_de.fun < 1e9:
            pts = _decode_symmetric_safe(res_de.x)
            _consider(pts)
    except Exception:
        pass

    # Phase 2: Local refinement in safe parameter space
    if best_pts is not None and time.time() < deadline - 30:
        # We need to recover parameters from best_pts? 
        # No, we keep the best *parameters* found by DE.
        # Let's track best_params separately.
        
        # Re-run DE logic to track params? 
        # Better: just run local optimizers on the best params found.
        
        # We didn't track best_params in the DE call above.
        # Let's do a quick restart of DE or just use the result if available.
        # Since we can't easily recover params from pts, we should track them.
        
        # Let's modify the flow:
        # 1. Run DE, get best_params_de.
        # 2. Run NM/Powell on best_params_de.
        
        # We need to restructure slightly to capture params.
        pass

    # Let's rewrite the main loop to properly track params.
    
    # Reset bests
    best_score = 0.0
    best_pts = None
    best_params = None

    def _consider_params(params, pts):
        nonlocal best_score, best_pts, best_params
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_pts = pts.copy()
            best_params = params.copy()

    # Phase 1: DE
    try:
        res_de = differential_evolution(
            _objective_safe, bounds,
            args=(deadline,),
            seed=42, maxiter=100, tol=1e-8,
            popsize=20, mutation=(0.5, 1.0), recombination=0.7,
            polish=False, update_interval=1
        )
        if res_de is not None and np.isfinite(res_de.fun) and res_de.fun < 1e9:
            pts = _decode_symmetric_safe(res_de.x)
            _consider_params(res_de.x, pts)
    except Exception:
        pass

    # Phase 2: Local refinement on best_params
    if best_params is not None:
        # NM
        if time.time() < deadline - 20:
            res_nm = _nm(best_params, deadline, _objective_safe, max_iter=500)
            if res_nm is not None and np.isfinite(res_nm.fun) and res_nm.fun < 1e9:
                pts = _decode_symmetric_safe(res_nm.x)
                _consider_params(res_nm.x, pts)
        
        # Powell
        if time.time() < deadline - 10:
            res_pw = _powell(best_params, deadline, _objective_safe, max_iter=500)
            if res_pw is not None and np.isfinite(res_pw.fun) and res_pw.fun < 1e9:
                pts = _decode_symmetric_safe(res_pw.x)
                _consider_params(res_pw.x, pts)

    # Fallback
    if best_pts is None:
        # Regular 14-gon
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_pts = np.column_stack([np.cos(angles), np.sin(angles)])
    
    final_points = _normalize_unit_area(best_pts)
    
    # Safety check
    try:
        hull = ConvexHull(final_points)
        if hull.volume < 1e-10:
            angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
            final_points = _normalize_unit_area(np.column_stack([np.cos(angles), np.sin(angles)]))
    except Exception:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        final_points = _normalize_unit_area(np.column_stack([np.cos(angles), np.sin(angles)]))
    
    return final_points
