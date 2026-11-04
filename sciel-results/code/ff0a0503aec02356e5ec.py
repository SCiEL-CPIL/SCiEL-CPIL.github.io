import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize, differential_evolution

N = 14

def _min_triangle_area_vec(points):
    """Vectorized minimum triangle area."""
    n = len(points)
    if n < 3:
        return 0.0
    
    # Generate indices for all combinations i < j < k
    i, j, k = np.indices((n, n, n))
    mask = (i < j) & (j < k)
    i = i[mask].flatten()
    j = j[mask].flatten()
    k = k[mask].flatten()
    
    pi = points[i]
    pj = points[j]
    pk = points[k]
    
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def _convex_hull_area(points):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """Score = min triangle area / convex hull area."""
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-9:
        return 0.0
    min_area = _min_triangle_area_vec(points)
    return min_area / hull_area

def _build_ring_config(radii, angles_per_ring, phase_offsets):
    """
    Build point set from concentric rings.
    radii: array of ring radii (outermost first)
    angles_per_ring: list of counts per ring
    phase_offsets: list of angular offsets per ring
    """
    pts = []
    for r, n_ang, phase in zip(radii, angles_per_ring, phase_offsets):
        angles = np.linspace(0, 2 * np.pi, n_ang, endpoint=False) + phase
        ring_pts = np.column_stack([r * np.cos(angles), r * np.sin(angles)])
        pts.append(ring_pts)
    return np.vstack(pts)

def _param_to_points(params):
    """
    Convert parameter vector to points.
    Parameters: [r1, r2, r3, ..., phase1, phase2, phase3, ...]
    We use 3 rings: 6+5+3 or 5+5+4 or other splits.
    Let's try multiple ring split patterns.
    """
    # This function will be specialized per ring configuration
    raise NotImplementedError

def _optimize_ring_config(n1, n2, n3, r_init, phase_init, time_limit=15.0):
    """
    Optimize a 3-ring configuration with n1+n2+n3=N points.
    Parameters: [r1, r2, r3, phase1, phase2, phase3]
    r1 >= r2 >= r3 > 0, phases in [0, 2*pi)
    """
    start_time = time.time()
    
    def build_from_params(x):
        r1, r2, r3, p1, p2, p3 = x
        # Ensure ordering and positivity
        r1 = max(0.5, r1)
        r2 = max(0.1, min(r1, r2))
        r3 = max(0.01, min(r2, r3))
        p1 = p1 % (2 * np.pi)
        p2 = p2 % (2 * np.pi)
        p3 = p3 % (2 * np.pi)
        return _build_ring_config([r1, r2, r3], [n1, n2, n3], [p1, p2, p3])
    
    def neg_score_params(x):
        pts = build_from_params(x)
        return -_score(pts)
    
    x0 = np.array([r_init[0], r_init[1], r_init[2], phase_init[0], phase_init[1], phase_init[2]])
    best_score = _score(build_from_params(x0))
    best_x = x0.copy()
    
    bounds = [(0.5, 2.0), (0.1, 2.0), (0.01, 2.0), (0, 2*np.pi), (0, 2*np.pi), (0, 2*np.pi)]
    
    # Try multiple local optimizations from different starts
    starts = [x0]
    # Add some perturbed starts
    rng = np.random.RandomState(42 + n1*100 + n2*10 + n3)
    for _ in range(3):
        x_pert = x0 + rng.normal(0, 0.1, size=6)
        x_pert[0] = max(0.5, x_pert[0])
        x_pert[1] = max(0.1, min(x_pert[1], x_pert[0]))
        x_pert[2] = max(0.01, min(x_pert[2], x_pert[1]))
        x_pert[3:] = x_pert[3:] % (2*np.pi)
        starts.append(x_pert)
    
    for xi in starts:
        if time.time() > start_time + time_limit - 2:
            break
        try:
            res = minimize(
                neg_score_params, xi, method='Nelder-Mead',
                bounds=bounds,
                options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            s = -res.fun
            if s > best_score:
                best_score = s
                best_x = res.x.copy()
        except Exception:
            pass
    
    best_pts = build_from_params(best_x)
    return best_score, best_pts

def _optimize_two_ring_config(n1, n2, r_init, phase_init, time_limit=15.0):
    """
    Optimize a 2-ring configuration with n1+n2=N points.
    Parameters: [r1, r2, phase1, phase2]
    """
    start_time = time.time()
    
    def build_from_params(x):
        r1, r2, p1, p2 = x
        r1 = max(0.5, r1)
        r2 = max(0.05, min(r1, r2))
        p1 = p1 % (2 * np.pi)
        p2 = p2 % (2 * np.pi)
        return _build_ring_config([r1, r2], [n1, n2], [p1, p2])
    
    def neg_score_params(x):
        pts = build_from_params(x)
        return -_score(pts)
    
    x0 = np.array([r_init[0], r_init[1], phase_init[0], phase_init[1]])
    best_score = _score(build_from_params(x0))
    best_x = x0.copy()
    
    bounds = [(0.5, 2.0), (0.05, 2.0), (0, 2*np.pi), (0, 2*np.pi)]
    
    starts = [x0]
    rng = np.random.RandomState(42 + n1*100 + n2)
    for _ in range(4):
        x_pert = x0 + rng.normal(0, 0.1, size=4)
        x_pert[0] = max(0.5, x_pert[0])
        x_pert[1] = max(0.05, min(x_pert[1], x_pert[0]))
        x_pert[2:] = x_pert[2:] % (2*np.pi)
        starts.append(x_pert)
    
    for xi in starts:
        if time.time() > start_time + time_limit - 2:
            break
        try:
            res = minimize(
                neg_score_params, xi, method='Nelder-Mead',
                bounds=bounds,
                options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            s = -res.fun
            if s > best_score:
                best_score = s
                best_x = res.x.copy()
        except Exception:
            pass
    
    best_pts = build_from_params(best_x)
    return best_score, best_pts

def _local_optimize_full(points, time_limit=10.0):
    """Full 28D Nelder-Mead refinement as fallback."""
    start_time = time.time()
    x0 = points.flatten()
    
    def obj(x):
        pts = x.reshape(-1, 2)
        return -_score(pts)
    
    best_score = _score(points)
    best_pts = points.copy()
    
    try:
        res = minimize(
            obj, x0, method='Nelder-Mead',
            options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
        )
        if time.time() - start_time < time_limit:
            if res.fun < -best_score:
                best_score = -res.fun
                best_pts = res.x.reshape(-1, 2).copy()
    except Exception:
        pass
        
    return best_score, best_pts

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    best_overall_score = 0.0
    best_overall_points = None
    
    # Strategy: Explore symmetry-reduced ring configurations first,
    # then use remaining time for full-space refinement.
    
    # Ring configurations to try: (n1, n2, n3) splits
    ring_configs_3 = [
        (6, 5, 3),
        (5, 5, 4),
        (7, 4, 3),
        (5, 4, 5),
        (4, 5, 5),
        (6, 4, 4),
        (4, 6, 4),
        (7, 5, 2),
        (2, 6, 6),
    ]
    
    # Two-ring configurations
    ring_configs_2 = [
        (7, 7),
        (6, 8),
        (8, 6),
        (5, 9),
        (9, 5),
    ]
    
    # Initial parameters for each config
    # Outer ring radius ~1, inner rings smaller
    base_r = [1.0, 0.6, 0.3]
    base_phases = [0.0, np.pi/7, np.pi/5]
    
    # Try 3-ring configs
    for (n1, n2, n3) in ring_configs_3:
        if time.time() > deadline - 40:
            break
        
        # Try multiple initial radius ratios
        for r_ratio in [0.5, 0.6, 0.7, 0.8]:
            if time.time() > deadline - 40:
                break
            r_init = [1.0, 0.5 + 0.3*r_ratio, 0.2 + 0.3*r_ratio]
            # Ensure decreasing
            r_init[1] = min(r_init[1], r_init[0] * 0.9)
            r_init[2] = min(r_init[2], r_init[1] * 0.9)
            
            # Try a few phase offsets
            for phase_shift in [0.0, np.pi/(2*n1), np.pi/n1]:
                if time.time() > deadline - 40:
                    break
                phase_init = [0.0, phase_shift, 2*phase_shift]
                
                s, p = _optimize_ring_config(n1, n2, n3, r_init, phase_init, time_limit=5.0)
                if s > best_overall_score:
                    best_overall_score = s
                    best_overall_points = p
    
    # Try 2-ring configs
    for (n1, n2) in ring_configs_2:
        if time.time() > deadline - 25:
            break
        
        for r_ratio in [0.4, 0.5, 0.6, 0.7, 0.8]:
            if time.time() > deadline - 25:
                break
            r_init = [1.0, 0.3 + 0.6*r_ratio]
            r_init[1] = min(r_init[1], r_init[0] * 0.95)
            
            for phase_shift in [0.0, np.pi/(2*n1), np.pi/n1, np.pi/(2*n2)]:
                if time.time() > deadline - 25:
                    break
                phase_init = [0.0, phase_shift]
                
                s, p = _optimize_two_ring_config(n1, n2, r_init, phase_init, time_limit=4.0)
                if s > best_overall_score:
                    best_overall_score = s
                    best_overall_points = p
    
    # Final: full-space local optimization from best found
    if best_overall_points is not None and time.time() < deadline - 15:
        s, p = _local_optimize_full(best_overall_points, time_limit=12.0)
        if s > best_overall_score:
            best_overall_score = s
            best_overall_points = p
    
    # Fallback: regular polygon
    if best_overall_points is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_overall_points = np.column_stack([np.cos(angles), np.sin(angles)])
        best_overall_score = _score(best_overall_points)
    
    return best_overall_points
