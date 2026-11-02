"""Heilbronn 11 triangles: Active-Set Derivative-Free Optimization."""
import numpy as np
import time
import itertools
from scipy.optimize import minimize

N = 11
S3 = np.sqrt(3)
S3H = S3 / 2
TRI_AREA = S3 / 4

DEADLINE = 145

# Precompute all triplets for N=11 points
_TRIPLETS = list(itertools.combinations(range(N), 3))
_T_I = np.array([t[0] for t in _TRIPLETS])
_T_J = np.array([t[1] for t in _TRIPLETS])
_T_K = np.array([t[2] for t in _TRIPLETS])
_N_TRIPLETS = len(_TRIPLETS)


def _bary_to_cart(bary):
    """Convert barycentric coordinates (u, v) to Cartesian (x, y)."""
    u = bary[:, 0]
    v = bary[:, 1]
    x = 0.5 - 0.5 * u + 0.5 * v
    y = S3H * (1.0 - u - v)
    return np.column_stack([x, y])


def _cart_to_bary(pts):
    """Convert Cartesian to barycentric coordinates."""
    x = pts[:, 0]
    y = pts[:, 1]
    u = 1.0 - x - y / (2.0 * S3H)
    v = x - y / (2.0 * S3H)
    return np.column_stack([u, v])


def _min_triangle_area(pts):
    """Compute the minimum area of any triangle formed by three points (vectorized)."""
    pi = pts[_T_I]
    pj = pts[_T_J]
    pk = pts[_T_K]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    cross = dx1 * dy2 - dy1 * dx2
    areas = np.abs(cross) * 0.5
    return np.min(areas)


def _get_min_areas_and_indices(pts):
    """Get all triangle areas and indices of the minimum ones."""
    pi = pts[_T_I]
    pj = pts[_T_J]
    pk = pts[_T_K]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    cross = dx1 * dy2 - dy1 * dx2
    areas = np.abs(cross) * 0.5
    min_area = np.min(areas)
    # Find indices of triangles with area close to min (within 1e-6)
    min_indices = np.where(areas <= min_area + 1e-6)[0]
    return min_area, min_indices, areas


def _project_to_triangle(pts):
    """Project points into the equilateral triangle."""
    pts = pts.copy()
    for i in range(len(pts)):
        x, y = pts[i]
        y = max(0.0, min(S3H, y))
        x_min = y / S3
        x_max = 1 - y / S3
        x = max(x_min, min(x_max, x))
        pts[i] = [x, y]
    return pts


def _active_set_optimize(pts, max_iter=1000, step_size=0.05):
    """
    Derivative-free active-set local search.
    Iteratively:
    1. Find the minimum area triangle(s).
    2. Identify the points involved in these triangles (active set).
    3. Try small perturbations of active points to increase the minimum area.
    4. Accept if improvement, else try smaller step or different direction.
    """
    current_pts = pts.copy()
    current_area = _min_triangle_area(current_pts)
    step = step_size
    no_improve_count = 0
    
    for it in range(max_iter):
        min_area, min_indices, _ = _get_min_areas_and_indices(current_pts)
        
        # Get unique point indices involved in min triangles
        active_points = set()
        for idx in min_indices:
            active_points.add(_T_I[idx])
            active_points.add(_T_J[idx])
            active_points.add(_T_K[idx])
        
        if len(active_points) == 0:
            break
            
        improved = False
        # Try perturbing each active point in random directions
        for p_idx in active_points:
            for _ in range(10):  # 10 random directions per point
                # Random direction in 2D
                angle = np.random.uniform(0, 2 * np.pi)
                direction = np.array([np.cos(angle), np.sin(angle)])
                
                new_pts = current_pts.copy()
                new_pts[p_idx] = current_pts[p_idx] + step * direction
                new_pts = _project_to_triangle(new_pts)
                
                new_area = _min_triangle_area(new_pts)
                if new_area > current_area + 1e-12:
                    current_pts = new_pts
                    current_area = new_area
                    improved = True
                    no_improve_count = 0
                    break
            if improved:
                break
        
        if not improved:
            no_improve_count += 1
            if no_improve_count > 20:
                # Reduce step size
                step *= 0.5
                no_improve_count = 0
                if step < 1e-6:
                    break
        else:
            # Slightly increase step if improving
            step = min(step * 1.05, step_size)
    
    return current_pts, current_area


def _structured_init():
    """
    Create a high-quality structured initial configuration.
    Uses symmetry: 3 corners, 3 midpoints of sides, 1 center, and 4 symmetric interior points.
    """
    # Vertices
    p0 = np.array([0.0, 0.0])
    p1 = np.array([1.0, 0.0])
    p2 = np.array([0.5, S3H])
    
    # Midpoints of sides
    p3 = np.array([0.5, 0.0])       # Mid of bottom
    p4 = np.array([0.75, S3H/2])    # Mid of right
    p5 = np.array([0.25, S3H/2])    # Mid of left
    
    # Center
    p6 = np.array([0.5, S3H/3])
    
    # 4 symmetric interior points
    # Use barycentric coordinates for symmetry
    # Points at (1/3, 1/3, 1/3) is center, already used.
    # Try points near the corners but inside
    # Barycentric (u, v) where u is weight of (0,0), v is weight of (1,0), w=1-u-v is weight of (0.5, S3H)
    # Let's place them symmetrically around the center
    
    # Point 7: closer to bottom-left
    b7 = np.array([0.4, 0.3])  # u=0.4, v=0.3, w=0.3
    p7 = _bary_to_cart(b7.reshape(1, 2))[0]
    
    # Point 8: closer to bottom-right
    b8 = np.array([0.3, 0.4])
    p8 = _bary_to_cart(b8.reshape(1, 2))[0]
    
    # Point 9: closer to top
    b9 = np.array([0.3, 0.3])
    p9 = _bary_to_cart(b9.reshape(1, 2))[0]
    
    # Point 10: another interior point for balance
    b10 = np.array([0.35, 0.35])
    p10 = _bary_to_cart(b10.reshape(1, 2))[0]
    
    pts = np.array([p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10])
    return _project_to_triangle(pts)


def _random_init(rng):
    """Random initialization inside triangle."""
    pts = np.zeros((N, 2))
    for i in range(N):
        u = rng.random()
        v = rng.random()
        if u + v > 1:
            u = 1 - u
            v = 1 - v
        pts[i] = _bary_to_cart(np.array([[u, v]]))[0]
    return _project_to_triangle(pts)


def construct():
    start_time = time.time()
    rng = np.random.default_rng(42)
    
    best_pts = None
    best_area = 0.0
    
    def _elapsed_ok(margin):
        return (time.time() - start_time) < (DEADLINE - margin)
    
    def _update_best(pts):
        nonlocal best_pts, best_area
        pts = _project_to_triangle(pts.copy())
        area = _min_triangle_area(pts)
        if area > best_area:
            best_area = area
            best_pts = pts.copy()
    
    # Initial structured configuration
    init_pts = _structured_init()
    _update_best(init_pts)
    
    # Phase 1: Active-set optimization from structured init
    if _elapsed_ok(30):
        opt_pts, opt_area = _active_set_optimize(init_pts, max_iter=2000, step_size=0.1)
        _update_best(opt_pts)
    
    # Phase 2: Multiple random restarts with active-set optimization
    n_restarts = 20
    for r in range(n_restarts):
        if not _elapsed_ok(30):
            break
        
        # Mix of random and perturbed structured
        if r % 3 == 0:
            base_pts = _random_init(rng)
        elif r % 3 == 1:
            base_pts = init_pts + rng.normal(0, 0.1, size=(N, 2))
            base_pts = _project_to_triangle(base_pts)
        else:
            base_pts = best_pts + rng.normal(0, 0.05, size=(N, 2))
            base_pts = _project_to_triangle(base_pts)
        
        # Active-set optimization
        opt_pts, opt_area = _active_set_optimize(base_pts, max_iter=1000, step_size=0.05)
        _update_best(opt_pts)
    
    # Phase 3: Fine-tuning with smaller step sizes on best solution
    if best_pts is not None and _elapsed_ok(15):
        for step in [0.02, 0.01, 0.005, 0.002]:
            if not _elapsed_ok(10):
                break
            opt_pts, opt_area = _active_set_optimize(best_pts, max_iter=500, step_size=step)
            _update_best(opt_pts)
    
    # Phase 4: Final Nelder-Mead polish on the best solution
    if best_pts is not None and _elapsed_ok(10):
        def _objective(flat_params):
            pts = flat_params.reshape(-1, 2)
            pts = _project_to_triangle(pts)
            return -_min_triangle_area(pts)
        
        try:
            res = minimize(
                _objective,
                best_pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            polished_pts = _project_to_triangle(res.x.reshape(-1, 2))
            _update_best(polished_pts)
        except Exception:
            pass
    
    if best_pts is None:
        best_pts = _random_init(rng)
    
    return best_pts
