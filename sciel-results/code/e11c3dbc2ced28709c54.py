import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 14

# Precompute indices for all combinations of 3 points
_indices = np.arange(N)
IDX_I, IDX_J, IDX_K = np.meshgrid(_indices, _indices, _indices, indexing='ij')
mask = (IDX_I < IDX_J) & (IDX_J < IDX_K)
IDX_I = IDX_I[mask].flatten()
IDX_J = IDX_J[mask].flatten()
IDX_K = IDX_K[mask].flatten()

def _all_areas(points):
    pi = points[IDX_I]
    pj = points[IDX_J]
    pk = points[IDX_K]
    v1x = pj[:, 0] - pi[:, 0]
    v1y = pj[:, 1] - pi[:, 1]
    v2x = pk[:, 0] - pi[:, 0]
    v2y = pk[:, 1] - pi[:, 1]
    areas = 0.5 * np.abs(v1x * v2y - v1y * v2x)
    return areas

def _compute_min_triangle_area(points):
    return np.min(_all_areas(points))

def _compute_hull_area(points):
    try:
        if np.allclose(points[:, 0], points[0, 0]) or np.allclose(points[:, 1], points[0, 1]):
            return 0.0
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    hull_area = _compute_hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    min_area = _compute_min_triangle_area(points)
    return min_area / hull_area

def _objective_soft(flat, beta=50.0):
    """Soft-min objective for full 28D parameterization."""
    pts = flat.reshape(N, 2)
    if np.any(~np.isfinite(pts)):
        return 1e10
    areas = _all_areas(pts)
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
    hull_area = _compute_hull_area(pts)
    if hull_area < 1e-10:
        return 1e10
    return hull_area / soft_min

def _objective_hard(flat):
    """Hard objective for full 28D parameterization."""
    pts = flat.reshape(N, 2)
    return -_score(pts)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    best_points = None
    best_score = -np.inf

    def update_best(pts, score):
        nonlocal best_points, best_score
        if score > best_score:
            best_score = score
            best_points = pts.copy()

    def refine(pts_init):
        """Refine a set of points using soft-min then hard-min."""
        flat_init = pts_init.flatten()
        
        # Step 1: L-BFGS-B on soft objective to get into a good basin
        try:
            res_soft = minimize(
                lambda x: _objective_soft(x, beta=50.0), 
                flat_init, 
                method='L-BFGS-B', 
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            pts_soft = res_soft.x.reshape(N, 2)
        except Exception:
            pts_soft = pts_init

        # Step 2: Nelder-Mead on hard objective for precision
        try:
            res_hard = minimize(
                _objective_hard, 
                pts_soft.flatten(), 
                method='Nelder-Mead', 
                options={'maxiter': 500, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True}
            )
            pts_hard = res_hard.x.reshape(N, 2)
            score = -res_hard.fun
            return pts_hard, score
        except Exception:
            score = _score(pts_soft)
            return pts_soft, score

    # Generate initial candidates
    candidates = []
    
    # 1. Regular 14-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    regular = np.column_stack((np.cos(angles), np.sin(angles)))
    candidates.append(regular)

    # 2. Jittered 14-gons
    rng = np.random.default_rng(42)
    for i in range(8):
        jitter_angles = angles + rng.normal(0, 0.1, N)
        jitter_radii = 1.0 + rng.normal(0, 0.1, N)
        jittered = np.column_stack((np.cos(jitter_angles) * jitter_radii, np.sin(jitter_angles) * jitter_radii))
        candidates.append(jittered)

    # 3. Grid-like structures
    # Create a 4x4 grid, center it, scale it, and pick 14 points
    x = np.linspace(-1, 1, 4)
    y = np.linspace(-1, 1, 4)
    xx, yy = np.meshgrid(x, y)
    grid_pts = np.column_stack((xx.flatten(), yy.flatten()))
    perm = rng.permutation(16)
    grid_subset = grid_pts[perm[:14]]
    candidates.append(grid_subset)
    
    # More grid variations
    for i in range(4):
        x = np.linspace(-1.2, 1.2, 4) + rng.normal(0, 0.1, 4)
        y = np.linspace(-1.2, 1.2, 4) + rng.normal(0, 0.1, 4)
        xx, yy = np.meshgrid(x, y)
        grid_pts = np.column_stack((xx.flatten(), yy.flatten()))
        perm = rng.permutation(16)
        grid_subset = grid_pts[perm[:14]]
        candidates.append(grid_subset)

    # 4. Random points in unit disk
    for i in range(5):
        angles_r = rng.uniform(0, 2 * np.pi, N)
        radii_r = np.sqrt(rng.uniform(0.5, 1.0, N))
        random_pts = np.column_stack((np.cos(angles_r) * radii_r, np.sin(angles_r) * radii_r))
        candidates.append(random_pts)

    # Evaluate candidates
    for i, cand in enumerate(candidates):
        if time.time() > deadline - 15.0:
            break
            
        # Quick score check
        s = _score(cand)
        if s > best_score:
            update_best(cand, s)
            
        # Refine
        refined_pts, refined_score = refine(cand)
        if refined_score > best_score:
            update_best(refined_pts, refined_score)

    # Final polish of the best point found
    if best_points is not None and time.time() < deadline - 5.0:
        # Use a very tight Nelder-Mead on the best point
        try:
            res_final = minimize(
                _objective_hard, 
                best_points.flatten(), 
                method='Nelder-Mead', 
                options={'maxiter': 2000, 'xatol': 1e-11, 'fatol': 1e-13, 'adaptive': True}
            )
            pts_final = res_final.x.reshape(N, 2)
            score_final = -res_final.fun
            if score_final > best_score:
                update_best(pts_final, score_final)
        except Exception:
            pass

    if best_points is None:
        # Fallback
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack((np.cos(angles), np.sin(angles)))

    return best_points
