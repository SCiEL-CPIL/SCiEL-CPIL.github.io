# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

def _triangle_areas(pts, tri_idx):
    """Compute all triangle areas vectorized."""
    a = pts[tri_idx[:, 0]]
    b = pts[tri_idx[:, 1]]
    c = pts[tri_idx[:, 2]]
    area = 0.5 * np.abs(
        a[:, 0] * (b[:, 1] - c[:, 1]) +
        b[:, 0] * (c[:, 1] - a[:, 1]) +
        c[:, 0] * (a[:, 1] - b[:, 1])
    )
    return area

def _hull_area(pts):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0

def _get_tri_indices(n):
    idx = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                idx.append((i, j, k))
    return np.array(idx, dtype=int)

def _compute_score(pts, tri_idx):
    """Compute score given points and precomputed triangle indices."""
    areas = _triangle_areas(pts, tri_idx)
    min_area = areas.min()
    hull_area = _hull_area(pts)
    if hull_area <= 1e-9:
        return 0.0
    return min_area / hull_area

def _soft_min_logsumexp(areas, beta):
    """
    Smooth approximation of min(areas) using log-sum-exp.
    As beta -> infinity, this approaches the true min.
    """
    # Use log-sum-exp trick for numerical stability
    # soft_min = (1/beta) * log(sum(exp(-beta * areas)))
    # = -1/beta * logsumexp(-beta * areas)
    # For numerical stability:
    shifted = -beta * areas
    max_shifted = np.max(shifted)
    log_sum = max_shifted + np.log(np.sum(np.exp(shifted - max_shifted)))
    return -log_sum / beta

def _soft_score(pts, tri_idx, beta=50.0):
    """
    Smooth score using soft-min for the triangle areas.
    This is differentiable and suitable for gradient-based optimization.
    """
    areas = _triangle_areas(pts, tri_idx)
    # Avoid log(0) issues
    areas_clipped = np.maximum(areas, 1e-15)
    soft_min = _soft_min_logsumexp(areas_clipped, beta)
    hull_area = _hull_area(pts)
    if hull_area <= 1e-9:
        return 0.0
    return soft_min / hull_area

def _soft_objective(x, n, tri_idx, beta=50.0):
    """Objective function for L-BFGS-B (minimize negative soft score)."""
    pts = x.reshape(n, 2)
    score = _soft_score(pts, tri_idx, beta)
    return -score

def _soft_gradient(x, n, tri_idx, beta=50.0):
    """
    Numerical gradient of the soft objective.
    """
    eps = 1e-7
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_plus = x.copy()
        x_plus[i] += eps
        x_minus = x.copy()
        x_minus[i] -= eps
        f_plus = _soft_objective(x_plus, n, tri_idx, beta)
        f_minus = _soft_objective(x_minus, n, tri_idx, beta)
        grad[i] = (f_plus - f_minus) / (2 * eps)
    return grad

def _regular_polygon(n, radius=1.0, center=(0, 0), angle_offset=0.0):
    angles = angle_offset + np.linspace(0, 2 * np.pi, n, endpoint=False)
    xs = center[0] + radius * np.cos(angles)
    ys = center[1] + radius * np.sin(angles)
    return np.column_stack([xs, ys])

def _generate_lattice_config(n):
    """
    Generate a lattice-based configuration for n points.
    Uses a grid arrangement that tends to have good minimum triangle areas.
    """
    # Try different grid layouts
    best_pts = None
    best_score = -1
    
    tri_idx = _get_tri_indices(n)
    
    # Grid layouts: rows x cols
    layouts = []
    for rows in range(1, n + 1):
        cols = (n + rows - 1) // rows
        if rows * cols >= n:
            layouts.append((rows, cols))
    
    for rows, cols in layouts:
        # Create grid points
        xs = np.linspace(-1, 1, cols) if cols > 1 else np.array([0.0])
        ys = np.linspace(-1, 1, rows) if rows > 1 else np.array([0.0])
        
        grid_pts = []
        for y in ys:
            for x in xs:
                grid_pts.append([x, y])
                if len(grid_pts) == n:
                    break
            if len(grid_pts) == n:
                break
        
        if len(grid_pts) != n:
            continue
        
        pts = np.array(grid_pts)
        
        # Add small jitter to break degeneracies
        np.random.seed(42)
        pts += np.random.normal(0, 0.01, pts.shape)
        
        score = _compute_score(pts, tri_idx)
        if score > best_score:
            best_score = score
            best_pts = pts.copy()
    
    return best_pts if best_pts is not None else _regular_polygon(n)

def _generate_initial_points(partition, tri_idx):
    """Generate initial points based on a partition of 13 into rings."""
    pts = []
    remaining = 13
    ring_counts = list(partition)
    
    for i, count in enumerate(ring_counts):
        if count == 0:
            continue
        if i == 0 and count == 1:
            # Center point
            pts.append([0.0, 0.0])
        else:
            # Distribute remaining rings evenly in radius
            radius = 1.0 / max(1, len(ring_counts) - i) * (i + 1)
            # Add small jitter to break symmetry
            np.random.seed(42 + i)
            angle_offset = np.random.uniform(0, 2 * np.pi)
            poly = _regular_polygon(count, radius, (0, 0), angle_offset)
            pts.extend(poly)
            
    pts = np.array(pts)
    if len(pts) != 13:
        # Fallback to regular 13-gon
        pts = _regular_polygon(13)
    return pts

def _coordinate_descent(pts, tri_idx, max_iter=500, step_size=0.02):
    """
    Coordinate descent with line search on each coordinate.
    """
    n = len(pts)
    best_score = _compute_score(pts, tri_idx)
    pts = pts.copy()
    
    for it in range(max_iter):
        improved = False
        for i in range(n):
            for d in range(2):
                # Try to improve coordinate (i, d)
                best_val = pts[i, d]
                best_local_score = best_score
                
                # Golden section search for best step
                a = pts[i, d] - step_size * 10
                b = pts[i, d] + step_size * 10
                gr = (np.sqrt(5) - 1) / 2  # Golden ratio conjugate
                
                c = b - gr * (b - a)
                d_val = a + gr * (b - a)
                
                f_c = _compute_score(pts.copy().__class__(pts), tri_idx)  # placeholder
                pts_c = pts.copy()
                pts_c[i, d] = c
                f_c = _compute_score(pts_c, tri_idx)
                
                pts_d = pts.copy()
                pts_d[i, d] = d_val
                f_d = _compute_score(pts_d, tri_idx)
                
                for _ in range(20):
                    if f_c < f_d:
                        b = d_val
                        d_val = c
                        f_d = f_c
                        c = b - gr * (b - a)
                        pts_c = pts.copy()
                        pts_c[i, d] = c
                        f_c = _compute_score(pts_c, tri_idx)
                    else:
                        a = c
                        c = d_val
                        f_c = f_d
                        d_val = a + gr * (b - a)
                        pts_d = pts.copy()
                        pts_d[i, d] = d_val
                        f_d = _compute_score(pts_d, tri_idx)
                
                best_x = (a + b) / 2
                pts_best = pts.copy()
                pts_best[i, d] = best_x
                best_local_score = _compute_score(pts_best, tri_idx)
                
                if best_local_score > best_score + 1e-10:
                    pts[i, d] = best_x
                    best_score = best_local_score
                    improved = True
                    
        if not improved:
            break
            
    return pts, best_score

def _lbfgs_optimize(pts, tri_idx, n, time_limit=20.0):
    """
    L-BFGS-B optimization with soft-min surrogate.
    """
    start_time = time.time()
    x0 = pts.flatten()
    
    best_pts = pts.copy()
    best_score = _compute_score(best_pts, tri_idx)
    
    # Try multiple beta values for soft-min
    for beta in [30.0, 50.0, 100.0]:
        if time.time() - start_time > time_limit:
            break
        
        try:
            result = minimize(
                _soft_objective,
                x0,
                args=(n, tri_idx, beta),
                method='L-BFGS-B',
                jac=_soft_gradient,
                options={
                    'maxiter': 200,
                    'maxfun': 5000,
                    'ftol': 1e-10,
                    'gtol': 1e-8,
                    'maxls': 50
                }
            )
            
            if result.success or result.fun < 0:
                pts_opt = result.x.reshape(n, 2)
                score_opt = _compute_score(pts_opt, tri_idx)
                if score_opt > best_score:
                    best_score = score_opt
                    best_pts = pts_opt.copy()
        except Exception:
            pass
    
    return best_pts, best_score

def heilbronn_convex13() -> np.ndarray:
    n = 13
    tri_idx = _get_tri_indices(n)
    
    deadline = time.time() + 580  # Safety margin
    
    best_pts = None
    best_score = -1.0
    
    # 1. Generate multiple initial configurations
    np.random.seed(42)
    
    # Regular 13-gon
    pts_reg = _regular_polygon(13)
    score_reg = _compute_score(pts_reg, tri_idx)
    if score_reg > best_score:
        best_score = score_reg
        best_pts = pts_reg.copy()
    
    # Lattice configuration
    pts_lat = _generate_lattice_config(n)
    score_lat = _compute_score(pts_lat, tri_idx)
    if score_lat > best_score:
        best_score = score_lat
        best_pts = pts_lat.copy()
    
    # Ring-based configurations
    partitions = [
        (1, 12),
        (3, 10),
        (4, 9),
        (5, 8),
        (6, 7),
        (1, 3, 9),
        (1, 5, 7),
        (1, 6, 6),
        (2, 5, 6),
        (3, 4, 6),
        (2, 2, 9),
    ]
    
    for part in partitions:
        if time.time() > deadline - 100:
            break
        pts_init = _generate_initial_points(part, tri_idx)
        score_init = _compute_score(pts_init, tri_idx)
        if score_init > best_score:
            best_score = score_init
            best_pts = pts_init.copy()
    
    # 2. L-BFGS-B optimization with soft-min surrogate
    if best_pts is not None and time.time() < deadline - 60:
        time_for_lbfgs = min(40.0, deadline - time.time() - 20)
        pts_lbfgs, score_lbfgs = _lbfgs_optimize(best_pts, tri_idx, n, time_limit=time_for_lbfgs)
        if score_lbfgs > best_score:
            best_score = score_lbfgs
            best_pts = pts_lbfgs.copy()
    
    # 3. Coordinate descent refinement
    if time.time() < deadline - 30:
        pts_cd, score_cd = _coordinate_descent(best_pts, tri_idx, max_iter=300, step_size=0.03)
        if score_cd > best_score:
            best_score = score_cd
            best_pts = pts_cd.copy()
    
    # 4. Another round of L-BFGS-B from the improved configuration
    if time.time() < deadline - 20:
        time_for_lbfgs2 = min(20.0, deadline - time.time() - 10)
        pts_lbfgs2, score_lbfgs2 = _lbfgs_optimize(best_pts, tri_idx, n, time_limit=time_for_lbfgs2)
        if score_lbfgs2 > best_score:
            best_score = score_lbfgs2
            best_pts = pts_lbfgs2.copy()
    
    # 5. Final coordinate descent polish with tighter step
    if time.time() < deadline - 10:
        pts_final, score_final = _coordinate_descent(best_pts, tri_idx, max_iter=500, step_size=0.01)
        if score_final > best_score:
            best_score = score_final
            best_pts = pts_final.copy()
    
    # Ensure finite
    if not np.all(np.isfinite(best_pts)):
        return _regular_polygon(13)
    
    return best_pts

# EVOLVE-BLOCK-END
