"""Heilbronn convex regions n=14: maximize min triangle area / convex hull area."""
import time
import numpy as np
from scipy.optimize import minimize

N = 14

# EVOLVE-BLOCK-START
def _triangle_areas(pts):
    """Compute areas of all C(N,3) triangles. Returns array of shape (M,)."""
    n = len(pts)
    areas = np.empty((n * (n - 1) * (n - 2)) // 6)
    idx = 0
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                ax, ay = pts[i]
                bx, by = pts[j]
                cx, cy = pts[k]
                area = 0.5 * abs((bx - ax) * (cy - ay) - (cx - ax) * (by - ay))
                areas[idx] = area
                idx += 1
    return areas


def _min_triangle_area(pts):
    return np.min(_triangle_areas(pts))


def _convex_hull_area(pts):
    """Compute convex hull area using scipy."""
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(pts)
        return hull.volume  # In 2D, volume is area
    except Exception:
        # Fallback: shoelace on sorted points (not correct but safe)
        return 1.0


def _score(pts):
    """Score = min triangle area / convex hull area."""
    mta = _min_triangle_area(pts)
    cha = _convex_hull_area(pts)
    if cha <= 1e-12:
        return 0.0
    return mta / cha


def _soft_min_triangle_area(pts, beta):
    """
    Soft-min surrogate: -log(sum(exp(-beta * area_i))) / beta
    This is a smooth approximation to min(area_i).
    """
    areas = _triangle_areas(pts)
    # Numerical stability: subtract max
    shifted = -beta * areas
    max_shifted = np.max(shifted)
    log_sum = max_shifted + np.log(np.sum(np.exp(shifted - max_shifted)))
    return -log_sum / beta


def _negative_soft_min(pts, beta):
    """Negative of soft-min triangle area (to maximize via minimization)."""
    return -_soft_min_triangle_area(pts, beta)


def _normalize_to_unit_area(pts):
    """Normalize point set so convex hull has unit area, centered at origin."""
    cha = _convex_hull_area(pts)
    if cha <= 1e-12:
        return pts
    centroid = np.mean(pts, axis=0)
    pts_centered = pts - centroid
    pts_scaled = pts_centered / np.sqrt(cha)
    return pts_scaled


def _two_ring_parameterization(params):
    """
    Parameterize 14 points as two rings:
    - Inner ring: 7 points on a circle of radius r1, angles offset
    - Outer ring: 7 points on a circle of radius r2, angles offset
    params = [r1, r2, angle_offset1, angle_offset2, rotation1, rotation2]
    """
    r1 = params[0]
    r2 = params[1]
    off1 = params[2]
    off2 = params[3]
    rot1 = params[4]
    rot2 = params[5]
    
    angles1 = np.linspace(0, 2 * np.pi, 7, endpoint=False) + off1 + rot1
    angles2 = np.linspace(0, 2 * np.pi, 7, endpoint=False) + off2 + rot2
    
    x1 = r1 * np.cos(angles1)
    y1 = r1 * np.sin(angles1)
    x2 = r2 * np.cos(angles2)
    y2 = r2 * np.sin(angles2)
    
    pts = np.column_stack([np.concatenate([x1, x2]), np.concatenate([y1, y2])])
    return pts


def _optimize_from_initial(pts0, deadline, max_iter=500):
    """
    Run L-BFGS-B optimization on the soft-min surrogate.
    """
    # Flatten points
    x0 = pts0.flatten()
    
    best_x = x0.copy()
    best_score = _score(pts0)
    
    # Start with moderate beta, increase for polish
    betas = [10.0, 50.0, 200.0, 1000.0]
    
    for beta in betas:
        if time.time() > deadline - 5:
            break
        
        def objective(x):
            pts = x.reshape(-1, 2)
            # Normalize to keep convex hull area ~1 to stabilize
            pts_norm = _normalize_to_unit_area(pts)
            return -_soft_min_triangle_area(pts_norm, beta)
        
        def gradient(x):
            pts = x.reshape(-1, 2)
            pts_norm = _normalize_to_unit_area(pts)
            # Numerical gradient (expensive but robust)
            eps = 1e-6
            grad = np.zeros_like(x)
            f0 = objective(x)
            for i in range(len(x)):
                xp = x.copy()
                xp[i] += eps
                grad[i] = (objective(xp) - f0) / eps
            return grad
        
        try:
            result = minimize(
                objective, best_x, method='L-BFGS-B',
                jac=None,  # Use numerical gradient via L-BFGS-B's internal
                options={'maxiter': max_iter, 'gtol': 1e-8, 'ftol': 1e-12}
            )
            if result.fun < objective(best_x):
                best_x = result.x
                best_score = _score(result.x.reshape(-1, 2))
        except Exception:
            pass
    
    return best_x.reshape(-1, 2), best_score


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    best_pts = None
    best_score = -1.0
    
    # Try multiple random initializations with two-ring structure
    np.random.seed(N)
    
    n_trials = 20  # Will be limited by time
    
    for trial in range(n_trials):
        if time.time() > deadline - 10:
            break
        
        # Generate initial two-ring configuration
        r1 = np.random.uniform(0.3, 0.7)
        r2 = np.random.uniform(0.8, 1.2)
        off1 = np.random.uniform(0, 2 * np.pi)
        off2 = np.random.uniform(0, 2 * np.pi)
        rot1 = np.random.uniform(0, 2 * np.pi)
        rot2 = np.random.uniform(0, 2 * np.pi)
        
        params = np.array([r1, r2, off1, off2, rot1, rot2])
        pts_init = _two_ring_parameterization(params)
        pts_init = _normalize_to_unit_area(pts_init)
        
        # Also try a perturbed regular polygon initialization
        if trial % 2 == 1:
            # Regular 14-gon with random perturbations
            angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
            radii = np.ones(N) + np.random.normal(0, 0.1, N)
            pts_init = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
            pts_init = _normalize_to_unit_area(pts_init)
        
        # Optimize
        pts_opt, score_opt = _optimize_from_initial(pts_init, deadline, max_iter=200)
        
        if score_opt > best_score:
            best_score = score_opt
            best_pts = pts_opt.copy()
    
    # Final polish on best solution with higher beta
    if best_pts is not None and time.time() < deadline - 5:
        x_best = best_pts.flatten()
        
        def objective_polish(x):
            pts = x.reshape(-1, 2)
            pts_norm = _normalize_to_unit_area(pts)
            return -_soft_min_triangle_area(pts_norm, 5000.0)
        
        try:
            result = minimize(
                objective_polish, x_best, method='L-BFGS-B',
                options={'maxiter': 300, 'gtol': 1e-10, 'ftol': 1e-14}
            )
            if result.fun < objective_polish(x_best):
                best_pts = result.x.reshape(-1, 2)
                best_score = _score(best_pts)
        except Exception:
            pass
    
    if best_pts is None:
        # Fallback: regular polygon
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_pts = np.column_stack([np.cos(angles), np.sin(angles)])
        best_pts = _normalize_to_unit_area(best_pts)
    
    return best_pts
# EVOLVE-BLOCK-END
