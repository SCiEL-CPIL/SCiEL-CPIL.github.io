# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
import itertools
from scipy.optimize import minimize

_START_TIME = time.time()
_DEADLINE = _START_TIME + 590.0  # 10s safety margin for 600s limit

def _min_triangle_area(points):
    """
    Compute the minimum area among all triangles formed by triples of points.
    """
    n = len(points)
    if n < 3:
        return 0.0
    
    # Generate all combinations of 3 indices
    indices = np.array(list(itertools.combinations(range(n), 3)))
    
    # Extract points
    p1 = points[indices[:, 0]]
    p2 = points[indices[:, 1]]
    p3 = points[indices[:, 2]]
    
    # Calculate areas: 0.5 * | (p2-p1) x (p3-p1) |
    v1 = p2 - p1
    v2 = p3 - p1
    # Cross product in 2D is a scalar: v1_x * v2_y - v1_y * v2_x
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)

def _hull_area(points):
    """
    Compute the area of the convex hull of the points.
    """
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """
    Calculate the metric: A_min / A_hull.
    """
    a_hull = _hull_area(points)
    if a_hull <= 1e-12:
        return 0.0
    a_min = _min_triangle_area(points)
    return a_min / a_hull

def _energy(points):
    """
    Compute the repulsive energy of the point configuration.
    We use a soft-min of the pairwise distances to encourage a uniform distribution.
    Specifically, we minimize -log(sum(exp(-d^2 / T))) which is equivalent to maximizing the soft-min of distances.
    To make it a minimization problem, we return the negative of the log-sum-exp term.
    Actually, let's use a standard Thomson-like energy: sum(1/d^2) or similar.
    However, 1/d^2 is singular. A better smooth surrogate for "maximize min distance" is:
    -log(sum(exp(-d^2 / T)))
    Minimizing this encourages all distances to be large and similar.
    """
    n = len(points)
    if n < 2:
        return 0.0
    
    # Compute pairwise squared distances
    # d2[i, j] = ||p_i - p_j||^2
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    d2 = np.sum(diff**2, axis=2)
    
    # Mask diagonal
    np.fill_diagonal(d2, np.inf)
    
    # Only upper triangle to avoid double counting, though for sum it doesn't matter much
    # We want to minimize E = -log(sum_{i<j} exp(-d2_ij / T))
    # This is the "soft-min" of the distances (squared).
    T = 0.1  # Temperature parameter
    # To avoid overflow/underflow, use logsumexp trick
    # sum(exp(-d2/T)) = exp(max(-d2/T)) * sum(exp((-d2/T) - max(-d2/T)))
    # Let z = -d2 / T
    z = -d2 / T
    z_max = np.max(z)
    if not np.isfinite(z_max):
        return 0.0
    log_sum_exp = z_max + np.log(np.sum(np.exp(z - z_max)))
    
    return -log_sum_exp

def _objective_energy(points_flat):
    """
    Objective for L-BFGS-B: Minimize the repulsive energy.
    """
    points = points_flat.reshape(13, 2)
    if not np.all(np.isfinite(points)):
        return 1e6
    return _energy(points)

def _objective_exact(points_flat):
    """
    Objective for final polishing: Maximize the exact score (A_min / A_hull).
    """
    points = points_flat.reshape(13, 2)
    if not np.all(np.isfinite(points)):
        return 1e6
    s = _score(points)
    if s <= 1e-12:
        return 1e6
    return -s

def _generate_initial_points(seed=0):
    """
    Generate diverse initial configurations.
    """
    rng = np.random.default_rng(seed)
    n = 13
    
    configs = []
    
    # 1. Regular polygon
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    poly = np.column_stack([np.cos(angles), np.sin(angles)])
    configs.append(poly)
    
    # 2. Center + 2 rings (1+6+6)
    center = np.array([[0.0, 0.0]])
    r1 = 0.5
    a1 = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    ring1 = np.column_stack([r1 * np.cos(a1), r1 * np.sin(a1)])
    r2 = 1.0
    a2 = np.linspace(0, 2 * np.pi, 6, endpoint=False) + np.pi/6
    ring2 = np.column_stack([r2 * np.cos(a2), r2 * np.sin(a2)])
    configs.append(np.vstack([center, ring1, ring2]))
    
    # 3. Random points in unit circle
    for _ in range(5):
        angles = rng.uniform(0, 2 * np.pi, n)
        radii = rng.uniform(0.2, 1.0, n) # Avoid center clustering too much initially
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        configs.append(pts)
        
    # 4. Perturbed regular polygon
    for _ in range(5):
        noise = rng.normal(0, 0.1, (n, 2))
        configs.append(poly + noise)

    return configs

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of n points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points. Here n = 13.

    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    n = 13
    best_score = -1.0
    best_points = np.random.default_rng(0).random((n, 2))
    
    initializers = _generate_initial_points(seed=42)
    
    # We will iterate through initializers and optimize them
    # Phase 1: L-BFGS-B on Energy (smooth)
    # Phase 2: Powell or Nelder-Mead on Exact Score (non-smooth) for polishing
    
    # To save time, we might limit the number of restarts or iterations per restart
    # L-BFGS-B is fast.
    
    for i, init_pts in enumerate(initializers):
        if time.time() > _DEADLINE - 30:
            break
            
        init_flat = init_pts.flatten()
        
        # Phase 1: Smooth energy minimization
        try:
            res_energy = minimize(
                _objective_energy,
                init_flat,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-6}
            )
            pts_energy = res_energy.x.reshape(n, 2)
            
            if not np.all(np.isfinite(pts_energy)):
                continue
                
            # Check if we have a valid hull
            if _hull_area(pts_energy) <= 1e-12:
                continue
                
            # Phase 2: Exact polishing
            # Use Powell's method which is robust for non-smooth functions
            # Or Nelder-Mead. Powell is often better for multi-modal non-smooth.
            # Given the landscape is likely complex, let's try Powell with a reasonable maxiter.
            # To ensure we don't timeout, we check time.
            
            # We can also try a few different local optimizers for the exact score
            # 1. Powell
            res_powell = minimize(
                _objective_exact,
                pts_energy,
                method='Powell',
                options={'maxiter': 100, 'ftol': 1e-8, 'xtol': 1e-8, 'disp': False}
            )
            pts_powell = res_powell.x.reshape(n, 2)
            
            if np.all(np.isfinite(pts_powell)):
                score_powell = _score(pts_powell)
                if score_powell > best_score:
                    best_score = score_powell
                    best_points = pts_powell.copy()
                    
            # 2. Nelder-Mead (sometimes finds different local minima)
            # Only run if time permits
            if time.time() < _DEADLINE - 10:
                res_nm = minimize(
                    _objective_exact,
                    pts_energy,
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True, 'disp': False}
                )
                pts_nm = res_nm.x.reshape(n, 2)
                
                if np.all(np.isfinite(pts_nm)):
                    score_nm = _score(pts_nm)
                    if score_nm > best_score:
                        best_score = score_nm
                        best_points = pts_nm.copy()
                        
        except Exception:
            continue
            
    return best_points

# EVOLVE-BLOCK-END
