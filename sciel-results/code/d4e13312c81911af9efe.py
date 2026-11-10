# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull, Delaunay
from scipy.optimize import minimize
import itertools
import time


def _triangle_area(p1, p2, p3):
    """Compute the area of triangle formed by three points."""
    return abs(p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1])) / 2.0


def _min_triangle_area(points):
    """Compute the minimum triangle area among all triples of points."""
    n = len(points)
    min_area = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                area = _triangle_area(points[i], points[j], points[k])
                if area < min_area:
                    min_area = area
    return min_area


def _hull_area(points):
    """Compute the area of the convex hull of points."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0


def _score(points):
    """Compute the score: A_min / A_hull."""
    a_min = _min_triangle_area(points)
    a_hull = _hull_area(points)
    if a_hull <= 0 or not np.isfinite(a_min) or not np.isfinite(a_hull):
        return 0.0
    return a_min / a_hull


def _logsumexp_min_triangle(points, beta=50.0):
    """
    Compute a smooth surrogate for the minimum triangle area using log-sum-exp.
    Lower values of the surrogate correspond to larger minimum areas.
    We return -logsumexp(-beta * areas) which is a smooth lower bound on min(areas).
    """
    n = len(points)
    areas = np.empty((n, n, n))
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = _triangle_area(points[i], points[j], points[k])
                areas[count] = a
                count += 1
    areas = areas[:count]
    # Use log-sum-exp for the minimum: min(a) ≈ -log(sum(exp(-beta*a)))/beta
    # We want to maximize min, so minimize -logsumexp(-beta * areas)
    shifted = -beta * areas
    max_shifted = np.max(shifted)
    logsumexp_val = max_shifted + np.log(np.sum(np.exp(shifted - max_shifted)))
    return logsumexp_val / beta


def _objective(x, beta=50.0):
    """
    Objective function for optimization.
    x = [x0, y0, x1, y1, x2, y2, x3, y3] representing 4 basis vectors from origin.
    We generate 13 points from a hexagonal lattice defined by these parameters.
    """
    # Parse parameters
    x0, y0 = x[0], x[1]
    v1x, v1y = x[2], x[3]
    v2x, v2y = x[4], x[5]
    v3x, v3y = x[6], x[7]

    # Generate 13 points in a hexagonal pattern
    # Hexagonal lattice points: (i*v1 + j*v2) for i,j in some range
    # For 13 points, we can use a hexagon of radius 2:
    # Center + 6 neighbors + 12 next-nearest = 19, too many
    # Let's use: center, 6 at distance 1, 6 at distance 2 but only some
    # Actually, let's use a simpler approach: 13 points in a hexagonal arrangement
    # (0,0), (1,0), (0,1), (1,1), (2,0), (0,2), (2,1), (1,2), (-1,0), (0,-1), (-1,-1), (1,-1), (-1,1)
    
    # Define the 13 lattice coordinates in the basis (v1, v2)
    # Using axial coordinates for hexagonal grid
    lattice_coords = np.array([
        [0, 0],
        [1, 0], [0, 1], [1, 1],
        [2, 0], [0, 2], [2, 1], [1, 2],
        [-1, 0], [0, -1], [-1, -1],
        [1, -1], [-1, 1]
    ], dtype=float)

    # Generate points: origin + i*v1 + j*v2
    points = np.empty((13, 2))
    for idx, (i, j) in enumerate(lattice_coords):
        points[idx, 0] = x0 + i * v1x + j * v2x
        points[idx, 1] = y0 + i * v1y + j * v2y

    # Check for degenerate cases
    if not np.all(np.isfinite(points)):
        return 1e10

    # Compute the smooth surrogate objective
    # We want to maximize min_triangle_area / hull_area
    # So minimize -log(min_triangle_area) + log(hull_area)
    # Using log-sum-exp for min_triangle_area
    
    a_min_surrogate = _logsumexp_min_triangle(points, beta=beta)
    
    try:
        a_hull = _hull_area(points)
        if a_hull <= 0:
            return 1e10
    except Exception:
        return 1e10

    # Objective: minimize log(a_hull) - log(a_min_surrogate)
    # = log(a_hull / a_min_surrogate)
    # We want to minimize this, which maximizes a_min/a_hull
    obj = np.log(a_hull) - np.log(a_min_surrogate)
    
    if not np.isfinite(obj):
        return 1e10
    
    return obj


def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points on or inside a convex region to maximize
    the ratio of the minimum triangle area to the convex hull area.
    
    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    start_time = time.time()
    deadline = 600.0  # Safety margin
    
    # Initial guess: hexagonal lattice with standard basis
    # v1 = (1, 0), v2 = (0.5, sqrt(3)/2) for 60-degree hexagonal lattice
    v1 = np.array([1.0, 0.0])
    v2 = np.array([0.5, np.sqrt(3) / 2])
    
    x0_init = 0.0
    y0_init = 0.0
    
    x0 = np.array([x0_init, y0_init, v1[0], v1[1], v2[0], v2[1], 0.0, 0.0])
    
    # Try multiple random initializations
    best_points = None
    best_score = -np.inf
    
    rng = np.random.default_rng(seed=12345)
    
    # First, try the standard hexagonal lattice
    v1_std = np.array([1.0, 0.0])
    v2_std = np.array([0.5, np.sqrt(3) / 2])
    x0_std = np.array([0.0, 0.0, v1_std[0], v1_std[1], v2_std[0], v2_std[1], 0.0, 0.0])
    
    lattice_coords = np.array([
        [0, 0],
        [1, 0], [0, 1], [1, 1],
        [2, 0], [0, 2], [2, 1], [1, 2],
        [-1, 0], [0, -1], [-1, -1],
        [1, -1], [-1, 1]
    ], dtype=float)
    
    points_std = np.empty((13, 2))
    for idx, (i, j) in enumerate(lattice_coords):
        points_std[idx, 0] = x0_std[0] + i * x0_std[2] + j * x0_std[4]
        points_std[idx, 1] = x0_std[1] + i * x0_std[3] + j * x0_std[5]
    
    score_std = _score(points_std)
    if score_std > best_score:
        best_score = score_std
        best_points = points_std.copy()
    
    # Try optimization from the standard hexagonal lattice
    if time.time() - start_time < deadline:
        try:
            result = minimize(_objective, x0_std, method='L-BFGS-B', 
                            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-12})
            if result.success and result.fun < 1e9:
                # Reconstruct points from optimized parameters
                x_opt = result.x
                points_opt = np.empty((13, 2))
                for idx, (i, j) in enumerate(lattice_coords):
                    points_opt[idx, 0] = x_opt[0] + i * x_opt[2] + j * x_opt[4]
                    points_opt[idx, 1] = x_opt[1] + i * x_opt[3] + j * x_opt[5]
                
                score_opt = _score(points_opt)
                if score_opt > best_score:
                    best_score = score_opt
                    best_points = points_opt.copy()
        except Exception:
            pass
    
    # Try several random initializations
    for trial in range(10):
        if time.time() - start_time > deadline - 30:
            break
        
        # Random basis vectors
        v1_rand = rng.normal(size=2) * 1.0
        v2_rand = rng.normal(size=2) * 1.0
        # Ensure they're not too close to parallel
        cross = v1_rand[0] * v2_rand[1] - v1_rand[1] * v2_rand[0]
        if abs(cross) < 0.1:
            continue
        
        x0_rand = np.array([0.0, 0.0, v1_rand[0], v1_rand[1], v2_rand[0], v2_rand[1], 0.0, 0.0])
        
        try:
            result = minimize(_objective, x0_rand, method='L-BFGS-B',
                            options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-10})
            if result.success and result.fun < 1e9:
                x_opt = result.x
                points_opt = np.empty((13, 2))
                for idx, (i, j) in enumerate(lattice_coords):
                    points_opt[idx, 0] = x_opt[0] + i * x_opt[2] + j * x_opt[4]
                    points_opt[idx, 1] = x_opt[1] + i * x_opt[3] + j * x_opt[5]
                
                score_opt = _score(points_opt)
                if score_opt > best_score:
                    best_score = score_opt
                    best_points = points_opt.copy()
        except Exception:
            continue
    
    # Final local refinement using Nelder-Mead on the best points found
    if best_points is not None and time.time() - start_time < deadline - 10:
        def _nm_objective(pts_flat):
            pts = pts_flat.reshape(13, 2)
            return -_score(pts)
        
        try:
            result_nm = minimize(_nm_objective, best_points.flatten(), method='Nelder-Mead',
                               options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10})
            if result_nm.success:
                pts_nm = result_nm.x.reshape(13, 2)
                score_nm = _score(pts_nm)
                if score_nm > best_score:
                    best_score = score_nm
                    best_points = pts_nm.copy()
        except Exception:
            pass
    
    if best_points is None:
        # Fallback: return the standard hexagonal lattice
        return points_std
    
    return best_points


# EVOLVE-BLOCK-END
