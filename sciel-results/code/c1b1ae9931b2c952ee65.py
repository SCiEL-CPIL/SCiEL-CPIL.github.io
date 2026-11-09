# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
import itertools
from scipy.optimize import minimize, minimize_scalar

_START_TIME = time.time()
_DEADLINE = _START_TIME + 580.0  # 20s safety margin

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

def _soft_min_area(points, beta):
    """
    Compute a differentiable approximation of the minimum triangle area using log-sum-exp.
    We maximize the min area, so we minimize -soft_min_area.
    soft_min(x) = -1/beta * log(sum(exp(-beta * x_i)))
    """
    n = len(points)
    indices = np.array(list(itertools.combinations(range(n), 3)))
    p1 = points[indices[:, 0]]
    p2 = points[indices[:, 1]]
    p3 = points[indices[:, 2]]
    
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    # For numerical stability, subtract max
    max_area = np.max(areas)
    if max_area == 0:
        return 0.0
    
    # We want to minimize the minimum area, so we look at the "soft minimum"
    # log-sum-exp of negative areas gives -soft_min(areas)
    # soft_min(areas) = -1/beta * log(sum(exp(-beta * areas)))
    # To maximize soft_min, we minimize -soft_min = 1/beta * log(sum(exp(-beta * areas)))
    
    # Actually, let's just compute the value for the objective function which minimizes.
    # Objective: - (A_min / A_hull)
    # Surrogate: - (soft_min(areas) / A_hull)
    
    # Calculate soft min
    exp_terms = np.exp(-beta * (areas - max_area))
    sum_exp = np.sum(exp_terms)
    soft_min_val = max_area - (1.0 / beta) * np.log(sum_exp)
    
    return soft_min_val

def _objective_surrogate(points_flat, beta):
    """
    Objective for L-BFGS-B: Minimizes - (soft_min_area / hull_area)
    """
    points = points_flat.reshape(13, 2)
    
    # Check for degeneracy
    if not np.all(np.isfinite(points)):
        return 1e6
        
    a_hull = _hull_area(points)
    if a_hull <= 1e-12:
        return 1e6
        
    soft_min_a = _soft_min_area(points, beta)
    
    # We want to maximize soft_min_a / a_hull
    # So we minimize - (soft_min_a / a_hull)
    return -(soft_min_a / a_hull)

def _objective_exact(points_flat):
    """
    Objective for final Powell polish: Minimizes - (min_area / hull_area)
    """
    points = points_flat.reshape(13, 2)
    s = _score(points)
    if s <= 1e-12:
        return 1e6
    return -s

def _generate_spiral(n=13, seed=0):
    """
    Generate points in a spiral pattern, which tends to pack points more evenly
    in the interior than a regular polygon.
    """
    rng = np.random.default_rng(seed)
    # Archimedean spiral: r = a + b*theta
    # We want to fill a circle of radius ~1
    # n points, so theta goes from 0 to max_theta
    # We want roughly uniform area density.
    
    # Simple approach: sort random points by angle, assign radii based on sqrt(i/n) * max_r
    # This approximates a hexagonal packing density in a circle.
    
    angles = rng.uniform(0, 2 * np.pi, n)
    angles.sort()
    
    # Radii: sqrt(i/n) ensures uniform area density
    radii = np.sqrt(np.arange(n) / n) * 0.95 + 0.05
    
    points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    
    # Add small noise to break symmetry and avoid exact degeneracies
    points += rng.normal(0, 0.01, (n, 2))
    
    return points

def _generate_grid(n=13, seed=0):
    """
    Generate points from a perturbed grid.
    """
    rng = np.random.default_rng(seed)
    # 4x4 grid = 16 points, pick 13
    # Create grid
    x = np.linspace(-1, 1, 4)
    y = np.linspace(-1, 1, 4)
    X, Y = np.meshgrid(x, y)
    points = np.column_stack([X.flatten(), Y.flatten()])
    
    # Shuffle and take 13
    idx = rng.permutation(len(points))
    points = points[idx[:n]]
    
    # Center and scale
    center = points.mean(axis=0)
    points -= center
    max_r = np.max(np.linalg.norm(points, axis=1))
    if max_r > 0:
        points = points / max_r * 0.9
    
    # Add noise
    points += rng.normal(0, 0.05, (n, 2))
    
    return points

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
    
    generators = [
        _generate_spiral,
        _generate_grid,
        lambda n, seed: np.random.default_rng(seed).random((n, 2)) * 2 - 1 # Random
    ]
    
    current_seed = 0
    max_restarts = 100 # Limit restarts to allow time for optimization
    
    while time.time() < _DEADLINE and current_seed < max_restarts:
        if time.time() > _DEADLINE - 10:
            break
            
        gen_idx = current_seed % len(generators)
        gen_func = generators[gen_idx]
        
        try:
            initial_points = gen_func(n=n, seed=current_seed * 123 + 456)
            initial_flat = initial_points.flatten()
            
            # Phase 1: L-BFGS-B with soft-min surrogate
            # Start with moderate beta, increase for sharper minimum
            # We run a few iterations with increasing beta
            current_flat = initial_flat.copy()
            
            # Step 1: Low beta (smooth landscape)
            res1 = minimize(
                _objective_surrogate, 
                current_flat, 
                args=(50,),
                method='L-BFGS-B', 
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-6}
            )
            if res1.success:
                current_flat = res1.x
            
            # Step 2: Higher beta (sharper)
            res2 = minimize(
                _objective_surrogate, 
                current_flat, 
                args=(500,),
                method='L-BFGS-B', 
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if res2.success:
                current_flat = res2.x
                
            # Phase 2: Powell polish with exact objective
            # This cleans up any approximation error from the soft-min
            res3 = minimize(
                _objective_exact, 
                current_flat, 
                method='Powell', 
                options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-12, 'disp': False}
            )
            
            res_points = res3.x.reshape(n, 2)
            
            # Validate
            if not np.all(np.isfinite(res_points)):
                current_seed += 1
                continue
                
            res_score = _score(res_points)
            
            if res_score > best_score:
                best_score = res_score
                best_points = res_points.copy()
                
        except Exception:
            pass
            
        current_seed += 1
        
    return best_points

# EVOLVE-BLOCK-END
