"""Heilbronn convex regions: place 13 points to maximize min triangle area / hull area."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from itertools import combinations

N = 13
TIME_LIMIT = 120.0
SAFETY_MARGIN = 5.0  # seconds before limit to stop

def _triangle_areas(points):
    """Compute all triangle areas from n points. Returns array of size C(n,3)."""
    n = len(points)
    # Vectorized: for each triple (i,j,k), area = 0.5*|cross|
    # Use broadcasting for speed
    idx = np.array(list(combinations(range(n), 3)))
    p1 = points[idx[:, 0]]
    p2 = points[idx[:, 1]]
    p3 = points[idx[:, 2]]
    # Cross product in 2D
    cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
    return 0.5 * np.abs(cross)

def _convex_hull_area(points):
    """Compute area of convex hull of points."""
    try:
        hull = ConvexHull(points)
        return hull.volume  # In 2D, volume is area
    except Exception:
        return 0.0

def _score(points):
    """Score = min triangle area / convex hull area."""
    areas = _triangle_areas(points)
    min_area = np.min(areas)
    hull_area = _convex_hull_area(points)
    if hull_area <= 1e-12:
        return 0.0
    return min_area / hull_area

def _soft_min_objective(x, beta=100.0):
    """Soft-min surrogate: -soft_min(areas) / hull_area, to be minimized."""
    points = x.reshape(N, 2)
    areas = _triangle_areas(points)
    # Soft min via log-sum-exp: soft_min = -log(sum(exp(-beta*a)))/beta
    # But for numerical stability, subtract max first
    a_min = np.min(areas)
    a_max = np.max(areas)
    if a_max - a_min < 1e-15:
        soft_min_val = a_min
    else:
        # soft_min ≈ a_min - (1/beta)*log(sum(exp(-beta*(a - a_min))))
        # This is numerically stable
        exp_terms = np.exp(-beta * (areas - a_min))
        soft_min_val = a_min - (1.0 / beta) * np.log(np.sum(exp_terms))
    hull_area = _convex_hull_area(points)
    if hull_area <= 1e-12:
        return 1e10
    return -soft_min_val / hull_area

def _construct_initial():
    """Create a reasonable initial configuration: regular polygon + center-ish points."""
    # Start with a regular 13-gon, then perturb
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    points = np.column_stack([np.cos(angles), np.sin(angles)])
    # Add some randomness
    rng = np.random.default_rng(42)
    points += rng.normal(0, 0.05, size=points.shape)
    return points

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT - SAFETY_MARGIN
    
    best_points = _construct_initial()
    best_score = _score(best_points)
    
    # Try multiple initializations and local optimizations
    rng = np.random.default_rng(12345)
    
    # Different initial configurations
    inits = []
    # Regular polygon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    inits.append(np.column_stack([np.cos(angles), np.sin(angles)]))
    # Regular polygon with random rotation
    inits.append(np.column_stack([np.cos(angles + 0.3), np.sin(angles + 0.3)]))
    # Random points in unit square
    inits.append(rng.random((N, 2)))
    # Random points in unit circle
    theta = rng.uniform(0, 2 * np.pi, N)
    r = np.sqrt(rng.uniform(0, 1, N))
    inits.append(np.column_stack([r * np.cos(theta), r * np.sin(theta)]))
    # Fibonacci spiral
    golden = (1 + np.sqrt(5)) / 2
    for i in range(N):
        angle = i * 2 * np.pi / golden
        radius = np.sqrt((i + 0.5) / N)
        if i == 0:
            fib = np.array([[radius * np.cos(angle), radius * np.sin(angle)]])
        else:
            fib = np.vstack([fib, [radius * np.cos(angle), radius * np.sin(angle)]])
    inits.append(fib)
    
    # Soft-min beta schedule: start low, increase
    betas = [10.0, 30.0, 100.0, 300.0, 1000.0]
    
    for init_idx, init_pts in enumerate(inits):
        if time.time() > deadline:
            break
        
        current_pts = init_pts.copy()
        current_score = _score(current_pts)
        
        # Multi-stage optimization with increasing beta
        for beta in betas:
            if time.time() > deadline:
                break
            
            # Optimize using L-BFGS-B
            try:
                result = minimize(
                    _soft_min_objective,
                    current_pts.flatten(),
                    args=(beta,),
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                if result.success or result.fun < 1e10:
                    new_pts = result.x.reshape(N, 2)
                    new_score = _score(new_pts)
                    if new_score > current_score:
                        current_pts = new_pts
                        current_score = new_score
            except Exception:
                pass
        
        # Final polish with high beta
        if time.time() < deadline:
            try:
                result = minimize(
                    _soft_min_objective,
                    current_pts.flatten(),
                    args=(1000.0,),
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                if result.success or result.fun < 1e10:
                    new_pts = result.x.reshape(N, 2)
                    new_score = _score(new_pts)
                    if new_score > current_score:
                        current_pts = new_pts
                        current_score = new_score
            except Exception:
                pass
        
        if current_score > best_score:
            best_score = current_score
            best_points = current_pts.copy()
    
    # Try a few more random restarts if time permits
    restarts = 0
    while time.time() < deadline and restarts < 5:
        restarts += 1
        # Perturb best solution
        perturbed = best_points + rng.normal(0, 0.1, size=best_points.shape)
        current_score = _score(perturbed)
        
        for beta in [100.0, 1000.0]:
            if time.time() > deadline:
                break
            try:
                result = minimize(
                    _soft_min_objective,
                    perturbed.flatten(),
                    args=(beta,),
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                if result.success or result.fun < 1e10:
                    new_pts = result.x.reshape(N, 2)
                    new_score = _score(new_pts)
                    if new_score > current_score:
                        perturbed = new_pts
                        current_score = new_score
            except Exception:
                pass
        
        if current_score > best_score:
            best_score = current_score
            best_points = perturbed.copy()
    
    return best_points
