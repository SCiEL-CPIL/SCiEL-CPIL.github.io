# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
from scipy.spatial import ConvexHull

def _compute_score(points: np.ndarray) -> float:
    """
    Compute the raw score: A_min / A_hull.
    Returns 0 if invalid.
    """
    n = points.shape[0]
    if n < 3:
        return 0.0
    
    try:
        hull = ConvexHull(points)
        A_hull = hull.volume
    except Exception:
        return 0.0
        
    if A_hull <= 1e-10:
        return 0.0
        
    idx = np.arange(n)
    triples = list(itertools.combinations(idx, 3))
    triples_arr = np.array(triples)
    
    p0 = points[triples_arr[:, 0]]
    p1 = points[triples_arr[:, 1]]
    p2 = points[triples_arr[:, 2]]
    
    v1 = p1 - p0
    v2 = p2 - p0
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    
    if min_area <= 0:
        return 0.0
        
    return min_area / A_hull

def _soft_min_area(points: np.ndarray, temperature: float = 0.01):
    """
    Compute a smooth approximation of the minimum triangle area.
    Uses the log-sum-exp trick for numerical stability.
    Returns (soft_min_area, gradient_wrt_points)
    """
    n = points.shape[0]
    idx = np.arange(n)
    triples = list(itertools.combinations(idx, 3))
    triples_arr = np.array(triples)
    m = len(triples)
    
    p0 = points[triples_arr[:, 0]]
    p1 = points[triples_arr[:, 1]]
    p2 = points[triples_arr[:, 2]]
    
    v1 = p1 - p0
    v2 = p2 - p0
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    # To avoid issues with 0 areas, add a small epsilon
    areas = np.maximum(areas, 1e-12)
    
    # Log-sum-exp for soft min: -log(sum(exp(-areas/T))) * T
    # We want to minimize -soft_min, so we maximize soft_min.
    # Let's define objective = -soft_min_area + lambda * A_hull
    # We need gradient of soft_min_area.
    
    # Soft min: S = T * log(sum(exp(-A_i / T)))
    # Wait, standard soft min is: T * log(sum(exp(-A_i/T)))
    # As T->0, this approaches min(A_i).
    
    log_terms = -areas / temperature
    max_log = np.max(log_terms)
    sum_exp = np.sum(np.exp(log_terms - max_log))
    soft_min_val = temperature * (max_log + np.log(sum_exp))
    
    # Gradient of soft_min_val wrt areas:
    # d(soft_min)/dA_i = -exp(-A_i/T) / sum(exp(-A_j/T))
    weights = np.exp(log_terms - max_log) / sum_exp
    grad_areas = -weights
    
    # Now chain rule: dA_i/dp
    # A_i = 0.5 * |cross_i|
    # dA_i/dp = 0.5 * sign(cross_i) * d(cross_i)/dp
    # cross_i = (p1-p0)x(p2-p0)
    # d(cross)/dp0 = - (p1-p0)_y * d(p2-p0)_x/dp0 ... 
    # Actually, let's use the vector derivative.
    # cross = x1(y2-y3) + x2(y3-y1) + x3(y1-y2)
    # d(cross)/dx1 = y2-y3
    # d(cross)/dy1 = -(x2-x3)
    # d(cross)/dx2 = y3-y1
    # d(cross)/dy2 = -(x3-x1)
    # d(cross)/dx3 = y1-y2
    # d(cross)/dy3 = -(x1-x2)
    
    grad_points = np.zeros((n, 2))
    
    # We need to accumulate gradients for each point in each triangle.
    # For each triangle (i, j, k):
    # grad_A = 0.5 * sign(cross) * grad_cross
    # grad_cross is a 6-vector [dx_i, dy_i, dx_j, dy_j, dx_k, dy_k]
    
    # Vectorized approach:
    # For each triangle t:
    # x1, y1 = p0[t]
    # x2, y2 = p1[t]
    # x3, y3 = p2[t]
    # sign_c = np.sign(cross[t])
    # coeff = 0.5 * sign_c * grad_areas[t]
    
    # d(cross)/dx1 = y2-y3
    # d(cross)/dy1 = x3-x2
    # d(cross)/dx2 = y3-y1
    # d(cross)/dy2 = x1-x3
    # d(cross)/dx3 = y1-y2
    # d(cross)/dy3 = x2-x1
    
    coeffs = 0.5 * np.sign(cross) * grad_areas
    
    # Accumulate gradients
    # For p0 (index i):
    # grad_x += coeffs * (y2 - y3)
    # grad_y += coeffs * (x3 - x2)
    
    grad_x = np.zeros(n)
    grad_y = np.zeros(n)
    
    # p0
    grad_x[triples_arr[:, 0]] += coeffs * (p1[:, 1] - p2[:, 1])
    grad_y[triples_arr[:, 0]] += coeffs * (p2[:, 0] - p1[:, 0])
    
    # p1
    grad_x[triples_arr[:, 1]] += coeffs * (p2[:, 1] - p0[:, 1])
    grad_y[triples_arr[:, 1]] += coeffs * (p0[:, 0] - p2[:, 0])
    
    # p2
    grad_x[triples_arr[:, 2]] += coeffs * (p0[:, 1] - p1[:, 1])
    grad_y[triples_arr[:, 2]] += coeffs * (p1[:, 0] - p0[:, 0])
    
    grad_points[:, 0] = grad_x
    grad_points[:, 1] = grad_y
    
    return soft_min_val, grad_points

def _hull_area_and_grad(points: np.ndarray):
    """
    Compute convex hull area and a numerical gradient approximation.
    Note: Analytical gradient of convex hull area is complex due to changing hull vertices.
    We will use finite differences for the hull area gradient in the optimization loop,
    or just use the area value and rely on the soft-min gradient to push points apart.
    Actually, to make this a proper constrained optimization, we need the gradient of A_hull.
    However, A_hull is piecewise linear. 
    Alternative: Instead of Lagrangian on A_hull, we can normalize the points to unit area 
    and just maximize A_min. 
    But normalization changes the geometry.
    
    Let's stick to a simpler strategy: 
    1. Maximize A_min using gradient ascent on the soft-min.
    2. Periodically normalize the point set to have A_hull = 1.
    This decouples the optimization of shape (maximizing min area) from scaling.
    """
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points to maximize the smallest triangle area
    relative to the convex hull area using Gradient Ascent on Soft-Min Area.
    """
    start_time = time.time()
    deadline = 600  # seconds, with a safety margin
    
    # Initial guess: Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, 13, endpoint=False)
    points = np.column_stack([0.5 + 0.4 * np.cos(angles), 0.5 + 0.4 * np.sin(angles)])
    
    best_points = points.copy()
    best_score = _compute_score(best_points)
    
    # Optimization parameters
    learning_rate = 0.01
    temperature = 0.05  # For soft-min
    momentum = 0.9
    velocity = np.zeros_like(points)
    
    # Main optimization loop
    for step in range(10000):
        if time.time() - start_time > deadline - 5:
            break
        
        # 1. Compute soft-min area and its gradient
        soft_min, grad_soft_min = _soft_min_area(points, temperature)
        
        # 2. Compute hull area
        A_hull = _hull_area_and_grad(points)
        
        if A_hull < 1e-10:
            # If hull is degenerate, add noise
            points += np.random.normal(0, 0.01, points.shape)
            continue
            
        # 3. Normalize points to have area 1
        # This keeps the scale fixed and lets the gradient work on shape
        scale = 1.0 / np.sqrt(A_hull)
        # Center points to avoid drift
        center = np.mean(points, axis=0)
        points_normalized = (points - center) * scale
        # Re-center to [0,1] box roughly? 
        # Actually, we can just work in centered coordinates.
        # The score is scale invariant.
        
        # Update points using gradient ascent on soft_min (which is scale invariant if we normalize)
        # But we need to be careful: the gradient computed was on the original points.
        # Let's just do a simple gradient step on the original points, then normalize.
        
        # Gradient step
        velocity = momentum * velocity + learning_rate * grad_soft_min
        points_new = points + velocity
        
        # Clip to reasonable bounds to prevent explosion
        points_new = np.clip(points_new, -10, 10)
        
        # Normalize
        A_hull_new = _hull_area_and_grad(points_new)
        if A_hull_new > 1e-10:
            center_new = np.mean(points_new, axis=0)
            points_new = (points_new - center_new) * (1.0 / np.sqrt(A_hull_new))
            points = points_new
        else:
            # If normalization failed, revert
            points = points - velocity # undo step?
            velocity = 0
            
        # Compute score
        current_score = _compute_score(points)
        
        if current_score > best_score:
            best_score = current_score
            best_points = points.copy()
            
        # Anneal temperature
        if step % 100 == 0:
            temperature *= 0.95
            learning_rate *= 0.98
            
    # Final fallback if best_points is invalid
    if not np.all(np.isfinite(best_points)):
        angles = np.linspace(0, 2 * np.pi, 13, endpoint=False)
        best_points = np.column_stack([0.5 + 0.4 * np.cos(angles), 0.5 + 0.4 * np.sin(angles)])
        
    return best_points

# EVOLVE-BLOCK-END
