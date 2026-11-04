"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import time
import itertools

def _convex_hull_area(points):
    """Compute the area of the convex hull of a set of 2D points."""
    if len(points) < 3:
        return 0.0
    try:
        hull = ConvexHull(points)
        return hull.volume  # In 2D, volume is area
    except Exception:
        return 1.0  # Fallback to avoid division by zero

def _min_triangle_area(points):
    """Compute the minimum area of any triangle formed by triples of points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                # Area of triangle = 0.5 * |cross product|
                area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
                if area < min_area:
                    min_area = area
    return min_area

def _score(points):
    """Compute the score: min triangle area / convex hull area."""
    hull_area = _convex_hull_area(points)
    if hull_area <= 1e-12:
        return 0.0
    min_tri = _min_triangle_area(points)
    return min_tri / hull_area

def _soft_min_objective(params, beta):
    """
    Soft-min surrogate for the min triangle area.
    params: flattened 2D points of shape (n*2,)
    beta: temperature parameter for log-sum-exp
    Returns: -soft_min (we minimize negative to maximize soft_min)
    """
    points = params.reshape(-1, 2)
    n = len(points)
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
                areas.append(area)
    areas = np.array(areas)
    # Log-sum-exp for soft-min
    max_a = np.max(areas)
    log_sum = max_a + np.log(np.sum(np.exp(areas - max_a)))
    soft_min = log_sum / beta
    # We want to maximize soft_min, so minimize negative
    return -soft_min

def _soft_min_gradient(params, beta):
    """
    Compute gradient of -soft_min for L-BFGS-B.
    """
    points = params.reshape(-1, 2)
    n = len(points)
    areas = np.zeros((n * (n-1) * (n-2)) // 6)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
                areas[idx] = area
                idx += 1
    
    max_a = np.max(areas)
    exp_areas = np.exp(areas - max_a)
    sum_exp = np.sum(exp_areas)
    weights = exp_areas / sum_exp  # These are the softmax weights
    
    # Gradient of soft_min = sum(weights * d(area)/d(params))
    # Gradient of -soft_min = -sum(weights * d(area)/d(params))
    grad = np.zeros_like(params)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                w = weights[idx]
                p1, p2, p3 = points[i], points[j], points[k]
                # Area = 0.5 * |cross|, assume positive for gradient purposes
                # cross = (p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1])
                # d(area)/dp1 = -0.5 * (cross_hat) where cross_hat is the signed cross
                # For simplicity, use numerical gradient or analytical
                # Let's use analytical for the signed area
                cross = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1])
                area_signed = 0.5 * cross
                # Gradient of area_signed w.r.t. p1, p2, p3
                # d(area_signed)/dp1x = -0.5 * (p3[1]-p1[1]) + 0.5*(p2[1]-p1[1]) ... this is getting complex
                # Let's just use finite differences for the gradient in the optimization
                idx += 1
    
    # Since analytical gradient is complex, let's use a simpler approach
    # We'll use the fact that L-BFGS-B can use numerical gradients
    return None  # Use numerical gradient

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    best_points = None
    best_score = -float('inf')
    
    # Initial guess: regular polygon on unit circle
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    initial_points = np.column_stack([np.cos(angles), np.sin(angles)])
    initial_params = initial_points.flatten()
    
    # Try multiple initializations
    inits = [initial_params]
    
    # Add some random perturbations
    rng = np.random.RandomState(42)
    for _ in range(5):
        perturbed = initial_params + rng.randn(N * 2) * 0.1
        inits.append(perturbed)
    
    # Also try a different initial configuration
    # Square-like configuration
    square_angles = np.linspace(0, 2 * np.pi, 4, endpoint=False)
    square_points = np.column_stack([np.cos(square_angles), np.sin(square_angles)])
    # Duplicate and perturb
    for _ in range(3):
        perturbed_sq = square_points + rng.randn(4, 2) * 0.05
        # Expand to 14 points by adding more
        extra_angles = np.linspace(0, 2 * np.pi, 10, endpoint=False)
        extra_points = np.column_stack([np.cos(extra_angles), np.sin(extra_angles)])
        combined = np.vstack([perturbed_sq, extra_points])
        inits.append(combined.flatten())
    
    for init_params in inits:
        if time.time() > deadline - 10:
            break
        
        # Two-stage optimization
        # Stage 1: beta=50 (broad)
        result1 = minimize(
            _soft_min_objective,
            init_params,
            args=(50,),
            method='L-BFGS-B',
            options={'maxiter': 100, 'ftol': 1e-8}
        )
        
        if result1.success or result1.fun < _soft_min_objective(init_params, 50):
            params1 = result1.x
        else:
            params1 = init_params
        
        # Stage 2: beta=250 (sharp)
        result2 = minimize(
            _soft_min_objective,
            params1,
            args=(250,),
            method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-10}
        )
        
        if result2.success or result2.fun < _soft_min_objective(params1, 250):
            params2 = result2.x
        else:
            params2 = params1
        
        # Evaluate actual score
        points = params2.reshape(N, 2)
        score = _score(points)
        
        if score > best_score:
            best_score = score
            best_points = points.copy()
    
    # Final polish: use exact min with L-BFGS-B on the soft-min with high beta
    if best_points is not None and time.time() < deadline - 5:
        final_params = best_points.flatten()
        result_final = minimize(
            _soft_min_objective,
            final_params,
            args=(500,),
            method='L-BFGS-B',
            options={'maxiter': 300, 'ftol': 1e-12}
        )
        
        if result_final.success or result_final.fun < _soft_min_objective(final_params, 500):
            final_points = result_final.x.reshape(N, 2)
            final_score = _score(final_points)
            if final_score > best_score:
                best_points = final_points
                best_score = final_score
    
    if best_points is None:
        # Fallback to initial regular polygon
        best_points = initial_points
    
    return best_points
# EVOLVE-BLOCK-END
