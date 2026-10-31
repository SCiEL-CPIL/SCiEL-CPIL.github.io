"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
import time
import itertools

def _triangle_area(p1, p2, p3):
    """Compute the area of a triangle given three 2D points."""
    return 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))

def _min_area_exact(points):
    """Compute the minimum area among all triangles formed by three points."""
    n = len(points)
    min_area = float('inf')
    # Using itertools for cleaner code, performance is fine for N=11
    for i, j, k in itertools.combinations(range(n), 3):
        area = _triangle_area(points[i], points[j], points[k])
        if area < min_area:
            min_area = area
    return min_area

def _penalty(points):
    """Calculate penalty for points outside the triangle."""
    x = points[:, 0]
    y = points[:, 1]
    # Triangle: (0,0), (1,0), (0.5, sqrt(3)/2)
    # Inequalities:
    # 1. y >= 0
    # 2. y <= sqrt(3) * x
    # 3. y <= -sqrt(3) * (x - 1) => y <= sqrt(3) * (1 - x)
    
    # Distance to boundary (squared)
    d1 = np.maximum(0, -y)
    d2 = np.maximum(0, y - np.sqrt(3) * x)
    d3 = np.maximum(0, y - np.sqrt(3) * (1 - x))
    
    return np.sum(d1**2) + np.sum(d2**2) + np.sum(d3**2)

def _objective_smooth(params, beta):
    """Smooth objective function for global search."""
    points = params.reshape(-1, 2)
    
    # Calculate all triangle areas
    n = len(points)
    areas = np.array([_triangle_area(points[i], points[j], points[k]) 
                      for i, j, k in itertools.combinations(range(n), 3)])
    
    if len(areas) == 0:
        return 0.0
        
    # Log-sum-exp for numerical stability
    # We want softmin(areas) = -1/beta * log(sum(exp(-beta * areas)))
    scaled_areas = -beta * areas
    max_val = np.max(scaled_areas)
    log_sum_exp = max_val + np.log(np.sum(np.exp(scaled_areas - max_val)))
    smooth_min = -log_sum_exp / beta
    
    pen = _penalty(points)
    # We want to maximize smooth_min, so minimize -smooth_min
    return -smooth_min + 100.0 * pen

def _objective_exact(params):
    """Exact objective function for local refinement."""
    points = params.reshape(-1, 2)
    min_a = _min_area_exact(points)
    pen = _penalty(points)
    return -min_a + 1000.0 * pen

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Keep 10s buffer for safety
    
    # Bounds for Differential Evolution
    # x in [0, 1], y in [0, sqrt(3)/2]
    bounds = []
    for _ in range(N):
        bounds.append((0.0, 1.0))
        bounds.append((0.0, np.sqrt(3)/2))
        
    best_params = None
    best_score = -np.inf
    
    # Phase 1: Global Search with Differential Evolution
    # Using a smooth objective to help DE navigate the landscape
    try:
        # maxiter and popsize tuned for speed and quality
        de_result = differential_evolution(
            lambda x: _objective_smooth(x, 20.0), 
            bounds, 
            maxiter=50, 
            popsize=10, 
            tol=1e-4, 
            seed=42,
            polish=False,
            updating='immediate'
        )
        if de_result.success or de_result.fun < 10.0: # Check if it found something reasonable
            best_params = de_result.x
            # Evaluate exact score for tracking
            current_score = -_objective_exact(best_params)
            if current_score > best_score:
                best_score = current_score
    except Exception as e:
        print(f"DE Error: {e}")
        
    if best_params is None:
        # Fallback: Random initialization
        rng = np.random.RandomState(42)
        best_params = rng.rand(N, 2)
        best_params[:, 0] *= 1.0
        best_params[:, 1] *= np.sqrt(3)/2

    elapsed = time.time() - start_time
    if elapsed > deadline - 30:
        return best_params.reshape(N, 2)

    # Phase 2: Local Refinement with BFGS
    # BFGS uses numerical gradients, works well with non-smooth min area
    try:
        res = minimize(
            _objective_exact, 
            best_params, 
            method='BFGS', 
            options={'maxiter': 100, 'gtol': 1e-6}
        )
        if res.fun < _objective_exact(best_params):
            best_params = res.x
            best_score = -res.fun
    except Exception as e:
        print(f"BFGS Error: {e}")

    elapsed = time.time() - start_time
    if elapsed > deadline - 10:
        return best_params.reshape(N, 2)

    # Phase 3: Final Polish
    # Try to squeeze out more precision
    try:
        res2 = minimize(
            _objective_exact, 
            best_params, 
            method='BFGS', 
            options={'maxiter': 50, 'gtol': 1e-8}
        )
        if res2.fun < _objective_exact(best_params):
            best_params = res2.x
    except Exception as e:
        print(f"Polish Error: {e}")

    return best_params.reshape(N, 2)

# EVOLVE-BLOCK-END
