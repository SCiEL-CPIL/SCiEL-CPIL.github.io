import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 14

def _compute_min_triangle_area(points):
    """Compute the minimum area of any triangle formed by triples of points."""
    n = len(points)
    min_area = float('inf')
    # Unroll loops for speed? 364 triangles is small.
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                # Area = 0.5 * |cross(p2-p1, p3-p1)|
                v1 = points[j] - points[i]
                v2 = points[k] - points[i]
                area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                if area < min_area:
                    min_area = area
    return min_area

def _convex_hull_area(points):
    """Compute the area of the convex hull."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _soft_min(areas, beta):
    """
    Approximate the minimum of a set of values using log-sum-exp.
    -1/beta * ln(sum(exp(-beta * a))) approx min(a) for large beta.
    """
    # For numerical stability, shift by min
    a_min = np.min(areas)
    # If beta is small, this is roughly the average. If large, it's the min.
    # We want to maximize min(areas), so we minimize -soft_min(areas).
    # soft_min = -1/beta * ln(sum(exp(-beta * (a - a_min)))) - a_min
    # Let's just compute the log-sum-exp directly on -areas?
    # Actually, simpler:
    # min(a) ~ -1/beta * log(sum(exp(-beta * a)))
    # To avoid underflow/overflow, subtract max(-beta * a) = -beta * min(a)
    # log(sum(exp(-beta * a))) = -beta * min(a) + log(sum(exp(-beta * (a - min(a)))))
    # So -1/beta * log(...) = min(a) - 1/beta * log(sum(exp(-beta * (a - min(a)))))
    # This is stable.
    
    shifted = areas - a_min
    # exp(-beta * shifted)
    exp_vals = np.exp(-beta * shifted)
    log_sum = np.log(np.sum(exp_vals))
    return a_min - (1.0 / beta) * log_sum

def _objective(params, beta):
    """
    Objective function for minimization.
    params: flattened array of shape (2*N,)
    """
    points = params.reshape(N, 2)
    
    # Check for degeneracy
    if not np.all(np.isfinite(points)):
        return 1e10
        
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-6:
        return 1e10
    
    # Normalize to unit area
    scale = 1.0 / np.sqrt(hull_area)
    points_norm = points * scale
    
    # Compute all triangle areas
    # Vectorized calculation for speed
    # i, j, k
    areas = []
    for i in range(N):
        for j in range(i + 1, N):
            for k in range(j + 1, N):
                v1 = points_norm[j] - points_norm[i]
                v2 = points_norm[k] - points_norm[i]
                area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                areas.append(area)
    areas = np.array(areas)
    
    # We want to MAXIMIZE the minimum area.
    # So we MINIMIZE the negative soft-min.
    sm = _soft_min(areas, beta)
    return -sm

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    best_score = -1.0
    best_points = None
    
    # Initial guess: Regular 14-gon
    angles = 2 * np.pi * np.arange(N) / N
    # Circumradius for unit area regular n-gon:
    # Area = n/2 * R^2 * sin(2*pi/n) = 1 => R = sqrt(2 / (n * sin(2*pi/n)))
    R = np.sqrt(2.0 / (N * np.sin(2 * np.pi / N)))
    init_points = np.column_stack([R * np.cos(angles), R * np.sin(angles)])
    
    # List of starting points to try
    starts = [init_points.copy()]
    
    # Add some perturbed versions
    rng = np.random.default_rng(42)
    for _ in range(4):
        noise = rng.normal(0, 0.1 * R, size=(N, 2))
        p = init_points + noise
        starts.append(p)
        
    # Try different aspect ratios (ellipses)
    for sx in [0.5, 1.5, 2.0]:
        for sy in [1.0/sx, 1.0, 2.0]: # Keep area roughly similar before normalization
            p = init_points.copy()
            p[:, 0] *= sx
            p[:, 1] *= sy
            starts.append(p)

    betas = [10, 50, 100, 200]
    
    for start_p in starts:
        if time.time() > deadline:
            break
            
        current_p = start_p.copy()
        
        # Refine with increasing beta
        for beta in betas:
            if time.time() > deadline:
                break
                
            # Flatten for scipy
            x0 = current_p.flatten()
            
            try:
                res = minimize(_objective, x0, args=(beta,), method='L-BFGS-B', 
                               options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8})
                
                if res.success or res.status in [1, 2, 3, 4, 5]:
                    # Extract new points
                    new_p = res.x.reshape(N, 2)
                    
                    # Verify and normalize
                    hull_area = _convex_hull_area(new_p)
                    if hull_area > 1e-6:
                        new_p = new_p / np.sqrt(hull_area)
                        
                        # Calculate actual score (min triangle area)
                        score = _compute_min_triangle_area(new_p)
                        
                        if score > best_score:
                            best_score = score
                            best_points = new_p
                            
                        current_p = new_p
            except Exception:
                pass

    # If we didn't find anything (shouldn't happen), return initial
    if best_points is None:
        best_points = init_points
        # Normalize
        hull_area = _convex_hull_area(best_points)
        if hull_area > 0:
            best_points = best_points / np.sqrt(hull_area)
        else:
            best_points = np.array([[0,0],[1,0],[0,1]] + [[0.1, 0.1]]*(N-3))
            # Fallback normalization
            hull_area = _convex_hull_area(best_points)
            if hull_area > 0:
                best_points = best_points / np.sqrt(hull_area)

    return best_points

# EVOLVE-BLOCK-END
