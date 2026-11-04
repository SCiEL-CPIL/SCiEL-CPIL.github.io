"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize, differential_evolution
from itertools import combinations

def _min_triangle_area(pts):
    """Compute minimum triangle area for n points."""
    n = len(pts)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                ax, ay = pts[i]
                bx, by = pts[j]
                cx, cy = pts[k]
                area = 0.5 * abs((bx - ax) * (cy - ay) - (by - ay) * (cx - ax))
                if area < min_area:
                    min_area = area
    return min_area

def _hull_area(pts):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 1.0

def _score(pts):
    """Score = min triangle area / hull area."""
    hull_a = _hull_area(pts)
    if hull_a <= 1e-15:
        return 0.0
    min_tri = _min_triangle_area(pts)
    return min_tri / hull_a

def _all_triangle_areas(pts):
    """Compute all triangle areas as a flat array."""
    n = len(pts)
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                ax, ay = pts[i]
                bx, by = pts[j]
                cx, cy = pts[k]
                area = 0.5 * abs((bx - ax) * (cy - ay) - (by - ay) * (cx - ax))
                areas.append(area)
    return np.array(areas)

def _soft_min_score(pts, beta):
    """
    Smooth approximation of min triangle area using log-sum-exp.
    soft_min(x) = -log(sum(exp(-x/beta))) / beta
    As beta -> 0, this approaches min(x).
    """
    areas = _all_triangle_areas(pts)
    if len(areas) == 0:
        return 0.0
    scaled = -areas / beta
    max_scaled = np.max(scaled)
    log_sum_exp = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    soft_min = -log_sum_exp * beta
    return soft_min

def _neg_soft_min_score(x, beta):
    """Objective to minimize: negative soft-min score."""
    pts = x.reshape((-1, 2))
    return -_soft_min_score(pts, beta)

def _grad_neg_soft_min(x, beta):
    """
    Gradient of negative soft-min score with respect to point coordinates.
    """
    pts = x.reshape((-1, 2))
    n = len(pts)
    
    areas = np.zeros(1)
    grads = []
    
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                ax, ay = pts[i]
                bx, by = pts[j]
                cx, cy = pts[k]
                area = 0.5 * abs((bx - ax) * (cy - ay) - (by - ay) * (cx - ax))
                areas = np.append(areas, area)
                
                cross = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
                sign = 1.0 if cross >= 0 else -1.0
                
                g_ax = -0.5 * sign * (cy - ay)
                g_ay = 0.5 * sign * (cx - ax)
                g_bx = 0.5 * sign * (cy - ay)
                g_by = -0.5 * sign * (cx - ax)
                g_cx = 0.5 * sign * (by - ay)
                g_cy = -0.5 * sign * (bx - ax)
                
                grad_vec = np.zeros(2 * n)
                grad_vec[2*i] = g_ax
                grad_vec[2*i+1] = g_ay
                grad_vec[2*j] = g_bx
                grad_vec[2*j+1] = g_by
                grad_vec[2*k] = g_cx
                grad_vec[2*k+1] = g_cy
                grads.append(grad_vec)
    
    areas = areas[1:]
    grads = np.array(grads)
    
    scaled = -areas / beta
    max_scaled = np.max(scaled)
    exp_scaled = np.exp(scaled - max_scaled)
    sum_exp = np.sum(exp_scaled)
    weights = exp_scaled / sum_exp
    
    total_grad = np.sum(weights[:, np.newaxis] * grads, axis=0)
    
    return total_grad

def _fourier_to_points(fourier_params, n=N):
    """
    Convert Fourier series parameters to n points on a convex polygon.
    
    The radius function is: r(theta) = r0 + sum_{k=1}^{K} (a_k * cos(k*theta) + b_k * sin(k*theta))
    
    Parameters:
    - fourier_params: array of [r0, a1, b1, a2, b2, ..., aK, bK]
    - n: number of points
    
    Returns:
    - points: (n, 2) array
    """
    r0 = fourier_params[0]
    K = (len(fourier_params) - 1) // 2
    
    # Generate n evenly-spaced angles
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    
    # Compute radius at each angle
    r = np.full(n, r0)
    for k in range(1, K + 1):
        a_k = fourier_params[2 * k - 1]
        b_k = fourier_params[2 * k]
        r += a_k * np.cos(k * theta) + b_k * np.sin(k * theta)
    
    # Convert to Cartesian
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    
    return np.column_stack((x, y))

def _fourier_objective(fourier_params, beta=0.01):
    """
    Objective function for differential evolution.
    Minimizes negative soft-min score of points generated from Fourier parameters.
    """
    pts = _fourier_to_points(fourier_params)
    
    # Ensure points are in a reasonable range
    if np.any(np.isnan(pts)) or np.any(np.isinf(pts)):
        return 1e6
    
    # Normalize to unit area
    hull_a = _hull_area(pts)
    if hull_a <= 1e-15:
        return 1e6
    
    # Score = min triangle area / hull area
    min_tri = _min_triangle_area(pts)
    score = min_tri / hull_a
    
    # Return negative score (we minimize)
    return -score

def _cartesian_refine(pts, max_time=30.0):
    """
    Refine points using L-BFGS-B with soft-min surrogate.
    """
    start_time = time.time()
    deadline = start_time + max_time
    
    bounds = [(0.0, 1.0)] * (N * 2)
    x0 = pts.flatten()
    best_pts = pts.copy()
    best_score = _score(pts)
    
    beta_schedule = [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01]
    
    for beta in beta_schedule:
        if time.time() > deadline:
            break
        
        try:
            result = minimize(
                _neg_soft_min_score,
                x0,
                args=(beta,),
                jac=_grad_neg_soft_min,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 20}
            )
            
            new_pts = result.x.reshape((-1, 2))
            new_score = _score(new_pts)
            if new_score > best_score:
                best_score = new_score
                best_pts = new_pts.copy()
                x0 = new_pts.flatten()
            else:
                x0 = result.x.copy()
                
        except Exception:
            pass
    
    return best_pts, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    # Phase 1: Fourier series parameterization with differential evolution
    # Use K=2 (4 Fourier coefficients + r0 = 5 parameters)
    K = 2
    n_params = 1 + 2 * K  # r0, a1, b1, a2, b2
    
    # Bounds for Fourier parameters
    # r0 should be positive and reasonable
    # a_k, b_k should be small enough to maintain convexity
    bounds = [
        (0.2, 0.6),  # r0
        (-0.15, 0.15),  # a1
        (-0.15, 0.15),  # b1
        (-0.08, 0.08),  # a2
        (-0.08, 0.08),  # b2
    ]
    
    best_pts = None
    best_score = 0.0
    
    # Run differential evolution with multiple seeds
    if time.time() < deadline - 60:
        for seed in [42, 123, 456]:
            if time.time() > deadline - 40:
                break
            
            try:
                de_result = differential_evolution(
                    _fourier_objective,
                    bounds,
                    seed=seed,
                    maxiter=50,
                    popsize=15,
                    tol=1e-6,
                    polish=True,
                    init='sobol'
                )
                
                if de_result.success:
                    pts = _fourier_to_points(de_result.x)
                    score = _score(pts)
                    if score > best_score:
                        best_score = score
                        best_pts = pts.copy()
                        
            except Exception:
                pass
    
    # If DE didn't find anything good, use a regular polygon as fallback
    if best_pts is None or best_score < 0.01:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_pts = np.column_stack((0.5 + 0.45 * np.cos(angles), 
                                     0.5 + 0.45 * np.sin(angles)))
        best_score = _score(best_pts)
    
    # Phase 2: Cartesian refinement with L-BFGS-B
    if time.time() < deadline - 20:
        refined_pts, refined_score = _cartesian_refine(best_pts, max_time=deadline - time.time() - 10)
        if refined_score > best_score:
            best_score = refined_score
            best_pts = refined_pts.copy()
    
    # Phase 3: Try a few random restarts if time remains
    if time.time() < deadline - 30:
        rng = np.random.RandomState(42)
        for _ in range(3):
            if time.time() > deadline - 15:
                break
            
            # Generate random convex polygon
            angles = np.sort(rng.uniform(0, 2 * np.pi, N))
            radii = rng.uniform(0.3, 0.5, N)
            x = 0.5 + radii * np.cos(angles)
            y = 0.5 + radii * np.sin(angles)
            random_pts = np.column_stack((x, y))
            
            random_score = _score(random_pts)
            if random_score > best_score:
                best_score = random_score
                best_pts = random_pts.copy()
            
            # Quick refinement
            if time.time() < deadline - 10:
                refined_pts, refined_score = _cartesian_refine(best_pts, max_time=min(10.0, deadline - time.time() - 5))
                if refined_score > best_score:
                    best_score = refined_score
                    best_pts = refined_pts.copy()
    
    return best_pts
# EVOLVE-BLOCK-END
