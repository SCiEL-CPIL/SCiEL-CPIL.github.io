import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize, differential_evolution
import time
import itertools

N = 14

def _calculate_score(points):
    """
    Calculate the score: min triangle area / convex hull area.
    points: (N, 2) array
    """
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area < 1e-12:
            return 0.0
        
        min_area = np.inf
        # Compute all triangle areas
        for i in range(N):
            for j in range(i+1, N):
                for k in range(j+1, N):
                    p1 = points[i]
                    p2 = points[j]
                    p3 = points[k]
                    # Area = 0.5 * |cross(p2-p1, p3-p1)|
                    v1 = p2 - p1
                    v2 = p3 - p1
                    area = 0.5 * abs(v1[0]*v2[1] - v1[1]*v2[0])
                    if area < min_area:
                        min_area = area
        
        return min_area / hull_area
    except Exception:
        return 0.0

def _score_ring_params(radii, angles):
    """
    Score a configuration defined by radii (N,) and angles (N,).
    Points are (radii[i] * cos(angles[i]), radii[i] * sin(angles[i])).
    """
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    points = np.column_stack((x, y))
    return _calculate_score(points)

def _optimize_ring():
    """
    Optimize the polar coordinates (radii, angles) directly.
    We use differential evolution to search the 2N-dimensional space.
    To handle the symmetry and the fact that absolute angle doesn't matter,
    we can fix the first angle to 0.
    However, DE works well on bounded domains.
    Domain:
    radii: [0.1, 1.0] (avoid 0 to keep hull area reasonable, though 0 is allowed, 1.0 is max for unit area normalization? 
    Wait, the problem says "normalized to unit area". The evaluator normalizes by the point set convex hull area.
    So the scale of the points doesn't matter for the score!
    min_area / hull_area is scale-invariant.
    So we can fix the scale. Let's say the max radius is 1.0.
    Actually, since it's scale invariant, we can just constrain radii to [0.0, 1.0].
    If all points are 0, hull area is 0.
    Let's use radii in [0.0, 1.0] and angles in [0.0, 2*pi].
    
    To reduce dimensionality and exploit symmetry:
    1. Fix angle[0] = 0.
    2. Sort points by angle? No, that breaks differentiability if we do it inside the objective, but DE doesn't need differentiability.
    However, sorting makes the landscape rugged.
    
    Alternative:
    Use the fact that the optimal configuration likely has points on the boundary or near it?
    Actually, for maximizing min triangle area, points should be spread out.
    
    Let's try a simple DE on 2N variables.
    x = [r0, ..., r13, a0, ..., a13]
    Bounds: r in [0.0, 1.0], a in [0.0, 2*pi].
    
    To avoid the trivial solution where points collapse, we rely on the score function returning 0 for degenerate hulls.
    
    Since the score is scale invariant, the optimizer might wander in scale.
    We can fix the scale by normalizing the points such that the max radius is 1.0 inside the objective?
    No, that makes the function non-smooth at the max radius change.
    But DE is robust to non-smoothness.
    
    Actually, a better approach:
    Since scale doesn't matter, we can fix the convex hull area to be 1.0? No, that's hard to enforce.
    Let's just let DE search in [0,1] for radii and [0, 2pi] for angles.
    The best score will be found regardless of scale.
    
    One issue: The permutation of points.
    The score is invariant to permutation of points.
    DE searches in a fixed order.
    This creates many local minima corresponding to permutations.
    To mitigate this, we can sort the points by angle in the objective function.
    This makes the objective function well-defined (single-valued) for a set of points, regardless of their input order.
    Sorting is O(N log N) which is fast for N=14.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin
    
    def objective(params):
        if time.time() > deadline:
            return 0.0
        
        radii = params[:N]
        angles = params[N:]
        
        # Normalize scale: set max radius to 1.0 to remove scale ambiguity
        max_r = np.max(radii)
        if max_r < 1e-9:
            return 0.0
        radii = radii / max_r
        
        x = radii * np.cos(angles)
        y = radii * np.sin(angles)
        points = np.column_stack((x, y))
        
        # Sort by angle to remove permutation symmetry
        angles_sorted = np.arctan2(y, x)
        # Handle -pi to pi wrap for sorting
        angles_sorted = (angles_sorted + np.pi) % (2 * np.pi) - np.pi
        sort_idx = np.argsort(angles_sorted)
        points = points[sort_idx]
        
        return -_calculate_score(points)
    
    # Initial bounds
    # Radii: [0, 1]
    # Angles: [0, 2*pi]
    bounds = [(0.0, 1.0)] * N + [(0.0, 2 * np.pi)] * N
    
    # Use Differential Evolution
    # popsize=15 is default. Let's use 20.
    # maxiter: DE can be slow. Let's limit iterations based on time.
    # We'll run DE for a fixed number of iterations or until time runs out.
    # DE has a 'maxiter' option.
    # Let's estimate: 14*2 = 28 variables.
    # popsize 20 -> 20 evaluations per generation.
    # 100 generations -> 2000 evaluations.
    # Each evaluation is ~1ms (14 points, 364 triangles, hull calc).
    # 2000 * 1ms = 2s.
    # So we can afford many more iterations.
    # Let's try maxiter=500.
    
    try:
        res = differential_evolution(
            objective, 
            bounds, 
            seed=42, 
            maxiter=200, 
            popsize=20, 
            tol=1e-6,
            polish=False, # We'll polish manually
            init='latinhypercube',
            workers=1
        )
        best_params = res.x
        best_score = -res.fun
    except Exception as e:
        # Fallback to a heuristic
        best_params = np.zeros(2*N)
        best_params[:N] = 1.0
        best_params[N:] = np.linspace(0, 2*np.pi, N, endpoint=False)
        best_score = -objective(best_params)
        
    # Polish with Nelder-Mead on the sorted-parameter space?
    # The sorting makes the function non-differentiable and discontinuous in the parameter space (permutations swap).
    # So standard gradient methods or even NM might struggle if points cross in angle.
    # However, near the optimum, the order is likely stable.
    # Let's try a local NM polish on the original (unsorted) parameters, starting from the best DE result.
    # But we need to be careful: the objective used for DE sorts.
    # If we polish, we should use the same objective (with sorting) to ensure we are minimizing the same thing.
    # NM on a function with sorting: if the order doesn't change during the steps, it's smooth.
    # If order changes, it's a jump. NM might handle it if the jump is small or if it steps back.
    
    if time.time() < deadline - 5.0:
        try:
            res_polish = minimize(
                objective, 
                best_params, 
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            if -res_polish.fun > best_score:
                best_params = res_polish.x
                best_score = -res_polish.fun
        except Exception:
            pass
            
    # Convert back to points
    radii = best_params[:N]
    angles = best_params[N:]
    max_r = np.max(radii)
    if max_r < 1e-9:
        # Fallback
        radii = np.ones(N)
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    else:
        radii = radii / max_r
        
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    points = np.column_stack((x, y))
    
    return points, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    points, score = _optimize_ring()
    
    # Final verification and potential fallback
    final_score = _calculate_score(points)
    if final_score <= 0.0:
        # Regular polygon fallback
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        points = np.column_stack((np.cos(angles), np.sin(angles)))
        
    return points
