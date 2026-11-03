import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import itertools

N = 13

# EVOLVE-BLOCK-START

def _compute_score(points):
    """Compute the minimum triangle area divided by the convex hull area."""
    n = len(points)
    
    # Compute convex hull area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume  # In 2D, volume is area
    except Exception:
        return 0.0
    
    if hull_area <= 1e-12:
        return 0.0
    
    # Compute minimum triangle area
    min_tri_area = float('inf')
    # Vectorized approach for speed if possible, but 286 triangles is small
    # Loop is fine, but let's try to be efficient
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                # Area = 0.5 * |cross(p2-p1, p3-p1)|
                v1 = p2 - p1
                v2 = p3 - p1
                area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                if area < min_tri_area:
                    min_tri_area = area
    
    return min_tri_area / hull_area

def _neg_score(points):
    """Negative score for minimization."""
    return -_compute_score(points)

def _gen_1_plus_12(params):
    """1 center, 12 points on a circle."""
    R = params[0]
    # Ensure R is positive
    R = max(R, 0.1)
    points = np.zeros((13, 2))
    # Center
    points[0] = [0, 0]
    # 12 points
    angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    for i in range(12):
        points[i+1] = [R * np.cos(angles[i]), R * np.sin(angles[i])]
    return points

def _gen_1_plus_6_plus_6(params):
    """1 center, 6 inner, 6 outer."""
    R1 = max(params[0], 0.1)
    R2 = max(params[1], R1 + 0.1)
    phi1 = params[2]
    phi2 = params[3]
    
    points = np.zeros((13, 2))
    points[0] = [0, 0]
    
    angles1 = np.linspace(0, 2 * np.pi, 6, endpoint=False) + phi1
    for i in range(6):
        points[i+1] = [R1 * np.cos(angles1[i]), R1 * np.sin(angles1[i])]
        
    angles2 = np.linspace(0, 2 * np.pi, 6, endpoint=False) + phi2
    for i in range(6):
        points[i+7] = [R2 * np.cos(angles2[i]), R2 * np.sin(angles2[i])]
        
    return points

def _gen_13_gon(params):
    """Regular 13-gon."""
    R = max(params[0], 0.1)
    angles = np.linspace(0, 2 * np.pi, 13, endpoint=False)
    points = np.column_stack([R * np.cos(angles), R * np.sin(angles)])
    return points

def _optimize_structure(generator, initial_params, bounds=None, time_limit=5.0):
    """Optimize a parameterized structure."""
    start_time = time.time()
    
    # Initial evaluation
    init_points = generator(initial_params)
    init_score = _compute_score(init_points)
    
    def objective(params):
        if time.time() - start_time > time_limit:
            return 1e9
        try:
            pts = generator(params)
            return _neg_score(pts)
        except Exception:
            return 1e9
            
    # Use Powell method which is robust for small dimensions
    res = minimize(objective, initial_params, method='Powell', 
                   options={'maxiter': 200, 'maxfev': 500})
    
    if time.time() - start_time > time_limit:
        return init_points, init_score
        
    best_params = res.x
    best_points = generator(best_params)
    best_score = _compute_score(best_points)
    
    # If optimization failed or got worse, return initial
    if best_score < init_score:
        return init_points, init_score
        
    return best_points, best_score

def _local_search(points, max_iter=500, step_size=0.05, time_limit=10.0):
    """Simple local search with random perturbations."""
    start_time = time.time()
    current = points.copy()
    current_score = _compute_score(current)
    best = current.copy()
    best_score = current_score
    
    rng = np.random.RandomState(42)
    
    for i in range(max_iter):
        if time.time() - start_time > time_limit:
            break
            
        # Perturb a random point
        idx = rng.randint(0, N)
        new_points = current.copy()
        # Scale step size down over time
        current_step = step_size * (1.0 / (1.0 + i/100.0))
        new_points[idx] += rng.normal(0, current_step, 2)
        
        new_score = _compute_score(new_points)
        
        if new_score > current_score:
            current = new_points
            current_score = new_score
            if current_score > best_score:
                best = current.copy()
                best_score = current_score
        else:
            # With some probability, accept worse moves to escape local minima?
            # For simplicity and speed, let's stick to hill climbing for now.
            # Or use a simple SA acceptance.
            # Given the tight time, hill climbing is safer.
            pass
            
    return best, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    best_points = None
    best_score = -float('inf')
    
    # Strategy 1: Optimize Symmetric Structures
    
    # Structure A: 1 + 12
    try:
        if time.time() < deadline - 20:
            p, s = _optimize_structure(_gen_1_plus_12, [0.5], time_limit=10.0)
            if s > best_score:
                best_score = s
                best_points = p
    except Exception:
        pass
        
    # Structure B: 1 + 6 + 6
    try:
        if time.time() < deadline - 20:
            # Try a few initial guesses for 1+6+6
            inits = [
                [0.3, 0.6, 0.0, 0.0],
                [0.4, 0.7, np.pi/6, 0.0],
                [0.5, 0.8, 0.0, np.pi/6]
            ]
            for init in inits:
                if time.time() > deadline - 20:
                    break
                p, s = _optimize_structure(_gen_1_plus_6_plus_6, init, time_limit=10.0)
                if s > best_score:
                    best_score = s
                    best_points = p
    except Exception:
        pass
        
    # Structure C: 13-gon
    try:
        if time.time() < deadline - 20:
            p, s = _optimize_structure(_gen_13_gon, [0.5], time_limit=5.0)
            if s > best_score:
                best_score = s
                best_points = p
    except Exception:
        pass
        
    # If we didn't find anything, use a default
    if best_points is None:
        # Fallback: Fermat spiral
        i = np.arange(N)
        r = np.sqrt(i / N)
        theta = i * 2.399963229728653
        best_points = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        best_points -= np.mean(best_points, axis=0)
        best_score = _compute_score(best_points)
        
    # Strategy 2: Refine the best symmetric solution with a general local search
    # This allows breaking symmetry if it helps
    try:
        if time.time() < deadline - 10:
            refined_points, refined_score = _local_search(
                best_points, 
                max_iter=2000, 
                step_size=0.05, 
                time_limit=deadline - time.time() - 5
            )
            if refined_score > best_score:
                best_score = refined_score
                best_points = refined_points
    except Exception:
        pass
        
    # Final check
    if time.time() > deadline - 2:
        return best_points
        
    # One more very short refinement pass with smaller steps
    try:
        if time.time() < deadline - 2:
            final_points, final_score = _local_search(
                best_points, 
                max_iter=1000, 
                step_size=0.01, 
                time_limit=deadline - time.time() - 1
            )
            if final_score > best_score:
                best_points = final_points
                best_score = final_score
    except Exception:
        pass
        
    return best_points

# EVOLVE-BLOCK-END
