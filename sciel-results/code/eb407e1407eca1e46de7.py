import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import itertools

N = 13

def _min_triangle_area(points):
    # Calculate the minimum area of any triangle formed by 3 distinct points
    # Vectorized approach for speed
    # Number of triples is C(13, 3) = 286. This is small enough for a loop or simple vectorization.
    # Let's use a simple loop for clarity and safety, or vectorize carefully.
    # Vectorized:
    # points shape (N, 2)
    # We need all combinations of 3 indices.
    # Since N is small (13), we can just loop or use a precomputed index list.
    
    # Precompute indices for all triples
    # This is done once outside if we were caching, but here we just compute.
    # Actually, for N=13, a simple loop is fast enough, but let's vectorize for the optimization loop.
    
    # Create a matrix of all pairs? No, we need triples.
    # Let's just use a simple loop inside the objective, it's called many times but N is small.
    # Wait, minimize calls objective many times. 
    # C(13,3) = 286 triangles.
    # Computing area for 286 triangles is very fast.
    
    min_area = np.inf
    for i in range(N-2):
        for j in range(i+1, N-1):
            for k in range(j+1, N):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                # Area = 0.5 * |cross(p2-p1, p3-p1)|
                v1 = p2 - p1
                v2 = p3 - p1
                cross = v1[0]*v2[1] - v1[1]*v2[0]
                area = 0.5 * abs(cross)
                if area < min_area:
                    min_area = area
    return min_area

def _convex_hull_area(points):
    try:
        hull = ConvexHull(points)
        return hull.volume # In 2D, volume is area
    except Exception:
        # If points are collinear or degenerate, return a small number to penalize
        return 1e-10

def _score(points):
    # Score is min triangle area divided by convex hull area
    # We want to MAXIMIZE this.
    # Minimize the negative.
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-9:
        return -1e10
    min_tri = _min_triangle_area(points)
    return min_tri / hull_area

def _neg_score(points):
    return -_score(points)

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin for 150s limit
    
    best_score = -np.inf
    best_points = None
    
    # Generate a regular polygon as a baseline
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    base_points = np.column_stack((np.cos(angles), np.sin(angles)))
    best_score = _score(base_points)
    best_points = base_points.copy()
    
    # Try a few random restarts with Nelder-Mead
    # We will perturb the regular polygon and optimize.
    # Nelder-Mead is good for small dimensional problems (2*N = 26 dims).
    
    rng = np.random.default_rng(42)
    
    num_restarts = 10
    
    for r in range(num_restarts):
        if time.time() > deadline - 20:
            break
            
        # Perturb the regular polygon
        # Add random noise to angles and radii
        noise_angles = rng.uniform(-0.1, 0.1, N)
        noise_radii = rng.uniform(0.9, 1.1, N)
        
        perturbed_angles = angles + noise_angles
        radii = noise_radii
        
        x = radii * np.cos(perturbed_angles)
        y = radii * np.sin(perturbed_angles)
        x0 = np.column_stack((x, y))
        
        # Flatten for optimizer
        x0_flat = x0.flatten()
        
        # Run Nelder-Mead
        # maxiter needs to be reasonable. 
        # Let's try 500 iterations per restart.
        # If it's too slow, we reduce.
        # 26 variables. NM is O(N^2) per iteration roughly.
        
        try:
            res = minimize(_neg_score, x0_flat, method='Nelder-Mead', 
                           options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8})
            
            if res.success:
                opt_points = res.x.reshape(N, 2)
                s = _score(opt_points)
                if s > best_score:
                    best_score = s
                    best_points = opt_points
        except Exception:
            pass
            
        # Check time
        if time.time() > deadline - 5:
            break

    # If we have time, try a simple coordinate ascent or random hill climbing on the best point
    # This can help escape local minima of NM.
    if time.time() < deadline - 10 and best_points is not None:
        current_points = best_points.copy()
        current_score = best_score
        step_size = 0.1
        iterations = 10000
        
        for i in range(iterations):
            if time.time() > deadline - 2:
                break
            
            # Pick a random point and direction
            idx = rng.integers(0, N)
            angle = rng.uniform(0, 2 * np.pi)
            direction = np.array([np.cos(angle), np.sin(angle)])
            
            new_points = current_points.copy()
            new_points[idx] += direction * step_size
            
            new_score = _score(new_points)
            
            if new_score > current_score:
                current_points = new_points
                current_score = new_score
                # Maybe increase step size slightly? No, keep it simple.
                # Actually, let's decay step size slowly to converge
                # step_size = max(0.01, step_size * 0.999)
            else:
                # Decrease step size if we fail?
                # step_size = max(0.01, step_size * 0.9)
                pass
            
            if current_score > best_score:
                best_score = current_score
                best_points = current_points

    return best_points
