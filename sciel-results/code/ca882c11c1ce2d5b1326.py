"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize
import itertools

N = 11
T0 = time.time()
TIME_LIMIT = 140  # safety margin before 150s kill

def triangle_area(p1, p2, p3):
    return 0.5 * abs(np.cross(p2 - p1, p3 - p1))

def min_triangle_area(points):
    """Compute the minimum triangle area among all C(11,3)=165 triangles."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                a = triangle_area(points[i], points[j], points[k])
                if a < min_area:
                    min_area = a
    return min_area

def in_triangle(p):
    """Check if point p is inside or on the equilateral triangle (0,0), (1,0), (0.5, sqrt(3)/2)."""
    x, y = p
    h = np.sqrt(3) / 2
    if y < -1e-10 or y > h + 1e-10:
        return False
    if x < -1e-10 or x > 1 + 1e-10:
        return False
    # Check left edge: y <= sqrt(3)*x
    if y > np.sqrt(3) * x + 1e-10:
        return False
    # Check right edge: y <= sqrt(3)*(1-x)
    if y > np.sqrt(3) * (1 - x) + 1e-10:
        return False
    return True

def project_into_triangle(p):
    """Project point into the triangle if outside."""
    x, y = p
    h = np.sqrt(3) / 2
    # Clamp y
    y = max(0, min(h, y))
    # For given y, x must be between y/sqrt(3) and 1 - y/sqrt(3)
    x_min = y / np.sqrt(3)
    x_max = 1 - y / np.sqrt(3)
    x = max(x_min, min(x_max, x))
    return np.array([x, y])

def objective_flat(flat_points):
    """Objective for DE: negative of min triangle area (we maximize)."""
    points = flat_points.reshape(-1, 2)
    # Project any out-of-bounds points
    points = np.array([project_into_triangle(p) for p in points])
    return -min_triangle_area(points)

def soft_min_objective(flat_points, beta=50):
    """Smooth surrogate using log-sum-exp for the min area."""
    points = flat_points.reshape(-1, 2)
    points = np.array([project_into_triangle(p) for p in points])
    n = len(points)
    areas = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                areas.append(triangle_area(points[i], points[j], points[k]))
    areas = np.array(areas)
    # Soft min: -log-sum-exp(-beta * areas) / beta approximates min(areas)
    # To avoid overflow, shift by max
    shifted = -beta * areas
    max_shifted = np.max(shifted)
    log_sum_exp = max_shifted + np.log(np.sum(np.exp(shifted - max_shifted)))
    soft_min_val = -log_sum_exp / beta
    return -soft_min_val  # We want to maximize, so minimize negative

def refine_local(points, max_iter=500):
    """Local refinement using Nelder-Mead on the smooth surrogate."""
    flat = points.flatten()
    try:
        result = minimize(
            soft_min_objective,
            flat,
            method='Nelder-Mead',
            options={'maxiter': max_iter, 'xatol': 1e-10, 'fatol': 1e-12}
        )
        refined = result.x.reshape(-1, 2)
        refined = np.array([project_into_triangle(p) for p in refined])
        return refined
    except:
        return points

def construct():
    global T0
    start_time = time.time()
    
    # Bounds for each coordinate
    h = np.sqrt(3) / 2
    bounds = [(0, 1), (0, h)] * N
    
    # Initial population: diverse points
    # Mix of random and structured initialization
    np.random.seed(42)
    
    # Try multiple DE runs with different seeds and strategies
    best_points = None
    best_score = -float('inf')
    
    # Precompute some structured initial guesses
    # Grid-like points
    def make_grid_init(seed):
        rng = np.random.RandomState(seed)
        pts = []
        # Place points in a roughly uniform pattern
        # Use jittered grid
        rows = 4
        cols = 3
        for r in range(rows):
            for c in range(cols):
                if len(pts) >= N:
                    break
                y = (r + 0.5) / rows * h
                x_min = y / np.sqrt(3)
                x_max = 1 - y / np.sqrt(3)
                x = x_min + (c + 0.5) / cols * (x_max - x_min)
                x += rng.uniform(-0.05, 0.05)
                y += rng.uniform(-0.05, 0.05)
                pts.append(project_into_triangle(np.array([x, y])))
        # Add remaining random
        while len(pts) < N:
            x = rng.uniform(0.1, 0.9)
            y = rng.uniform(0.05, h - 0.05)
            pts.append(project_into_triangle(np.array([x, y])))
        return np.array(pts[:N])
    
    # Run DE with a few different initial populations
    for trial in range(3):
        if time.time() - start_time > TIME_LIMIT - 20:
            break
        
        # Different initialization strategies
        if trial == 0:
            init = make_grid_init(1)
        elif trial == 1:
            init = make_grid_init(2)
        else:
            init = make_grid_init(3)
        
        # Differential Evolution
        try:
            de_result = differential_evolution(
                objective_flat,
                bounds,
                seed=trial,
                maxiter=200,
                popsize=15,
                tol=1e-8,
                mutation=(0.5, 1.0),
                recombination=0.8,
                init=init,
                polish=False,  # We'll do our own polish
                updating='deferred',
                workers=1
            )
            
            if time.time() - start_time > TIME_LIMIT - 15:
                break
                
            candidate = de_result.x.reshape(-1, 2)
            candidate = np.array([project_into_triangle(p) for p in candidate])
            
            # Local refinement
            refined = refine_local(candidate, max_iter=300)
            score = min_triangle_area(refined)
            
            if score > best_score:
                best_score = score
                best_points = refined
                
        except Exception as e:
            pass
    
    # If DE didn't find anything good, try a simple local search from random starts
    if best_points is None or time.time() - start_time < TIME_LIMIT - 30:
        for rs in range(5):
            if time.time() - start_time > TIME_LIMIT - 20:
                break
            np.random.seed(100 + rs)
            pts = []
            for _ in range(N):
                x = np.random.uniform(0.1, 0.9)
                y = np.random.uniform(0.05, h - 0.05)
                pts.append(project_into_triangle(np.array([x, y])))
            pts = np.array(pts)
            refined = refine_local(pts, max_iter=200)
            score = min_triangle_area(refined)
            if score > best_score:
                best_score = score
                best_points = refined
    
    if best_points is None:
        # Fallback: simple grid
        best_points = make_grid_init(1)
    
    return best_points
