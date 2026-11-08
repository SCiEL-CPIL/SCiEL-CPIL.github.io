import numpy as np
from scipy.optimize import minimize
import time
import itertools

N = 11
SQRT3 = np.sqrt(3)
H = SQRT3 / 2  # height of unit base equilateral triangle

def triangle_area(p, q, r):
    """Compute area of triangle formed by points p, q, r."""
    return 0.5 * abs((q[0] - p[0]) * (r[1] - p[1]) - (r[0] - p[0]) * (q[1] - p[1]))

def min_triangle_area(points):
    """Compute the minimum area over all C(n,3) triangles."""
    n = len(points)
    min_a = float('inf')
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                a = triangle_area(points[i], points[j], points[k])
                if a < min_a:
                    min_a = a
    return min_a

def is_inside_triangle(p):
    """Check if point p is inside the equilateral triangle."""
    x, y = p
    return (y >= -1e-9) and (y <= H + 1e-9) and (x >= y/SQRT3 - 1e-9) and (x <= 1 - y/SQRT3 + 1e-9)

def penalty_for_outside(points):
    """Penalty for points outside the triangle."""
    pen = 0.0
    for p in points:
        x, y = p
        if y < 0:
            pen += abs(y) * 1000
        if y > H:
            pen += (y - H) * 1000
        if x < y/SQRT3:
            pen += (y/SQRT3 - x) * 1000
        if x > 1 - y/SQRT3:
            pen += (x - (1 - y/SQRT3)) * 1000
    return pen

def objective(params):
    """Negative of min triangle area + penalty for being outside."""
    points = params.reshape(N, 2)
    pen = penalty_for_outside(points)
    if pen > 1e-6:
        return 1.0 + pen
    return -min_triangle_area(points)

def generate_structured_guess(seed):
    """Generate a structured initial guess based on triangular lattice."""
    rng = np.random.RandomState(seed)
    
    # Strategy: Use a triangular lattice-like arrangement
    # For 11 points, try rows of 4, 3, 2, 1, 1 or similar
    # Let's try a few different row configurations
    
    configs = [
        [4, 3, 2, 1, 1],
        [3, 3, 2, 2, 1],
        [5, 2, 2, 1, 1],
        [4, 4, 2, 1],
        [3, 4, 3, 1],
        [2, 3, 3, 2, 1],
        [4, 3, 3, 1],
        [5, 3, 2, 1],
    ]
    
    config = configs[seed % len(configs)]
    n_rows = len(config)
    points = []
    
    for i, count in enumerate(config):
        # Place rows from bottom to top
        y_frac = (i + 0.5) / n_rows
        y = y_frac * H
        x_min = y / SQRT3
        x_max = 1 - y / SQRT3
        width = x_max - x_min
        
        if count == 1:
            x = (x_min + x_max) / 2
            points.append([x, y])
        else:
            for j in range(count):
                t = (j + 0.5) / count
                x = x_min + t * width
                points.append([x, y])
    
    # Add small random perturbation to break symmetry
    points = np.array(points)
    points += rng.normal(0, 0.01, points.shape)
    
    return points

def generate_random_guess(seed):
    """Generate random points inside the triangle."""
    rng = np.random.RandomState(seed)
    pts = []
    while len(pts) < N:
        u, v = rng.random(), rng.random()
        if u + v < 1:
            a = 1 - u - v
            b = u
            c = v
            x = b * 1 + c * 0.5
            y = c * H
            pts.append([x, y])
    return np.array(pts)

def local_refinement(points, start_time, deadline):
    """Perform local optimization starting from given points."""
    params0 = points.flatten()
    
    try:
        result = minimize(
            objective,
            params0,
            method='SLSQP',
            options={'maxiter': 300, 'ftol': 1e-10},
            tol=1e-10
        )
        
        if result.fun < 1.0:  # No penalty
            refined_points = result.x.reshape(N, 2)
            # Clamp points to be inside triangle
            for i in range(N):
                x, y = refined_points[i]
                if y < 0:
                    refined_points[i, 1] = 0
                if y > H:
                    refined_points[i, 1] = H
                x_min = refined_points[i, 1] / SQRT3
                x_max = 1 - refined_points[i, 1] / SQRT3
                if refined_points[i, 0] < x_min:
                    refined_points[i, 0] = x_min
                if refined_points[i, 0] > x_max:
                    refined_points[i, 0] = x_max
            
            if all(is_inside_triangle(p) for p in refined_points):
                return refined_points
        return None
    except Exception:
        return None

def construct():
    start_time = time.time()
    deadline = 150.0 - 15.0  # 15s safety margin
    
    best_points = None
    best_value = -1
    
    # Phase 1: Try structured initial guesses with optimization
    for seed in range(15):
        elapsed = time.time() - start_time
        if elapsed > deadline:
            break
        
        if seed < 8:
            params0 = generate_structured_guess(seed)
        else:
            params0 = generate_random_guess(seed * 13 + 7)
        
        refined = local_refinement(params0, start_time, deadline)
        if refined is not None:
            val = min_triangle_area(refined)
            if val > best_value:
                best_value = val
                best_points = refined.copy()
        
        elapsed = time.time() - start_time
        if elapsed > deadline:
            break
    
    # Phase 2: If we have a good solution, try perturbing it
    if best_points is not None:
        for seed in range(10):
            elapsed = time.time() - start_time
            if elapsed > deadline:
                break
            
            rng = np.random.RandomState(100 + seed)
            # Try different perturbation scales
            scale = 0.02 * (seed + 1)
            perturbation = rng.normal(0, scale, (N, 2))
            perturbed = best_points + perturbation
            
            # Clamp to triangle
            for i in range(N):
                x, y = perturbed[i]
                if y < 0:
                    perturbed[i, 1] = 0
                if y > H:
                    perturbed[i, 1] = H
                x_min = perturbed[i, 1] / SQRT3
                x_max = 1 - perturbed[i, 1] / SQRT3
                if perturbed[i, 0] < x_min:
                    perturbed[i, 0] = x_min
                if perturbed[i, 0] > x_max:
                    perturbed[i, 0] = x_max
            
            refined = local_refinement(perturbed, start_time, deadline)
            if refined is not None:
                val = min_triangle_area(refined)
                if val > best_value:
                    best_value = val
                    best_points = refined.copy()
    
    # Phase 3: Final polish if time remains
    if best_points is not None:
        elapsed = time.time() - start_time
        if elapsed < deadline - 20:
            # Try a few more fine-tuning steps
            for seed in range(5):
                elapsed = time.time() - start_time
                if elapsed > deadline:
                    break
                
                rng = np.random.RandomState(200 + seed)
                small_perturbation = rng.normal(0, 0.005, (N, 2))
                perturbed = best_points + small_perturbation
                
                # Clamp to triangle
                for i in range(N):
                    x, y = perturbed[i]
                    if y < 0:
                        perturbed[i, 1] = 0
                    if y > H:
                        perturbed[i, 1] = H
                    x_min = perturbed[i, 1] / SQRT3
                    x_max = 1 - perturbed[i, 1] / SQRT3
                    if perturbed[i, 0] < x_min:
                        perturbed[i, 0] = x_min
                    if perturbed[i, 0] > x_max:
                        perturbed[i, 0] = x_max
                
                refined = local_refinement(perturbed, start_time, deadline)
                if refined is not None:
                    val = min_triangle_area(refined)
                    if val > best_value:
                        best_value = val
                        best_points = refined.copy()
    
    if best_points is None:
        # Fallback to random
        rng = np.random.RandomState(7)
        points = []
        while len(points) < N:
            u, v = rng.random(), rng.random()
            if u + v < 0.98 and u > 0.01 and v > 0.01:
                points.append([u + 0.5 * v, H * v])
        best_points = np.array(points)
    
    return best_points
