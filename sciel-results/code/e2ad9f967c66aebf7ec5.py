import numpy as np
import scipy.optimize as opt
import time
import itertools

N = 11

# Triangle vertices
A = np.array([0.0, 0.0])
B = np.array([1.0, 0.0])
C = np.array([0.5, np.sqrt(3) / 2])

def bary_to_cart(u, v):
    """
    Convert barycentric coordinates (u, v) relative to A, B, C to Cartesian.
    w = 1 - u - v
    P = u*A + v*B + w*C
    """
    w = 1.0 - u - v
    # A=(0,0), B=(1,0), C=(0.5, sqrt(3)/2)
    x = u * A[0] + v * B[0] + w * C[0]
    y = u * A[1] + v * B[1] + w * C[1]
    return np.array([x, y])

def is_inside(p):
    """
    Check if point p is inside the triangle (with small tolerance).
    """
    x, y = p
    tol = 1e-9
    if y < -tol:
        return False
    # Line AC: y = sqrt(3) * x  => y - sqrt(3)*x <= 0
    if y - np.sqrt(3) * x > tol:
        return False
    # Line BC: y = -sqrt(3) * (x - 1) = sqrt(3) * (1 - x) => y + sqrt(3)*x - sqrt(3) <= 0
    if y + np.sqrt(3) * x - np.sqrt(3) > tol:
        return False
    return True

def min_triangle_area(points):
    """
    Calculate the minimum area of any triangle formed by 3 points.
    points: array of shape (N, 2)
    """
    min_area = float('inf')
    # Iterate over all combinations of 3 points
    for i, j, k in itertools.combinations(range(N), 3):
        p1 = points[i]
        p2 = points[j]
        p3 = points[k]
        # Area = 0.5 * |cross(p2-p1, p3-p1)|
        v1 = p2 - p1
        v2 = p3 - p1
        area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
        if area < min_area:
            min_area = area
    return min_area

def objective(x_flat):
    """
    Objective function for minimization.
    Returns negative of the minimum triangle area.
    Returns a large penalty if points are outside the triangle.
    """
    points = x_flat.reshape(N, 2)
    
    # Check if all points are inside
    for p in points:
        if not is_inside(p):
            return 1e6
    
    return -min_triangle_area(points)

def construct():
    start_time = time.time()
    # Leave 10 seconds buffer
    deadline = start_time + 110.0
    
    best_score = -1e9
    best_points = None
    
    # 1. Generate Initial Guesses
    
    init_guesses = []
    
    # Strategy A: Greedy Farthest Point Sampling on a Grid
    # Create a grid of barycentric coordinates
    grid_u = np.linspace(0, 1, 15)
    grid_v = np.linspace(0, 1, 15)
    
    grid_points = []
    for u in grid_u:
        for v in grid_v:
            if u >= 0 and v >= 0 and u + v <= 1:
                grid_points.append(bary_to_cart(u, v))
    
    # Select N points greedily
    selected = []
    # Start with the centroid
    centroid = (A + B + C) / 3.0
    selected.append(centroid)
    
    for _ in range(N - 1):
        max_min_dist = -1
        best_pt = None
        for pt in grid_points:
            # Calculate min distance to already selected points
            min_d = min(np.linalg.norm(pt - s) for s in selected)
            if min_d > max_min_dist:
                max_min_dist = min_d
                best_pt = pt
        if best_pt is not None:
            selected.append(best_pt)
    
    init_guesses.append(np.array(selected).flatten())
    
    # Strategy B: Random Points
    # Use fixed seed for reproducibility in the "random" part? 
    # The task says "random.Random(7)" in seed, but we are rewriting.
    # We can use np.random with a seed for consistency across runs if needed, 
    # but diverse random starts are better for global search.
    rng = np.random.default_rng(42)
    
    for _ in range(30):
        pts = []
        for _ in range(N):
            r1, r2 = rng.random(2)
            if r1 + r2 > 1:
                r1, r2 = 1 - r1, 1 - r2
            pts.append(bary_to_cart(r1, r2))
        init_guesses.append(np.array(pts).flatten())
        
    # Strategy C: Jittered Grid
    # Take the greedy selection and add small noise
    base = np.array(selected)
    for _ in range(10):
        noise = rng.normal(0, 0.05, size=(N, 2))
        jittered = base + noise
        # Project back to triangle if necessary? 
        # The objective function handles invalid points with penalty, 
        # but starting close to valid is better.
        # We'll just let the optimizer handle it.
        init_guesses.append(jittered.flatten())

    # 2. Optimization Loop
    
    for i, guess in enumerate(init_guesses):
        if time.time() > deadline:
            break
            
        try:
            # Powell method is robust for non-smooth functions
            # maxiter and tolerances tuned for speed
            res = opt.minimize(
                objective, 
                guess, 
                method='Powell',
                options={'maxiter': 200, 'xtol': 1e-7, 'ftol': 1e-7}
            )
            
            if time.time() > deadline:
                break
            
            # Check if result is valid (not a penalty)
            if res.fun < 1e5:
                current_score = -res.fun
                if current_score > best_score:
                    best_score = current_score
                    best_points = res.x.reshape(N, 2)
                    
        except Exception:
            continue

    # Fallback if no valid point found
    if best_points is None:
        # Use the first guess
        best_points = init_guesses[0].reshape(N, 2)
        
    return best_points
