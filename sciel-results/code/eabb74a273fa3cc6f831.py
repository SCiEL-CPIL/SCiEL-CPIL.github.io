"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import minimize

# Triangle vertices
A = np.array([0.0, 0.0])
B = np.array([1.0, 0.0])
C = np.array([0.5, np.sqrt(3) / 2])

def min_triangle_area(points):
    """Calculate the minimum area of any triangle formed by 3 points."""
    n = len(points)
    if n < 3:
        return 0.0
    
    # Calculate areas for all combinations of 3 points
    # Using vectorized operations for speed
    # Number of combinations is C(11, 3) = 165
    # We can just loop, it's fast enough
    min_area = float('inf')
    
    # Pre-compute cross products?
    # Area = 0.5 * | (x2-x1)*(y3-y1) - (x3-x1)*(y2-y1) |
    
    for i in range(n - 2):
        p1 = points[i]
        for j in range(i + 1, n - 1):
            p2 = points[j]
            dx12 = p2[0] - p1[0]
            dy12 = p2[1] - p1[1]
            for k in range(j + 1, n):
                p3 = points[k]
                dx13 = p3[0] - p1[0]
                dy13 = p3[1] - p1[1]
                area = 0.5 * abs(dx12 * dy13 - dx13 * dy12)
                if area < min_area:
                    min_area = area
                    if min_area == 0.0:
                        return 0.0
    return min_area

def is_inside_triangle(p):
    """Check if point p is inside the triangle ABC."""
    # Using barycentric coordinates or cross products
    # Cross product sign check
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
    
    d1 = sign(p, A, B)
    d2 = sign(p, B, C)
    d3 = sign(p, C, A)
    
    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    
    return not (has_neg and has_pos)

def project_to_triangle(p):
    """Project point p to the closest point inside the triangle."""
    # Simple clipping
    # 1. y >= 0
    if p[1] < 0:
        p = np.array([p[0], 0.0])
    
    # 2. y <= sqrt(3) * x  =>  y/sqrt(3) <= x  =>  x >= y/sqrt(3)
    # Line AC: y = sqrt(3)x. If point is below/left, project.
    # Actually, let's use barycentric coordinates for robust projection.
    # Barycentric coords (u, v, w) for p relative to A, B, C.
    # A=(0,0), B=(1,0), C=(0.5, sqrt(3)/2)
    # p = u*A + v*B + w*C
    # x = 0.5*w + v
    # y = (sqrt(3)/2)*w
    # w = 2*y/sqrt(3)
    # v = x - 0.5*w
    # u = 1 - v - w
    
    w = 2 * p[1] / np.sqrt(3)
    v = p[0] - 0.5 * w
    u = 1 - v - w
    
    # Clip barycentric coords to [0, 1]
    u = np.clip(u, 0, 1)
    v = np.clip(v, 0, 1)
    w = np.clip(w, 0, 1)
    
    # Normalize if sum != 1 (due to clipping)
    s = u + v + w
    if s == 0:
        return A
    u /= s
    v /= s
    w /= s
    
    # Convert back to Cartesian
    x = 0.5 * w + v
    y = (np.sqrt(3) / 2) * w
    return np.array([x, y])

def objective_func(flat_points):
    """Objective function for optimizer. Minimizes negative min area."""
    points = flat_points.reshape(-1, 2)
    
    # Project points to triangle to handle constraints softly/hardly
    # Since we are using Nelder-Mead, it might go outside.
    # We can penalize or project. Projection is better for geometry.
    projected_points = np.array([project_to_triangle(p) for p in points])
    
    area = min_triangle_area(projected_points)
    
    # If any point is projected significantly, we might want a penalty?
    # No, projection ensures validity. The optimizer will move points to maximize area.
    # If a point is forced to the edge, the area might drop, which is the correct behavior.
    
    return -area

def generate_grid_start():
    """Generate a start based on a triangular grid."""
    # 10 points from a 4-level grid
    points = []
    level = 4
    for i in range(level):
        for j in range(level - i):
            # Barycentric: (1 - (i+j)/level, i/level, j/level)
            u = (level - i - j) / level
            v = i / level
            w = j / level
            x = 0.5 * w + v
            y = (np.sqrt(3) / 2) * w
            points.append([x, y])
    
    # We have 10 points. Add one more at the centroid of the whole triangle?
    # Or a random point?
    # Centroid is (0.5, sqrt(3)/6)
    points.append([0.5, np.sqrt(3) / 6])
    
    return np.array(points)

def generate_random_start():
    """Generate a random start."""
    points = []
    for _ in range(N):
        # Random point in triangle
        # Use barycentric coords
        r1 = random.random()
        r2 = random.random()
        if r1 + r2 > 1:
            r1 = 1 - r1
            r2 = 1 - r2
        u = r1
        v = r2
        w = 1 - r1 - r2
        x = 0.5 * w + v
        y = (np.sqrt(3) / 2) * w
        points.append([x, y])
    return np.array(points)

def generate_symmetric_start():
    """Generate a symmetric start."""
    points = []
    # Vertices
    points.append(A)
    points.append(B)
    points.append(C)
    # Midpoints
    points.append((A + B) / 2)
    points.append((B + C) / 2)
    points.append((C + A) / 2)
    # Centroid
    points.append((A + B + C) / 3)
    
    # Need 4 more.
    # Points at 1/3 and 2/3 along the medians?
    # Or a small triangle around the centroid?
    # Let's add points on the edges closer to vertices?
    # Or use a grid-like pattern for the remaining 4.
    # Let's just add random points for the rest to break symmetry slightly?
    # Or add points at (0.5, y) for various y?
    
    # Let's try adding 4 points forming a smaller triangle + center?
    # Scale the main triangle by 0.5 and rotate?
    # Simpler: Add 4 random points.
    for _ in range(4):
        r1 = random.random()
        r2 = random.random()
        if r1 + r2 > 1:
            r1 = 1 - r1
            r2 = 1 - r2
        u = r1
        v = r2
        w = 1 - r1 - r2
        x = 0.5 * w + v
        y = (np.sqrt(3) / 2) * w
        points.append([x, y])
        
    return np.array(points)

def construct():
    start_time = time.time()
    deadline = start_time + 115.0 # 5s safety margin
    
    best_points = None
    best_score = -1.0
    
    # List of initializers
    initializers = [
        generate_grid_start,
        generate_symmetric_start,
        generate_random_start,
        generate_random_start,
        generate_random_start,
        generate_random_start,
        generate_random_start,
        generate_random_start
    ]
    
    # Shuffle to vary order
    random.shuffle(initializers)
    
    for init_func in initializers:
        if time.time() > deadline:
            break
            
        try:
            points = init_func()
            # Ensure points are inside
            points = np.array([project_to_triangle(p) for p in points])
            
            # Flatten for optimizer
            flat_points = points.flatten()
            
            # Run Nelder-Mead
            # maxiter needs to be small enough to fit in budget
            # We have ~115s total. 8 restarts. ~14s per restart.
            # Nelder-Mead is fast.
            res = minimize(
                objective_func, 
                flat_points, 
                method='Nelder-Mead', 
                options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8}
            )
            
            # Extract result
            opt_points = res.x.reshape(-1, 2)
            opt_points = np.array([project_to_triangle(p) for p in opt_points])
            
            score = min_triangle_area(opt_points)
            
            if score > best_score:
                best_score = score
                best_points = opt_points
                
        except Exception:
            continue
            
    # If no valid points found, return a default
    if best_points is None:
        return generate_grid_start()
        
    return best_points

# EVOLVE-BLOCK-END
