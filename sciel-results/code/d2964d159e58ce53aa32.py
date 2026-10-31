import numpy as np
from scipy.optimize import minimize
import time

N = 11

def min_triangle_area(points):
    """
    Calculate the minimum area of any triangle formed by 3 points.
    points: numpy array of shape (N, 2)
    """
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                # Area = 0.5 * |x1(y2 - y3) + x2(y3 - y1) + x3(y1 - y2)|
                area = 0.5 * abs(p1[0] * (p2[1] - p3[1]) + 
                                 p2[0] * (p3[1] - p1[1]) + 
                                 p3[0] * (p1[1] - p2[1]))
                if area < min_area:
                    min_area = area
    return min_area

def is_inside_triangle(p):
    """
    Check if point p (x,y) is inside the equilateral triangle
    with vertices (0,0), (1,0), (0.5, sqrt(3)/2).
    """
    x, y = p
    # Vertices: A(0,0), B(1,0), C(0.5, sqrt(3)/2)
    # Equations of lines:
    # AB: y = 0 => y >= 0
    # AC: y = sqrt(3) * x => y <= sqrt(3) * x  =>  sqrt(3)*x - y >= 0
    # BC: y = -sqrt(3) * (x - 1) => y <= -sqrt(3)*x + sqrt(3) => sqrt(3)*(1-x) - y >= 0
    
    if y < 0:
        return False
    if y > np.sqrt(3) * x:
        return False
    if y > np.sqrt(3) * (1 - x):
        return False
    if x < 0 or x > 1:
        return False
    return True

def objective(points_flat):
    """
    Objective function for scipy.optimize.minimize.
    Minimizes the negative of the minimum triangle area.
    Returns a large penalty if any point is outside the triangle.
    """
    points = points_flat.reshape(N, 2)
    
    # Check constraints
    for p in points:
        if not is_inside_triangle(p):
            # Return a large cost
            return 1000.0
            
    min_area = min_triangle_area(points)
    return -min_area

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s margin
    
    best_points = None
    best_score = -float('inf')
    
    # Initial configurations
    inits = []
    
    # 1. Random points
    rng = np.random.RandomState(42)
    for _ in range(5):
        pts = []
        while len(pts) < N:
            u, v = rng.random(2)
            if u + v < 1:
                # Map (u,v) in simplex to triangle
                # Standard mapping: x = u + 0.5*v, y = sqrt(3)/2 * v
                x = u + 0.5 * v
                y = (np.sqrt(3) / 2) * v
                # Ensure strictly inside to avoid boundary issues initially
                if 0.01 < x < 0.99 and 0.01 < y < 0.86:
                    pts.append([x, y])
        inits.append(np.array(pts).flatten())
        
    # 2. Grid-like / Hexagonal pattern
    # Try to place points in a structured way
    # A 3x3 grid plus 2 corners? Or just a perturbed grid.
    # Let's try a simple grid in barycentric coordinates
    # Barycentric coords (a,b,c) sum to 1.
    # Points:
    # (1/3, 1/3, 1/3) - center
    # (2/3, 1/6, 1/6), (1/6, 2/3, 1/6), (1/6, 1/6, 2/3) - midpoints of medians?
    # Let's just use a few symmetric patterns
    
    # Pattern 1: 1 center, 3 near midpoints of sides, 3 near vertices, 4 more?
    # Simpler: Use a grid of barycentric coordinates
    # For N=11, maybe a 3x3 grid of barycentric points (9 points) + 2 extra?
    # Barycentric grid points:
    # i, j from 0 to 2. k = 3 - i - j.
    # Scale: (i/2, j/2, k/2)
    bary_pts = []
    for i in range(3):
        for j in range(3):
            k = 2 - i - j
            if k >= 0:
                bary_pts.append([i/2.0, j/2.0, k/2.0])
    # This gives 6 points: (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5), (0.5,0.25,0.25)... wait.
    # Let's generate a finer grid.
    # Let's just add a few random starts and one structured start.
    
    # Structured start: 
    # Place points at specific barycentric coordinates
    # Center: (1/3, 1/3, 1/3)
    # 6 points around center, 4 more?
    # Let's stick to random restarts as they are robust.
    # Add 10 more random inits
    for _ in range(10):
        pts = []
        while len(pts) < N:
            u, v = rng.random(2)
            if u + v < 1:
                x = u + 0.5 * v
                y = (np.sqrt(3) / 2) * v
                if 0.01 < x < 0.99 and 0.01 < y < 0.86:
                    pts.append([x, y])
        inits.append(np.array(pts).flatten())

    for init in inits:
        if time.time() > deadline:
            break
            
        try:
            # Use Nelder-Mead
            res = minimize(objective, init, method='Nelder-Mead', 
                           options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-6})
            
            if res.success or res.fun < best_score:
                # Verify the result
                final_pts = res.x.reshape(N, 2)
                # Check if all inside
                valid = True
                for p in final_pts:
                    if not is_inside_triangle(p):
                        valid = False
                        break
                
                if valid:
                    score = min_triangle_area(final_pts)
                    if score > best_score:
                        best_score = score
                        best_points = final_pts
        except Exception:
            pass

    # If no valid points found (unlikely), return a fallback
    if best_points is None:
        # Return a simple valid configuration
        # Just use the first random init that was valid
        for init in inits:
            pts = init.reshape(N, 2)
            valid = True
            for p in pts:
                if not is_inside_triangle(p):
                    valid = False
                    break
            if valid:
                best_points = pts
                break
        
        if best_points is None:
            # Hardcode a valid set
            best_points = np.array([
                [0.5, 0.8], [0.2, 0.2], [0.8, 0.2], 
                [0.5, 0.4], [0.3, 0.5], [0.7, 0.5],
                [0.5, 0.6], [0.4, 0.3], [0.6, 0.3],
                [0.1, 0.1], [0.9, 0.1]
            ])
            # Ensure they are actually inside
            # (0.5, 0.8) -> y=0.8, sqrt(3)*0.5=0.866, sqrt(3)*(1-0.5)=0.866. OK.
            # (0.2, 0.2) -> y=0.2, sqrt(3)*0.2=0.346. OK.
            # (0.8, 0.2) -> y=0.2, sqrt(3)*(1-0.8)=0.346. OK.
            # (0.5, 0.4) -> OK.
            # (0.3, 0.5) -> y=0.5, sqrt(3)*0.3=0.519. OK.
            # (0.7, 0.5) -> y=0.5, sqrt(3)*(0.3)=0.519. OK.
            # (0.5, 0.6) -> y=0.6, sqrt(3)*0.5=0.866. OK.
            # (0.4, 0.3) -> y=0.3, sqrt(3)*0.4=0.69. OK.
            # (0.6, 0.3) -> y=0.3, sqrt(3)*0.4=0.69. OK.
            # (0.1, 0.1) -> y=0.1, sqrt(3)*0.1=0.17. OK.
            # (0.9, 0.1) -> y=0.1, sqrt(3)*0.1=0.17. OK.
            
    return best_points

# EVOLVE-BLOCK-START
# The code above is the new program. 
# I will wrap it in the required format.
# EVOLVE-BLOCK-END
