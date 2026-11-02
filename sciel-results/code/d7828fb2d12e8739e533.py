import numpy as np
import time
import itertools
from scipy.optimize import minimize

N = 11
# Vertices of the equilateral triangle with side length 1
# (0,0), (1,0), (0.5, sqrt(3)/2)
H = np.sqrt(3) / 2
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, H])

# Deadline management
START_TIME = time.time()
TIME_LIMIT = 145.0  # Leave 5s safety margin

def is_inside_triangle(p):
    """Check if point p is inside the triangle."""
    # Barycentric coordinates or cross products
    # p = (x, y)
    x, y = p
    if y < -1e-9: return False
    if y > H + 1e-9: return False
    # Left edge: line from (0,0) to (0.5, H) -> y = sqrt(3) x -> x = y/sqrt(3)
    # Right edge: line from (1,0) to (0.5, H) -> y = -sqrt(3) (x-1) -> x = 1 - y/sqrt(3)
    if x < y / np.sqrt(3) - 1e-9: return False
    if x > 1.0 - y / np.sqrt(3) + 1e-9: return False
    return True

def clip_to_triangle(p):
    """Project point p to the nearest point inside the triangle."""
    x, y = p
    # Simple clipping might not work perfectly for all cases, 
    # but for optimization, we can use a penalty or just rely on bounds if we transform coords.
    # However, Nelder-Mead doesn't support bounds.
    # We will use a penalty approach in the objective function instead of hard clipping 
    # to maintain continuity, or we can transform to barycentric coords.
    
    # Let's stick to Cartesian and use a large penalty for points outside.
    # This is robust for Nelder-Mead.
    return np.array([x, y])

def compute_areas(points):
    """Compute areas of all triangles formed by the points."""
    # points is shape (N, 2)
    n = len(points)
    areas = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                # Area = 0.5 * | (x2-x1)(y3-y1) - (y2-y1)(x3-x1) |
                area = 0.5 * abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p2[1]-p1[1])*(p3[0]-p1[0]))
                areas.append(area)
    return np.array(areas)

def objective_flat(x):
    """Objective function for minimizer. Minimizes -min_area + penalty."""
    # Reshape
    points = x.reshape((N, 2))
    
    # Calculate min area
    areas = compute_areas(points)
    min_area = np.min(areas)
    
    # Penalty for points outside the triangle
    penalty = 0.0
    for p in points:
        x, y = p
        # Distance to outside
        # 1. y < 0
        if y < 0: penalty += (0 - y) ** 2
        # 2. y > H
        if y > H: penalty += (y - H) ** 2
        # 3. Left of left edge: x < y/sqrt(3)
        if x < y / np.sqrt(3):
            dist = y / np.sqrt(3) - x
            penalty += dist ** 2
        # 4. Right of right edge: x > 1 - y/sqrt(3)
        if x > 1.0 - y / np.sqrt(3):
            dist = x - (1.0 - y / np.sqrt(3))
            penalty += dist ** 2
            
    # We want to maximize min_area, so minimize -min_area
    # Penalty is added to minimize, so it pushes points inside
    return -min_area + 100.0 * penalty

def construct():
    global START_TIME
    START_TIME = time.time()
    
    best_x = None
    best_score = -np.inf
    best_points = None
    
    # Generate initial guesses
    inits = []
    
    # 1. Random points
    for _ in range(5):
        pts = np.random.rand(N, 2) * 1.0
        # Scale and shift to be roughly inside
        # A better random generation:
        pts = []
        for _ in range(N):
            u = np.random.rand()
            v = np.random.rand()
            if u + v > 1:
                u = 1 - u
                v = 1 - v
            # Map to triangle
            x = u + 0.5 * v
            y = H * v
            pts.append([x, y])
        inits.append(np.array(pts).flatten())
        
    # 2. Grid-like / Structured
    # Try to place points near vertices and center
    base_pts = [V0, V1, V2]
    # Add midpoints
    base_pts.append((V0+V1)/2)
    base_pts.append((V1+V2)/2)
    base_pts.append((V2+V0)/2)
    # Add center
    base_pts.append((V0+V1+V2)/3)
    # Add a few random ones to fill up to 11
    while len(base_pts) < N:
        u = np.random.rand()
        v = np.random.rand()
        if u + v > 1:
            u = 1 - u
            v = 1 - v
        x = u + 0.5 * v
        y = H * v
        base_pts.append(np.array([x, y]))
    inits.append(np.array(base_pts).flatten())
    
    # 3. Perturbed regular grid
    # Create a small grid
    grid_pts = []
    for i in range(3):
        for j in range(3):
            # Barycentric coordinates
            a = i / 2.0
            b = j / 2.0
            c = 1.0 - a - b
            if c < 0: continue
            p = a*V0 + b*V1 + c*V2
            grid_pts.append(p)
    # We have 6 points from a 3x3 grid clipped to triangle (corners and midpoints of edges/center?)
    # Actually (0,0), (0.5,0), (1,0), (0,0.5H), (0.5,0.5H), (1,0.5H)... 
    # Let's just take the 9 points from a 3x3 grid in barycentric space
    # and pick 11 best or just use first 11? 
    # Let's just add a perturbed version of the base_pts
    base_pts_pert = [p + np.random.normal(0, 0.05, 2) for p in base_pts]
    inits.append(np.array(base_pts_pert).flatten())

    # Optimization loop
    for init in inits:
        if time.time() - START_TIME > TIME_LIMIT:
            break
            
        # Run Nelder-Mead
        # maxiter is a safety net, but time is the real limit
        res = minimize(objective_flat, init, method='Nelder-Mead', 
                       options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6})
        
        # Check if it's valid (inside triangle)
        points = res.x.reshape((N, 2))
        valid = True
        for p in points:
            if not is_inside_triangle(p):
                # Allow small tolerance
                if not is_inside_triangle(p + np.random.normal(0, 1e-6, 2)):
                     # Try to project
                     pass
                # For now, let's just check if penalty is small
                # Re-evaluate objective to see penalty
                obj_val = objective_flat(res.x)
                # If obj_val is very negative (large penalty), it's bad
                # -min_area is at most -0.43 (max area of unit tri is 0.433)
                # So if obj_val > 0.1, it likely has penalty
                if obj_val > 0.0:
                    valid = False
                    break
        
        if valid:
            score = -res.fun
            if score > best_score:
                best_score = score
                best_x = res.x
                best_points = points

    if best_points is None:
        # Fallback to initial random
        return np.array(inits[0]).reshape((N, 2))
        
    return best_points

# Wrapper to ensure the function name and signature match
# The task says: construct() -> array of shape (n, 2)
# The current program has construct() defined.
# I will keep the name construct.

if __name__ == "__main__":
    pts = construct()
    areas = compute_areas(pts)
    print(f"Min Area: {np.min(areas)}")
    print(f"Score (raw): {np.min(areas)}")
