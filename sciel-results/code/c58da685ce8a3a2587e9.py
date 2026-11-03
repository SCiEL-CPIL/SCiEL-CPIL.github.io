import numpy as np
import time
import itertools
from scipy.optimize import minimize

N = 11
# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3)/2.0])

def min_triangle_area(points):
    """
    Calculate the minimum area of any triangle formed by 3 points.
    points: np.array of shape (N, 2)
    """
    min_area = float('inf')
    n = len(points)
    # Precompute squared distances? No, area needs cross product.
    # Iterate through all combinations of 3 points
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                # Area = 0.5 * | (p2-p1) x (p3-p1) |
                v1 = p2 - p1
                v2 = p3 - p1
                cross = v1[0]*v2[1] - v1[1]*v2[0]
                area = 0.5 * abs(cross)
                if area < min_area:
                    min_area = area
    return min_area

def project_to_triangle(points):
    """
    Project points into the triangle defined by V0, V1, V2.
    Using barycentric coordinates or simple clipping.
    For an equilateral triangle, we can use half-plane inequalities.
    """
    points = np.array(points)
    # Inequalities:
    # 1. y >= 0
    # 2. y <= sqrt(3) * x  (Line V0-V2) -> y - sqrt(3)x <= 0
    # 3. y <= -sqrt(3) * (x - 1) (Line V1-V2) -> y + sqrt(3)x - sqrt(3) <= 0
    
    for i in range(len(points)):
        x, y = points[i]
        
        # Clip y >= 0
        if y < 0:
            y = 0.0
            
        # Clip y <= sqrt(3) * x
        # If y > sqrt(3) * x, we are outside on the left.
        # Intersection with y=0 is x=0. Intersection with top edge?
        # Simplest: If outside, move to closest point on edge.
        # But for optimization, we just need to keep it inside.
        # A robust way is to use barycentric coordinates and clamp.
        
        # Let's use barycentric coordinates (u, v, w) where u+v+w=1, u,v,w>=0
        # P = u*V0 + v*V1 + w*V2
        # V0=(0,0), V1=(1,0), V2=(0.5, h)
        # x = v*1 + w*0.5
        # y = w*h
        # w = y/h
        # v = x - 0.5*w
        # u = 1 - v - w
        
        h = np.sqrt(3)/2.0
        w = y / h
        v = x - 0.5 * w
        u = 1.0 - v - w
        
        # Clamp u, v, w to [0, 1] and renormalize?
        # Actually, if a coordinate is negative, the point is outside.
        # We can set the negative coordinate to 0 and renormalize the others to sum to 1.
        
        orig_u, orig_v, orig_w = u, v, w
        if u < 0 or v < 0 or w < 0:
            # Simple clipping: set negative to 0, scale others
            u = max(0, u)
            v = max(0, v)
            w = max(0, w)
            s = u + v + w
            if s == 0:
                # Point was at a vertex or outside in a way that made all 0? 
                # Fallback to center
                u, v, w = 1/3, 1/3, 1/3
            else:
                u /= s
                v /= s
                w /= s
                
        # Reconstruct x, y
        x_new = v * 1.0 + w * 0.5
        y_new = w * h
        points[i] = [x_new, y_new]
        
    return points

def objective(x):
    """
    Objective function for minimization.
    We want to MAXIMIZE min_area, so we MINIMIZE -min_area.
    x: flattened array of shape (2*N,)
    """
    points = x.reshape(-1, 2)
    points = project_to_triangle(points)
    area = min_triangle_area(points)
    return -area

def construct():
    start_time = time.time()
    deadline = start_time + 120.0 # Safety margin
    
    best_points = None
    best_score = -float('inf')
    
    # Generate initial configurations
    inits = []
    
    # 1. Lattice-like: 3x3 grid mapped to triangle
    # Create a 3x3 grid in barycentric coords
    # u, v, w >= 0, u+v+w=1
    # Grid points: i/2, j/2, k/2 where i+j+k=2? No, that's 6 points.
    # Let's try to place 11 points roughly evenly.
    
    # A common good start is a "hexagonal" packing or a grid.
    # Let's create a 3x3 grid in a square and map it? 
    # Better: Generate points in barycentric coords.
    
    # Init 1: Regular 3x3 grid in barycentric space (10 points) + center?
    # Barycentric grid for n=3 (4 levels):
    # (0,0,1), (0,1,1), (0,2,1)... sum=3?
    # Let's use a simple random start with some structure.
    
    # Start 1: Random
    np.random.seed(42)
    for _ in range(5):
        # Generate random barycentric coords
        u = np.random.dirichlet([1, 1, 1], size=N)
        pts = np.zeros((N, 2))
        for i in range(N):
            # P = v*V1 + w*V2
            pts[i] = u[i, 1] * V1 + u[i, 2] * V2
        inits.append(pts.flatten())
        
    # Start 2: Grid-like
    # Try to place points on a 3x3 grid inside the triangle
    # We can sample a 3x3 grid in the bounding box and keep those inside?
    # Or generate a specific pattern.
    
    # Let's try a "centered" lattice.
    # 1 center, 6 around, 4 more?
    # Let's just use several random starts and rely on Nelder-Mead.
    # To be safe, add a few more diverse random starts.
    for seed in [1, 2, 3, 4, 5]:
        np.random.seed(seed)
        u = np.random.dirichlet([1, 1, 1], size=N)
        pts = np.zeros((N, 2))
        for i in range(N):
            pts[i] = u[i, 1] * V1 + u[i, 2] * V2
        inits.append(pts.flatten())
        
    # Add a very regular start: 
    # 11 points. 
    # Maybe a 3x3 grid (9 points) + 2 extra?
    # Let's construct a 3x3 grid in barycentric coordinates.
    # Levels 0, 1, 2.
    # Points (i, j, k) with i+j+k=2. There are 6 such points.
    # Points with i+j+k=3. There are 10 such points.
    # We need 11.
    # Let's take the 10 points of the 3rd level (boundary of the 3x3 grid) and the center (1,1,1)?
    # Center is (1/3, 1/3, 1/3).
    # Boundary points of level 3:
    # (3,0,0), (0,3,0), (0,0,3) -> Vertices
    # (2,1,0), (1,2,0), (0,2,1), (1,0,2), (2,0,1), (0,1,2) -> Midpoints of edges? No, these are on the edges.
    # Actually, for n=3, the lattice points are:
    # (3,0,0), (2,1,0), (1,2,0), (0,3,0)
    # (2,0,1), (1,1,1), (0,2,1)
    # (1,0,2), (0,1,2)
    # (0,0,3)
    # Total 10 points.
    # We need 11. Add the center of the triangle? (1,1,1) is already there.
    # Add a point slightly perturbed?
    # Let's just use the 10 points and add one random point near the center.
    
    pts_grid = []
    for i in range(4):
        for j in range(4-i):
            k = 3 - i - j
            # Barycentric (i/3, j/3, k/3)
            u, v, w = i/3.0, j/3.0, k/3.0
            x = v * V1[0] + w * V2[0]
            y = v * V1[1] + w * V2[1]
            pts_grid.append([x, y])
    # This gives 10 points.
    # Add 11th point at center (1/3, 1/3, 1/3) -> (0.5, sqrt(3)/6)
    # Wait, (1/3, 1/3, 1/3) corresponds to i=j=k=1, which is in the list (i=1, j=1, k=1).
    # So we have 10 points.
    # Let's add a point at the centroid of one of the small triangles?
    # Or just add a random point.
    pts_grid.append([0.5, np.sqrt(3)/6.0]) # This is a duplicate of the center point from the grid?
    # Grid point for i=1, j=1, k=1: x = 1/3*1 + 1/3*0.5 = 0.5, y = 1/3*sqrt(3)/2 = sqrt(3)/6.
    # Yes, it's a duplicate.
    # Let's shift one point slightly to make it unique?
    # Or just use 9 points from grid and 2 random?
    # Let's just use the 10 grid points and 1 random point.
    np.random.seed(100)
    u_rand = np.random.dirichlet([1, 1, 1])
    x_rand = u_rand[1] * V1[0] + u_rand[2] * V2[0]
    y_rand = u_rand[1] * V1[1] + u_rand[2] * V2[1]
    pts_grid.append([x_rand, y_rand])
    
    inits.append(np.array(pts_grid).flatten())

    # Optimization loop
    for i, init in enumerate(inits):
        if time.time() > deadline - 10:
            break
            
        try:
            # Nelder-Mead
            res = minimize(objective, init, method='Nelder-Mead', 
                           options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8, 'disp': False})
            
            if res.success or res.fun < best_score:
                # res.fun is -min_area
                # We want to maximize min_area, so minimize -min_area
                # If res.fun is smaller (more negative), it's better.
                # Wait, best_score is initialized to -inf.
                # We compare res.fun (negative area) with best_score.
                # We want the most negative value (largest area).
                if res.fun < best_score:
                    best_score = res.fun
                    best_points = res.x
        except Exception:
            continue

    if best_points is None:
        # Fallback to a valid random configuration
        np.random.seed(7)
        u = np.random.dirichlet([1, 1, 1], size=N)
        pts = np.zeros((N, 2))
        for i in range(N):
            pts[i] = u[i, 1] * V1 + u[i, 2] * V2
        return np.array(pts)
        
    return np.array(best_points).reshape(-1, 2)
