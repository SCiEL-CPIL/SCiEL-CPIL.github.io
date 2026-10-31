import numpy as np
from scipy.optimize import minimize
import time

N = 11

# Vertices of the equilateral triangle
# Side length 1
V = np.array([
    [0.0, 0.0],
    [1.0, 0.0],
    [0.5, np.sqrt(3) / 2]
])

def point_in_triangle(p, v1, v2, v3):
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
    
    d1 = sign(p, v1, v2)
    d2 = sign(p, v2, v3)
    d3 = sign(p, v3, v1)
    
    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    
    return not (has_neg and has_pos)

def min_triangle_area(points):
    n = len(points)
    min_area = float('inf')
    # Calculate area of all triangles formed by 3 points
    # Area = 0.5 * | (x2-x1)(y3-y1) - (x3-x1)(y2-y1) |
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
                if area < min_area:
                    min_area = area
    return min_area

def objective(x):
    points = x.reshape(-1, 2)
    # Check if all points are inside the triangle
    # If any point is outside, return a large penalty (negative area)
    for p in points:
        if not point_in_triangle(p, V[0], V[1], V[2]):
            return 1.0 # We are minimizing -area, so large positive is bad? 
            # Wait, objective returns value to minimize. 
            # We want to maximize min_area. 
            # So objective = -min_area.
            # If invalid, we want a "bad" score. 
            # min_area would be 0 or undefined. 
            # Let's just return a large positive number so it's minimized away.
            # Actually, if points are outside, the "triangle formed by three points" might still be calculated, 
            # but the constraint is "inside". 
            # Let's enforce constraints by projecting or penalizing.
            # Penalty method: add penalty if outside.
            pass
    
    # Calculate min area
    min_area = min_triangle_area(points)
    return -min_area

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s safety margin
    
    best_points = None
    best_score = -1.0 # We want to maximize min_area, so minimize -min_area. 
                      # Actually, let's track the max min_area directly for clarity.
    best_min_area = 0.0
    
    def try_optimize(initial_points):
        nonlocal best_points, best_min_area
        
        x0 = initial_points.flatten()
        
        # Nelder-Mead is robust for small non-smooth problems (min of areas is non-smooth)
        # Powell is also good.
        try:
            res = minimize(objective, x0, method='Nelder-Mead', 
                           options={'xatol': 1e-6, 'fatol': 1e-8, 'maxiter': 2000, 'maxfev': 2000})
            final_points = res.x.reshape(-1, 2)
            
            # Verify constraints
            valid = True
            for p in final_points:
                if not point_in_triangle(p, V[0], V[1], V[2]):
                    valid = False
                    break
            
            if valid:
                current_min_area = min_triangle_area(final_points)
                if current_min_area > best_min_area:
                    best_min_area = current_min_area
                    best_points = final_points
        except Exception:
            pass

    # 1. Symmetric Initial Guesses
    
    # Guess 1: Center + 2 rings?
    # 11 points. 
    # Maybe a triangular lattice subset?
    # Side 3 lattice has 6 points (1,2,3). Side 4 has 10 points (1,2,3,4). 
    # 11 points is one more than the 10-point triangular grid (side 4).
    # Let's take the 10 points of a side-4 triangular grid and add a center or adjust.
    
    # Generate side-4 triangular grid points
    # Rows: 1, 2, 3, 4 points. Total 10.
    # Coordinates in barycentric or Cartesian.
    # Let's generate points on a grid.
    grid_points = []
    for i in range(4):
        for j in range(4 - i):
            # Barycentric coords (i, j, 4-1-i-j) ? 
            # Let's use Cartesian mapping from a standard triangle.
            # Row i (0 to 3), Col j (0 to 3-i)
            # x = (j + i/2) / 3.5 ? No.
            # Let's map (0..3, 0..3-i) to the triangle.
            # y = i * h / 3
            # x = j * w / 3 + i * w / 6 ?
            # Simpler: Use affine transformation from a reference triangle.
            # Ref: (0,0), (1,0), (0,1) -> Area 0.5
            # Our Tri: (0,0), (1,0), (0.5, sqrt(3)/2)
            
            # Let's just use a simple symmetric pattern manually.
            # Center (0.5, sqrt(3)/6)
            # 6 points around center at radius r1
            # 4 points at corners? No, corners are vertices.
            
            pass

    # Let's use a more systematic random + symmetric approach.
    
    # Symmetric Pattern 1: 
    # 1 center
    # 6 points forming a hexagon
    # 4 points forming a square? No, triangle symmetry.
    # 1 center, 3 on mid-edges, 6 on inner ring?
    
    # Let's try a "Best Packing" heuristic.
    # Start with 1 point at centroid.
    # Add points one by one at the location that maximizes the min distance to existing points? 
    # This is greedy and doesn't maximize triangle area, but gives a good start.
    
    def greedy_init():
        pts = [np.array([0.5, np.sqrt(3)/6])] # Centroid
        for _ in range(N - 1):
            best_loc = None
            best_min_dist = 0
            # Sample grid
            for u in np.linspace(0.05, 0.95, 20):
                for v in np.linspace(0.05, 0.95, 20):
                    if u + v < 1.0:
                        p = u * V[1] + v * V[2] # Barycentric? 
                        # V0=0, V1=(1,0), V2=(0.5, h)
                        # p = u*V1 + v*V2 + (1-u-v)*V0
                        p = np.array([u*1.0 + v*0.5, v*np.sqrt(3)/2])
                        if point_in_triangle(p, V[0], V[1], V[2]):
                            min_d = min(np.linalg.norm(p - q) for q in pts)
                            if min_d > best_min_dist:
                                best_min_dist = min_d
                                best_loc = p
            if best_loc is not None:
                pts.append(best_loc)
        return np.array(pts)

    # Symmetric Pattern 2: Triangular Grid (10 pts) + 1 center adjustment
    # The 10 points of a side-4 grid are:
    # (0,0), (1/3,0), (2/3,0), (1,0)
    # (1/6, h/3), (1/2, h/3), (5/6, h/3)
    # (1/3, 2h/3), (2/3, 2h/3)
    # (0.5, h)
    # This is 10 points. Adding a 11th at the centroid (0.5, h/3) is already there? 
    # Centroid is (0.5, h/3). Yes, it's in the second row.
    # So the 10-point grid includes the centroid.
    # Where to put the 11th? Maybe split one of the larger triangles.
    
    grid10 = np.array([
        [0.0, 0.0],
        [1/3, 0.0],
        [2/3, 0.0],
        [1.0, 0.0],
        [1/6, np.sqrt(3)/6],
        [0.5, np.sqrt(3)/6],
        [5/6, np.sqrt(3)/6],
        [1/3, np.sqrt(3)/3],
        [2/3, np.sqrt(3)/3],
        [0.5, np.sqrt(3)/2]
    ])
    # Add a point slightly offset from center or in a gap.
    # The gaps are the small triangles.
    # Let's just add the centroid again? No, duplicates not allowed effectively.
    # Let's add a point at (0.5, np.sqrt(3)/6 + 0.1)?
    
    # Let's rely on the optimizer to fix the grid.
    init1 = grid10.copy()
    # Add an 11th point. Let's put it at the centroid of the top triangle?
    # Top triangle vertices: (1/3, 2h/3), (2/3, 2h/3), (0.5, h)
    # Centroid: (0.5, (2h/3 + 2h/3 + h)/3) = (0.5, 7h/9)
    init1 = np.vstack([init1, [0.5, 7*np.sqrt(3)/18]])
    
    # Random Inits
    np.random.seed(42)
    
    # Run greedy
    if time.time() < deadline:
        try_optimize(greedy_init())
        
    # Run Grid
    if time.time() < deadline:
        try_optimize(init1)
        
    # Run Randoms
    num_randoms = 20
    for i in range(num_randoms):
        if time.time() > deadline - 5:
            break
            
        # Generate random points in triangle
        pts = []
        while len(pts) < N:
            u, v = np.random.random(2)
            if u + v < 1.0:
                p = np.array([u + 0.5*v, np.sqrt(3)/2 * v])
                pts.append(p)
        pts = np.array(pts)
        try_optimize(pts)
        
    if best_points is None:
        # Fallback: just return grid10 + one random
        pts = list(grid10)
        pts.append([0.5, np.sqrt(3)/6])
        best_points = np.array(pts)
        
    return best_points
