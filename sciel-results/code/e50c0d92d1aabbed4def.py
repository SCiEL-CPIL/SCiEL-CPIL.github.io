import numpy as np
from scipy.optimize import minimize
from itertools import combinations
import time

N = 11
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3)/2.0])

def check_inside(p):
    # Barycentric coordinates
    # p = u*V0 + v*V1 + w*V2
    # u+v+w = 1
    # Solve linear system
    # Or use cross products
    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    
    d1 = cross(p, V0, V1)
    d2 = cross(p, V1, V2)
    d3 = cross(p, V2, V0)
    
    # All must have same sign (positive if counter-clockwise)
    # V0, V1, V2 is CCW?
    # V0(0,0), V1(1,0), V2(0.5, 0.866)
    # Cross(V0, V1, V2) = (1-0)*(0.866-0) - (0-0)*(0.5-0) = 0.866 > 0.
    # So d1, d2, d3 should be >= 0.
    return d1 >= -1e-9 and d2 >= -1e-9 and d3 >= -1e-9

def min_area(points):
    min_a = 1e9
    # points is (N, 2)
    for i in range(N):
        for j in range(i+1, N):
            for k in range(j+1, N):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                area = 0.5 * abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1]))
                if area < min_a:
                    min_a = area
    return min_a

def objective(x):
    points = x.reshape(N, 2)
    # Penalty for being outside
    for p in points:
        if not check_inside(p):
            return 1e5
    return -min_area(points)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    best_val = -1e9
    best_points = None
    
    # Generate initial configurations
    inits = []
    
    # 1. Grid-based initialization
    # Create a grid
    grid_points = []
    steps = 15
    for i in range(steps):
        for j in range(steps):
            # Barycentric grid
            u = i / (steps - 1)
            v = j / (steps - 1)
            if u + v <= 1.0:
                # p = (1-u-v)*V0 + u*V1 + v*V2
                # V0 is (0,0)
                p = u * V1 + v * V2
                grid_points.append(p)
    
    # Select 11 points from grid to maximize min distance (greedy)
    # This gives a good spread
    if len(grid_points) >= N:
        selected = []
        remaining = grid_points.copy()
        # Start with a corner
        selected.append(remaining.pop(0))
        while len(selected) < N and remaining:
            best_idx = -1
            best_min_dist = -1
            for idx, p in enumerate(remaining):
                min_d = min(np.linalg.norm(p - s) for s in selected)
                if min_d > best_min_dist:
                    best_min_dist = min_d
                    best_idx = idx
            if best_idx != -1:
                selected.append(remaining.pop(best_idx))
        if len(selected) == N:
            inits.append(np.array(selected).flatten())

    # 2. Symmetric initialization
    # Vertices, midpoints, centroid
    sym_points = [
        V0, V1, V2,
        (V0+V1)/2, (V1+V2)/2, (V2+V0)/2,
        (V0+V1+V2)/3
    ]
    # Need 4 more. Add points at 1/3 and 2/3 along edges? Or interior hexagon?
    # Let's just add 4 random interior points to this symmetric core for variety
    rng = np.random.RandomState(42)
    for _ in range(4):
        u, v = rng.random(2)
        if u + v > 1:
            u, v = 1-u, 1-v
        p = u * V1 + v * V2
        sym_points.append(p)
    inits.append(np.array(sym_points).flatten())

    # 3. Random initializations
    for i in range(5):
        rng = np.random.RandomState(100 + i)
        pts = []
        while len(pts) < N:
            u, v = rng.random(2)
            if u + v < 1.0:
                p = u * V1 + v * V2
                pts.append(p)
        inits.append(np.array(pts).flatten())

    # Run optimization
    for init in inits:
        if time.time() > deadline:
            break
        
        try:
            res = minimize(objective, init, method='Nelder-Mead', options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-6})
            if time.time() > deadline:
                break
            
            val = -res.fun
            if val > best_val:
                # Check validity
                points = res.x.reshape(N, 2)
                valid = True
                for p in points:
                    if not check_inside(p):
                        valid = False
                        break
                if valid:
                    best_val = val
                    best_points = points
        except Exception:
            continue

    if best_points is None:
        # Fallback to grid
        best_points = np.array(inits[0]).reshape(N, 2)
        
    return best_points
