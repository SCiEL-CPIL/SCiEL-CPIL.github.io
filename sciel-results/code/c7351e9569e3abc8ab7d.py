import numpy as np
import time
import itertools
from scipy.optimize import minimize

N = 11
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
TRI_AREA = 0.5 * np.sqrt(3) / 2

def point_in_triangle(p, eps=1e-9):
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
    d1 = sign(p, V0, V1)
    d2 = sign(p, V1, V2)
    d3 = sign(p, V2, V0)
    has_neg = (d1 < -eps) or (d2 < -eps) or (d3 < -eps)
    has_pos = (d1 > eps) or (d2 > eps) or (d3 > eps)
    return not (has_neg and has_pos)

def min_triangle_area(points):
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                # Area = 0.5 * |det(Pj-Pi, Pk-Pi)|
                v1 = points[j] - points[i]
                v2 = points[k] - points[i]
                area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                if area < min_area:
                    min_area = area
    return min_area

def penalty(points):
    # Penalty for points outside the triangle
    # We use a smooth penalty based on barycentric coordinates
    # Barycentric coords: u, v, w >= 0
    # P = u*V0 + v*V1 + w*V2
    # V0=(0,0), V1=(1,0), V2=(0.5, sqrt(3)/2)
    # x = v + 0.5w
    # y = sqrt(3)/2 w
    # w = 2y/sqrt(3)
    # v = x - 0.5w
    # u = 1 - v - w
    pen = 0.0
    for p in points:
        x, y = p
        w = 2 * y / np.sqrt(3)
        v = x - 0.5 * w
        u = 1 - v - w
        if u < 0: pen += u*u
        if v < 0: pen += v*v
        if w < 0: pen += w*w
    return pen

def objective(params):
    points = params.reshape(N, 2)
    min_area = min_triangle_area(points)
    pen = penalty(points)
    # We want to maximize min_area, so minimize -min_area
    # Add penalty to keep points inside
    return -min_area + 100.0 * pen

def generate_hex_init():
    # Generate a hexagonal grid and pick 11 points
    # This is a heuristic initialization
    points = []
    # Center
    points.append(np.array([0.5, np.sqrt(3)/6]))
    # Ring 1 (6 points)
    for i in range(6):
        angle = i * np.pi / 3
        r = 0.2
        x = 0.5 + r * np.cos(angle)
        y = np.sqrt(3)/6 + r * np.sin(angle)
        points.append(np.array([x, y]))
    # Ring 2 (4 points, to make 11 total? 1+6=7, need 4 more)
    # Let's just add 4 more random-ish or grid points
    # Actually, let's just return the 7 and pad with random?
    # Better: Generate a denser grid and select.
    # For now, let's just use random for the rest in the caller.
    return np.array(points[:N])

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    best_score = -float('inf')
    best_points = None
    
    # Initializations
    inits = []
    
    # 1. Random initializations
    rng = np.random.default_rng(42)
    for _ in range(50):
        # Random points in triangle using barycentric coordinates
        u = rng.random(N)
        v = rng.random(N)
        mask = u + v > 1
        u[mask] = 1 - v[mask]
        v[mask] = 1 - u[mask]
        # u, v are barycentric for V0, V1? 
        # P = u*V0 + v*V1 + (1-u-v)*V2
        # Wait, standard barycentric: P = b0*V0 + b1*V1 + b2*V2
        # Let's use:
        # r1, r2 random. If r1+r2 > 1, reflect.
        # b0 = 1 - r1 - r2
        # b1 = r1
        # b2 = r2
        # This works.
        r1 = rng.random(N)
        r2 = rng.random(N)
        mask = r1 + r2 > 1
        r1[mask] = 1 - r2[mask]
        r2[mask] = 1 - r1[mask]
        b0 = 1 - r1 - r2
        b1 = r1
        b2 = r2
        
        pts = np.zeros((N, 2))
        for i in range(N):
            pts[i] = b0[i]*V0 + b1[i]*V1 + b2[i]*V2
        inits.append(pts.flatten())

    # 2. Hexagonal-ish initialization
    # Try to place points in a symmetric pattern
    # 1 center, 6 inner, 4 outer?
    # Let's try a grid
    grid_pts = []
    for i in range(5):
        for j in range(5):
            # Map grid to triangle
            # This is tricky. Let's just use random for now, but maybe structured random.
            pass
            
    # Let's add a few structured inits
    # Symmetric: 1 center, 6 around, 4 at midpoints of edges?
    center = np.array([0.5, np.sqrt(3)/6])
    inits.append(center) # This is just 1 point, need 11.
    # Let's construct a full 11-point symmetric guess
    sym_pts = []
    sym_pts.append(center)
    # 6 points in hexagon
    for i in range(6):
        angle = i * np.pi / 3
        r = 0.25
        x = 0.5 + r * np.cos(angle)
        y = np.sqrt(3)/6 + r * np.sin(angle)
        sym_pts.append(np.array([x, y]))
    # 4 points at edge midpoints?
    # Edge midpoints: (0.5, 0), (0.75, sqrt(3)/4), (0.25, sqrt(3)/4)
    # That's 3. Need 1 more. Center of one face?
    # Let's just add 4 random points near the boundary
    for i in range(4):
        angle = i * np.pi / 2 + np.pi/4
        r = 0.4
        x = 0.5 + r * np.cos(angle)
        y = np.sqrt(3)/6 + r * np.sin(angle)
        # Check if inside, if not, project
        sym_pts.append(np.array([x, y]))
    inits.append(np.array(sym_pts).flatten())
    
    # Run optimization for each init
    for init in inits:
        if time.time() > deadline:
            break
            
        # Reshape for minimize
        x0 = init
        # Nelder-Mead
        try:
            res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6})
            if res.success or res.fun < objective(x0):
                pts = res.x.reshape(N, 2)
                # Check validity
                valid = all(point_in_triangle(p) for p in pts)
                if valid:
                    score = min_triangle_area(pts)
                    if score > best_score:
                        best_score = score
                        best_points = pts
        except Exception:
            pass

    # If no valid points found, return random
    if best_points is None:
        return generate_hex_init()
        
    return best_points

# EVOLVE-BLOCK-END
