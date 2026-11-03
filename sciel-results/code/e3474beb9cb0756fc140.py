"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time
import itertools

V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
TRI_AREA = 0.5 * np.sqrt(3) / 2

def triangle_area(p1, p2, p3):
    return 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))

def dist_to_triangle(p):
    # Distance from point p to the triangle (V0, V1, V2)
    # If inside, distance is 0.
    # If outside, distance to the closest edge.
    def point_to_line_dist(p, a, b):
        ab = b - a
        ap = p - a
        ab_sq = np.dot(ab, ab)
        if ab_sq == 0:
            return np.linalg.norm(ap)
        t = np.dot(ap, ab) / ab_sq
        t = np.clip(t, 0, 1)
        proj = a + t * ab
        return np.linalg.norm(p - proj)
    
    # Check if inside
    # Barycentric coordinates or sign test
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
    
    d1 = sign(p, V0, V1)
    d2 = sign(p, V1, V2)
    d3 = sign(p, V2, V0)
    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    
    if not (has_neg and has_pos):
        return 0.0
    
    # Outside, find min distance to edges
    d01 = point_to_line_dist(p, V0, V1)
    d12 = point_to_line_dist(p, V1, V2)
    d20 = point_to_line_dist(p, V2, V0)
    return min(d01, d12, d20)

def smooth_max_0(x, eps=1e-6):
    # Smooth approximation of max(0, x)
    return 0.5 * (x + np.sqrt(x*x + eps))

def objective_smooth(params, lambda_val=10.0, mu=1000.0):
    points = params.reshape(-1, 2)
    
    # Penalty for being outside
    pen = 0.0
    for p in points:
        d = dist_to_triangle(p)
        if d > 0:
            pen += smooth_max_0(d, eps=1e-8) ** 2
    
    if pen > 1e-4:
        return mu * pen + 1e6 # Large penalty
    
    # Sum of exp(-lambda * area)
    total = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            for k in range(j + 1, N):
                a = triangle_area(points[i], points[j], points[k])
                total += np.exp(-lambda_val * a)
                
    return total

def min_triangle_area_raw(points):
    min_a = float('inf')
    for i in range(N):
        for j in range(i + 1, N):
            for k in range(j + 1, N):
                a = triangle_area(points[i], points[j], points[k])
                if a < min_a:
                    min_a = a
    return min_a

def random_points_inside(rng):
    points = []
    for _ in range(N):
        while True:
            u = rng.random()
            v = rng.random()
            if u + v <= 1:
                p = u * V1 + v * V2
                points.append(p)
                break
    return np.array(points)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Leave 10s margin
    
    best_points = None
    best_score = 0.0
    
    # Try a few different lambda values
    lambdas = [5.0, 10.0, 20.0, 50.0]
    
    import random
    rng = random.Random(42)
    
    restarts = 0
    for lam in lambdas:
        for _ in range(5): # 5 restarts per lambda
            if time.time() > deadline:
                break
            
            # Initial guess
            x0 = random_points_inside(rng).flatten()
            
            # Optimization
            try:
                res = minimize(
                    objective_smooth,
                    x0,
                    args=(lam, 1000.0),
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                points = res.x.reshape(-1, 2)
                
                # Check validity
                valid = True
                for p in points:
                    if dist_to_triangle(p) > 1e-6:
                        valid = False
                        break
                
                if valid:
                    score = min_triangle_area_raw(points)
                    if score > best_score:
                        best_score = score
                        best_points = points
            except Exception as e:
                pass
            
            restarts += 1
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
            
    if best_points is None:
        # Fallback to random
        best_points = random_points_inside(rng)
        
    return best_points
# EVOLVE-BLOCK-END
