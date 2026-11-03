import numpy as np
from scipy.optimize import minimize
import itertools
import time

N = 11

def point_in_triangle(p, a, b, c):
    # p: (2,) or (N, 2)
    # a, b, c: (2,)
    # Using barycentric coordinates
    # To support vectorization, we assume p is (N, 2) or (2,)
    p = np.atleast_2d(p)
    a = np.atleast_2d(a)
    b = np.atleast_2d(b)
    c = np.atleast_2d(c)
    
    # v0 = c - a, v1 = b - a, v2 = p - a
    v0 = c[0] - a[0]
    v1 = b[0] - a[0]
    v2 = p - a[0]
    
    d00 = np.sum(v0 * v0)
    d01 = np.sum(v0 * v1)
    d02 = np.sum(v0 * v2, axis=1)
    d11 = np.sum(v1 * v1)
    d12 = np.sum(v1 * v2, axis=1)
    
    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-10:
        return np.zeros(len(p), dtype=bool)
        
    u = (d11 * d02 - d01 * d12) / denom
    v = (d00 * d12 - d01 * d02) / denom
    
    return (u >= 0) & (v >= 0) & (u + v <= 1)

def min_triangle_area(points):
    n = len(points)
    # Precompute indices for all triangles
    # This is static for N=11
    if not hasattr(min_triangle_area, "indices"):
        idx = []
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    idx.append((i, j, k))
        min_triangle_area.indices = np.array(idx)
        
    indices = min_triangle_area.indices
    # Vectorized area calculation
    # points[indices[:, 0]], etc.
    p1 = points[indices[:, 0]]
    p2 = points[indices[:, 1]]
    p3 = points[indices[:, 2]]
    
    # Area = 0.5 * |x1(y2 - y3) + x2(y3 - y1) + x3(y1 - y2)|
    # Or 0.5 * |(p2 - p1) x (p3 - p1)|
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)

def objective(x):
    points = x.reshape(-1, 2)
    a = np.array([0.0, 0.0])
    b = np.array([1.0, 0.0])
    c = np.array([0.5, np.sqrt(3) / 2])
    
    # Check bounds
    inside = point_in_triangle(points, a, b, c)
    if not np.all(inside):
        # Penalty proportional to violation? Or just large constant.
        # Large constant is fine for Nelder-Mead.
        return 1e6
            
    area = min_triangle_area(points)
    return -area

def construct():
    a = np.array([0.0, 0.0])
    b = np.array([1.0, 0.0])
    c = np.array([0.5, np.sqrt(3) / 2])
    
    best_points = None
    best_score = -1.0
    
    # Start with a good hexagonal-like guess
    # 11 points in equilateral triangle
    # Approximate coordinates
    initial_guesses = []
    
    # Guess 1: Hexagonal packing approximation
    # Row 1 (bottom): 4 points
    # Row 2: 4 points
    # Row 3: 3 points
    # Adjusted for triangle shape
    h = np.sqrt(3) / 2
    
    # Simple grid
    pts1 = []
    for i in range(4):
        pts1.append([0.1 + i * 0.25, 0.1])
    for i in range(4):
        pts1.append([0.15 + i * 0.25, 0.35])
    for i in range(3):
        pts1.append([0.2 + i * 0.3, 0.6])
    initial_guesses.append(np.array(pts1).flatten())
    
    # Guess 2: Random
    rng = np.random.RandomState(42)
    for _ in range(10):
        pts = []
        for _ in range(N):
            while True:
                u = rng.random()
                v = rng.random()
                if u + v < 1.0:
                    p = u * b + v * c
                    pts.append(p)
                    break
        initial_guesses.append(np.array(pts).flatten())
        
    # Run optimization for each guess
    for x0 in initial_guesses:
        try:
            res = minimize(objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-10})
            
            if res.fun < 1e5:
                current_area = -res.fun
                if current_area > best_score:
                    best_score = current_area
                    best_points = res.x.reshape(-1, 2)
                    
                    # Optional: Add jittered versions of the best point to initial_guesses
                    # to explore nearby local minima
                    if len(initial_guesses) < 20:
                        jitter = np.random.normal(0, 0.01, size=x0.shape)
                        new_x0 = res.x + jitter
                        # Ensure new_x0 is still somewhat valid? Nelder-Mead will handle it.
                        initial_guesses.append(new_x0)
        except Exception as e:
            continue
            
    if best_points is None:
        # Fallback
        rng = np.random.RandomState(7)
        points = []
        while len(points) < N:
            u, v = rng.random(), rng.random()
            if u + v < 0.98:
                points.append([u + 0.5 * v, (3 ** 0.5) / 2 * v])
        best_points = np.array(points)
        
    return best_points
