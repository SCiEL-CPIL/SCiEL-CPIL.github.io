import numpy as np
import time
from scipy.optimize import minimize
import itertools

N = 11

def triangle_area(p1, p2, p3):
    return 0.5 * abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1]))

def min_triangle_area(points):
    min_a = float('inf')
    for i in range(N):
        for j in range(i+1, N):
            for k in range(j+1, N):
                a = triangle_area(points[i], points[j], points[k])
                if a < min_a:
                    min_a = a
    return min_a

def is_inside_triangle(p, h):
    # h is height of triangle
    if p[0] < -1e-9 or p[1] < -1e-9 or p[0] > 1 + 1e-9:
        return False
    # Line AC: y = sqrt(3) * x  (if A is (0,0) and C is (0.5, sqrt(3)/2)? No.)
    # Vertices: (0,0), (1,0), (0.5, sqrt(3)/2)
    # Edge AB: y >= 0
    # Edge BC: y <= -sqrt(3)*(x-1) = sqrt(3)*(1-x)
    # Edge AC: y <= sqrt(3)*x
    if p[1] < -1e-9: return False
    if p[1] > 1e-9 + np.sqrt(3)*(1-p[0]): return False
    if p[1] > 1e-9 + np.sqrt(3)*p[0]: return False
    return True

def objective(x):
    points = x.reshape(-1, 2)
    # Check if inside
    # If any point is outside, return a large penalty
    for p in points:
        if not is_inside_triangle(p, np.sqrt(3)/2):
            return 1e10
    return -min_triangle_area(points)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s margin
    
    best_points = None
    best_score = -1e10

    # Generate initial guesses
    guesses = []
    
    # Guess 1: Symmetric
    # Centroid
    G = np.array([0.5, np.sqrt(3)/6])
    # Vectors to vertices
    vA = np.array([0,0]) - G
    vB = np.array([1,0]) - G
    vC = np.array([0.5, np.sqrt(3)/2]) - G
    
    def make_symmetric(r1, r2, r3, r4):
        pts = [G]
        for r in [r1, r2, r3]:
            for v in [vA, vB, vC]:
                norm = np.linalg.norm(v)
                pts.append(G + (r/norm)*v)
        # 10 points so far. Add one at a vertex?
        pts.append(np.array([0,0])) # Vertex A
        return np.array(pts).flatten()
        
    guesses.append(make_symmetric(0.1, 0.2, 0.3, 0.4))
    guesses.append(make_symmetric(0.15, 0.3, 0.45, 0.6))
    
    # Guess 2: Random
    rng = np.random.default_rng(42)
    for _ in range(5):
        pts = []
        while len(pts) < N:
            u = rng.random()
            v = rng.random()
            if u + v < 1:
                x = u + 0.5*v
                y = np.sqrt(3)/2 * v
                pts.append([x, y])
        guesses.append(np.array(pts).flatten())

    # Guess 3: Lattice-like
    # Just use a grid and pick 11
    # ... skipping complex lattice generation, random + symmetric is usually enough

    for guess in guesses:
        if time.time() > deadline - 10:
            break
            
        # Optimize
        try:
            res = minimize(objective, guess, method='Nelder-Mead', options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8})
            if res.fun < best_score:
                best_score = -res.fun
                best_points = res.x.reshape(-1, 2)
        except Exception as e:
            pass
            
    # If no valid points found, return a safe default
    if best_points is None:
        # Return random points
        pts = []
        while len(pts) < N:
            u = np.random.random()
            v = np.random.random()
            if u + v < 1:
                pts.append([u + 0.5*v, np.sqrt(3)/2 * v])
        return np.array(pts)
        
    return best_points
