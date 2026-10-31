"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution

# Vertices of the equilateral triangle
A = np.array([0.0, 0.0])
B = np.array([1.0, 0.0])
C = np.array([0.5, np.sqrt(3) / 2])

def _inside_triangle(p):
    """Check if point p is inside the triangle ABC."""
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
    d1 = sign(p, A, B)
    d2 = sign(p, B, C)
    d3 = sign(p, C, A)
    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    return not (has_neg and has_pos)

def _area(p1, p2, p3):
    """Compute area of triangle formed by p1, p2, p3."""
    return 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))

def _min_area(points):
    """Compute the minimum area of any triangle formed by 3 points."""
    n = len(points)
    min_a = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = _area(points[i], points[j], points[k])
                if a < min_a:
                    min_a = a
    return min_a

def construct():
    start_time = time.time()
    deadline = start_time + 120  # 120 second deadline
    
    # Initial guess: use a symmetric arrangement
    # Center point + 10 points around
    center = (A + B + C) / 3
    points = [center]
    
    # Add points on a smaller concentric triangle and midpoints
    # Let's try a regular arrangement: 1 center, 3 on inner triangle, 6 on outer
    # Or just use a grid-like approach for initialization
    
    # Better initialization: use known good configurations or random
    # Let's start with a few random restarts and local optimization
    
    best_points = None
    best_val = 0
    
    # Generate several random initial configurations
    rng = np.random.RandomState(42)
    
    for restart in range(20):
        if time.time() > deadline - 10:
            break
            
        # Random points inside triangle
        pts = []
        for _ in range(N):
            while True:
                u = rng.random()
                v = rng.random()
                if u + v < 1:
                    p = u * B + v * C  # Barycentric coordinates
                    # Add small jitter to avoid degenerate triangles
                    p = p + rng.uniform(-0.01, 0.01, 2)
                    if _inside_triangle(p):
                        pts.append(p)
                        break
        
        pts = np.array(pts)
        val = _min_area(pts)
        if val > best_val:
            best_val = val
            best_points = pts.copy()
    
    # Now try to improve with local optimization
    # Use coordinate descent or Nelder-Mead on the objective
    
    from scipy.optimize import minimize
    
    def objective(x):
        pts = x.reshape(-1, 2)
        # Penalize points outside triangle
        penalty = 0
        for p in pts:
            if not _inside_triangle(p):
                penalty += 1000
        if penalty > 0:
            return penalty
        return -_min_area(pts)
    
    # Optimize from best random start
    x0 = best_points.flatten()
    
    # Use Nelder-Mead for local optimization
    for iteration in range(50):
        if time.time() > deadline - 5:
            break
            
        result = minimize(objective, x0, method='Nelder-Mead', 
                         options={'maxiter': 100, 'xatol': 1e-6, 'fatol': 1e-8})
        
        x_opt = result.x.reshape(-1, 2)
        # Ensure points are inside triangle
        valid = True
        for p in x_opt:
            if not _inside_triangle(p):
                valid = False
                break
        
        if valid:
            val = _min_area(x_opt)
            if val > best_val:
                best_val = val
                best_points = x_opt.copy()
        
        # Perturb and restart local optimization
        x0 = best_points.flatten() + rng.uniform(-0.05, 0.05, 2*N)
        # Project back inside triangle if needed
        for i in range(N):
            if not _inside_triangle(x0[2*i:2*i+2]):
                # Move towards center
                center = (A + B + C) / 3
                x0[2*i:2*i+2] = 0.5 * (x0[2*i:2*i+2] + center)
    
    # Final check and return
    if best_points is None:
        # Fallback: return random points
        pts = []
        for _ in range(N):
            while True:
                u = rng.random()
                v = rng.random()
                if u + v < 1:
                    p = u * B + v * C
                    pts.append(p)
                    break
        return np.array(pts)
    
    return best_points
# EVOLVE-BLOCK-END
