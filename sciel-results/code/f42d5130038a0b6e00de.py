"""Improved program for Heilbronn triangles: maximize smallest triangle area for 11 points."""
import numpy as np
from scipy.optimize import minimize
import time
from itertools import combinations

N = 11
h = np.sqrt(3) / 2
A = np.array([0.0, 0.0])
B = np.array([1.0, 0.0])
C = np.array([0.5, h])

def project_to_triangle(p):
    """
    p: array of shape (N, 2)
    Returns projected points inside the triangle (0,0), (1,0), (0.5, h)
    Uses barycentric coordinates for robust projection.
    """
    p = p.copy()
    # Barycentric coordinates relative to triangle ABC
    # A=(0,0), B=(1,0), C=(0.5, h)
    # For point p = (x, y):
    # lambda1 = 1 - 0.5*x - y/h  (weight for A)
    # lambda2 = 0.5*x + y/(2*h)  ... let me recompute
    
    # Standard barycentric:
    # v0 = B - A = (1, 0)
    # v1 = C - A = (0.5, h)
    # v2 = p - A = (x, y)
    
    # d00 = v0·v0 = 1
    # d01 = v0·v1 = 0.5
    # d11 = v1·v1 = 0.25 + 3/4 = 1
    # d02 = v0·v2 = x
    # d12 = v1·v2 = 0.5*x + h*y
    
    # denom = d00*d11 - d01*d01 = 1*1 - 0.25 = 0.75
    # w1 = (d11*d02 - d01*d12) / denom = (1*x - 0.5*(0.5x + hy)) / 0.75 = (x - 0.25x - 0.5hy)/0.75 = (0.75x - 0.5hy)/0.75 = x - (2/3)hy
    # w2 = (d00*d12 - d01*d02) / denom = (1*(0.5x + hy) - 0.5*x)/0.75 = (0.5x + hy - 0.5x)/0.75 = hy/0.75 = (4/3)hy
    # w0 = 1 - w1 - w2
    
    x = p[:, 0]
    y = p[:, 1]
    
    w1 = x - (2.0 / 3.0) * h * y
    w2 = (4.0 / 3.0) * h * y
    w0 = 1.0 - w1 - w2
    
    # Clip weights to [0, 1] and renormalize
    w0 = np.clip(w0, 0, 1)
    w1 = np.clip(w1, 0, 1)
    w2 = np.clip(w2, 0, 1)
    
    # Renormalize
    total = w0 + w1 + w2
    total[total < 1e-10] = 1.0
    w0 /= total
    w1 /= total
    w2 /= total
    
    # Reconstruct points
    p[:, 0] = w0 * 0.0 + w1 * 1.0 + w2 * 0.5
    p[:, 1] = w0 * 0.0 + w1 * 0.0 + w2 * h
    
    return p

def min_triangle_area(points):
    """
    Compute the minimum area of any triangle formed by 3 of the N points.
    points: array of shape (N, 2)
    Returns the minimum area (positive value).
    """
    n = len(points)
    min_area = np.inf
    
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                # Area = 0.5 * |cross product|
                v1 = points[j] - points[i]
                v2 = points[k] - points[i]
                cross = v1[0] * v2[1] - v1[1] * v2[0]
                area = 0.5 * abs(cross)
                if area < min_area:
                    min_area = area
                    if min_area < 1e-10:
                        return 0.0
    return min_area

def objective_flat(x):
    """
    Objective function for scipy.optimize.minimize.
    We want to maximize the minimum triangle area, so we minimize the negative.
    x: flat array of length 2*N
    """
    points = x.reshape(N, 2)
    points = project_to_triangle(points)
    area = min_triangle_area(points)
    return -area

def construct():
    deadline = time.time() + 140  # Safety margin before 150s limit
    best_points = None
    best_area = -np.inf
    
    # Initialize with a grid-like or known good configuration
    # Start with points at vertices and some interior points
    # Known heuristic: place points near vertices and distribute interior points
    
    # Initial guess: corners + evenly distributed interior
    initial_points = np.zeros((N, 2))
    # Place 3 points near the corners (slightly inside)
    eps = 0.05
    initial_points[0] = [eps, eps]
    initial_points[1] = [1 - eps, eps]
    initial_points[2] = [0.5, h * (1 - eps)]
    
    # Remaining 8 points: distribute in interior
    # Use a hexagonal-ish pattern or random
    rng = np.random.RandomState(42)
    for i in range(3, N):
        u, v = rng.random(), rng.random()
        while u + v > 1:
            u, v = rng.random(), rng.random()
        # Map to triangle
        x = u + 0.5 * v
        y = h * v
        initial_points[i] = [x, y]
    
    best_points = project_to_triangle(initial_points.copy())
    best_area = min_triangle_area(best_points)
    
    # Run multiple Nelder-Mead optimizations from different starting points
    for restart in range(30):
        if time.time() > deadline:
            break
        
        if restart == 0:
            x0 = initial_points.flatten().copy()
        else:
            # Perturb the best solution or use random restart
            if restart % 3 == 0:
                # Random restart
                x0 = rng.random((N, 2))
                # Map to triangle
                for i in range(N):
                    u, v = x0[2*i], x0[2*i+1]
                    while u + v > 1:
                        u, v = rng.random(), rng.random()
                    x0[2*i] = u + 0.5 * v
                    x0[2*i+1] = h * v
            else:
                # Perturb best
                x0 = best_points.flatten().copy()
                noise = 0.05 * rng.randn(N, 2)
                x0 = x0 + noise.flatten()
        
        try:
            result = minimize(
                objective_flat,
                x0,
                method='Nelder-Mead',
                options={
                    'adaptive': True,
                    'maxiter': 1000,
                    'xatol': 1e-6,
                    'fatol': 1e-8
                }
            )
            
            if result.success or result.fun < -best_area:
                points = project_to_triangle(result.x.reshape(N, 2))
                area = min_triangle_area(points)
                if area > best_area:
                    best_area = area
                    best_points = points.copy()
        except Exception:
            continue
    
    # Return the best points found
    return best_points

# Verify the function works
if __name__ == "__main__":
    pts = construct()
    print("Points:", pts)
    print("Min triangle area:", min_triangle_area(pts))
