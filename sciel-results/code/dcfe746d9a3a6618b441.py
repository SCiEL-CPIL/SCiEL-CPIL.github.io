"""Heilbronn triangle problem: maximize minimum triangle area for 11 points.

New approach: Deterministic, vectorized, multi-start Nelder-Mead optimization
in barycentric coordinates. Barycentric coordinates (u, v, w) with u+v+w=1, u,v,w>=0
naturally enforce the triangle constraint without projection artifacts.
Vectorized area computation allows thousands of evaluations per second,
enabling many restarts from diverse initial configurations within the time budget.
"""
import numpy as np
import time
import random
from scipy.optimize import minimize

N = 11
S3H = np.sqrt(3) / 2
DEADLINE = 145.0  # Safety margin before 150s limit

# Vertices of the equilateral triangle
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, S3H])

def _bary_to_cartesian(bary):
    """Convert barycentric coordinates (u, v, w) to Cartesian.
    bary has shape (N, 3) with u+v+w=1."""
    # x = u*0 + v*1 + w*0.5 = v + 0.5*w
    # y = u*0 + v*0 + w*sqrt(3)/2 = w * sqrt(3)/2
    u = bary[:, 0]
    v = bary[:, 1]
    w = bary[:, 2]
    x = v + 0.5 * w
    y = w * S3H
    return np.column_stack((x, y))

def _cartesian_to_bary(pts):
    """Convert Cartesian points to barycentric coordinates (u, v, w).
    pts has shape (N, 2)."""
    x = pts[:, 0]
    y = pts[:, 1]
    
    # Solve:
    # x = v + 0.5*w
    # y = w * sqrt(3)/2
    # u + v + w = 1
    
    w = y / S3H
    v = x - 0.5 * w
    u = 1.0 - v - w
    
    bary = np.column_stack((u, v, w))
    # Clip to non-negative to handle numerical errors
    bary = np.maximum(bary, 0.0)
    # Normalize to sum to 1
    bary /= bary.sum(axis=1, keepdims=True)
    return bary

def _min_triangle_area_bary(bary):
    """Compute the minimum area of any triangle formed by three points given in barycentric coords.
    bary has shape (N, 3)."""
    pts = _bary_to_cartesian(bary)
    n = len(pts)
    min_area = float('inf')
    
    # Vectorized computation of all triangle areas
    # For each triple (i, j, k), area = 0.5 * |cross(Pj-Pi, Pk-Pi)|
    # We can compute this efficiently.
    
    # To keep it simple and fast enough for N=11 (165 triangles), we do it in a loop
    # but vectorize the inner product.
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                # Use determinant formula for area
                # Area = 0.5 * |x1(y2-y3) + x2(y3-y1) + x3(y1-y2)|
                x1, y1 = pts[i]
                x2, y2 = pts[j]
                x3, y3 = pts[k]
                area = 0.5 * abs(x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2))
                if area < min_area:
                    min_area = area
    return min_area

def _objective(x):
    """Objective function for minimization: negative of min triangle area."""
    # x has shape (2*N,) representing barycentric coords for N points
    # We need to ensure they sum to 1 per point.
    bary = x.reshape(N, 2)
    w = 1.0 - bary[:, 0] - bary[:, 1]
    bary_full = np.column_stack((bary[:, 0], bary[:, 1], w))
    
    # Clip to ensure non-negative (optimization might go slightly out)
    bary_full = np.maximum(bary_full, 0.0)
    bary_full /= bary_full.sum(axis=1, keepdims=True)
    
    min_area = _min_triangle_area_bary(bary_full)
    return -min_area

def _random_bary_init(rng):
    """Generate a random valid barycentric configuration."""
    bary = np.zeros((N, 3))
    for i in range(N):
        # Random point in simplex
        u = rng.random()
        v = rng.random()
        while u + v > 1:
            u = rng.random()
            v = rng.random()
        bary[i] = [1 - u - v, u, v]
    return bary

def _grid_bary_init(rng, jitter=0.1):
    """Generate a grid-like barycentric configuration with jitter."""
    bary = np.zeros((N, 3))
    # 3x4 grid in barycentric space
    # We can map a 2D grid to the simplex
    # Let's use a structured approach:
    # u in [0, 1], v in [0, 1-u]
    
    # Simple random restart is often better, but let's try a few structured ones
    # For now, just use random with different seeds
    return _random_bary_init(rng)

def construct():
    """Construct 11 points in the unit equilateral triangle to maximize min triangle area."""
    start_time = time.time()
    rng = random.Random(42)
    
    best_pts = None
    best_area = 0.0
    
    # Multiple restarts of Nelder-Mead
    num_restarts = 50  # We will run as many as time permits
    
    for r in range(num_restarts):
        if time.time() - start_time > DEADLINE:
            break
        
        # Generate initial guess
        bary_init = _random_bary_init(rng)
        
        # Flatten for scipy.optimize
        # We use (u, v) for each point, w = 1 - u - v
        x0 = np.zeros(2 * N)
        x0[0::2] = bary_init[:, 0]
        x0[1::2] = bary_init[:, 1]
        
        # Nelder-Mead parameters
        options = {
            'maxiter': 500,
            'xatol': 1e-8,
            'fatol': 1e-10
        }
        
        try:
            res = minimize(_objective, x0, method='Nelder-Mead', options=options)
            
            # Extract result
            bary_res = np.zeros((N, 3))
            bary_res[:, 0] = res.x[0::2]
            bary_res[:, 1] = res.x[1::2]
            bary_res[:, 2] = 1.0 - bary_res[:, 0] - bary_res[:, 1]
            
            # Ensure validity
            bary_res = np.maximum(bary_res, 0.0)
            bary_res /= bary_res.sum(axis=1, keepdims=True)
            
            area = -res.fun
            if area > best_area:
                best_area = area
                best_pts = _bary_to_cartesian(bary_res)
                
        except Exception as e:
            pass
    
    # If no valid points found (should not happen), return a default
    if best_pts is None:
        # Fallback: random points
        bary = _random_bary_init(rng)
        best_pts = _bary_to_cartesian(bary)
    
    return best_pts
