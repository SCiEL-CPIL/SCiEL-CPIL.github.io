"""Heilbronn triangles: place 11 points in an equilateral triangle to maximize the minimum triangle area."""
import numpy as np
import time
from scipy.optimize import minimize
from itertools import combinations

N = 11

# Vertices of the equilateral triangle
A = np.array([0.0, 0.0])
B = np.array([1.0, 0.0])
C = np.array([0.5, np.sqrt(3) / 2.0])

def _to_cartesian(bary):
    """Convert barycentric coords (a,b,c) with a+b+c=1 to cartesian."""
    # bary[i] = [a_i, b_i, c_i] for each point
    return bary[:, 0:1] * A + bary[:, 1:2] * B + bary[:, 2:3] * C

def _min_area_from_bary(bary):
    """Compute minimum triangle area from barycentric coordinates."""
    pts = _to_cartesian(bary)
    n = len(pts)
    min_a = np.inf
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                p1, p2, p3 = pts[i], pts[j], pts[k]
                area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
                if area < min_a:
                    min_a = area
    return min_a

def _objective_neg(bary_flat):
    """Negative of min triangle area (we minimize this)."""
    bary = bary_flat.reshape(N, 3)
    # Enforce barycentric: a+b+c=1, a,b,c >= 0
    # Use softmax-like normalization or just clip
    bary = np.clip(bary, 1e-8, None)
    bary = bary / bary.sum(axis=1, keepdims=True)
    return -_min_area_from_bary(bary)

def _objective_grad(bary_flat):
    """Gradient of negative min area (numerical or analytical)."""
    # Use finite differences for simplicity
    eps = 1e-7
    grad = np.zeros_like(bary_flat)
    f0 = _objective_neg(bary_flat)
    for i in range(len(bary_flat)):
        old = bary_flat[i]
        bary_flat[i] += eps
        f1 = _objective_neg(bary_flat)
        bary_flat[i] = old
        grad[i] = (f1 - f0) / eps
    return grad

def _construct_one(start_bary=None):
    """Optimize from a single starting point."""
    if start_bary is None:
        # Random initialization
        start_bary = np.random.dirichlet(np.ones(N), size=1).T
    else:
        start_bary = start_bary.copy()
    
    start_flat = start_bary.flatten()
    
    # Bounds: each barycentric coord in [0, 1]
    bounds = [(0.0, 1.0)] * (N * 3)
    
    result = minimize(
        _objective_neg,
        start_flat,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
    )
    
    best_bary = result.x.reshape(N, 3)
    best_bary = np.clip(best_bary, 1e-8, None)
    best_bary = best_bary / best_bary.sum(axis=1, keepdims=True)
    
    return best_bary, -result.fun

def _polish(bary):
    """Final high-precision polish pass."""
    bary = bary.copy()
    for _ in range(5):
        start_flat = bary.flatten()
        bounds = [(0.0, 1.0)] * (N * 3)
        result = minimize(
            _objective_neg,
            start_flat,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 5000, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 100}
        )
        bary = result.x.reshape(N, 3)
        bary = np.clip(bary, 1e-8, None)
        bary = bary / bary.sum(axis=1, keepdims=True)
    
    return bary

def construct():
    """Construct 11 points in the equilateral triangle to maximize min triangle area."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    best_bary = None
    best_area = -np.inf
    
    # Multiple random restarts
    np.random.seed(42)
    
    n_restarts = 0
    while True:
        if time.time() > deadline - 10.0:
            break
        
        # Random initialization with Dirichlet distribution
        start_bary = np.random.dirichlet(np.ones(N), size=1).T
        
        # Optimize
        bary, area = _construct_one(start_bary)
        
        if area > best_area:
            best_area = area
            best_bary = bary.copy()
        
        n_restarts += 1
        
        # Check if we have time for polish
        if time.time() > deadline - 15.0:
            break
    
    # Final polish pass if we have time
    if best_bary is not None and time.time() < deadline - 5.0:
        best_bary = _polish(best_bary)
        best_area = _min_area_from_bary(best_bary)
    
    # Convert to cartesian
    if best_bary is None:
        # Fallback: simple random
        best_bary = np.random.dirichlet(np.ones(N), size=1).T
    
    points = _to_cartesian(best_bary)
    
    return np.array(points)
