import numpy as np
from scipy.optimize import minimize
import itertools
import time

N = 16
D = 2

def pairwise_dists_squared(points):
    # points: (N, 2)
    # returns (N, N) matrix of squared distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    return dists_sq

def softplus(x, eps=1.0):
    # Smooth approximation of max(0, x)
    # Using log(1 + exp(x/eps)) * eps
    # To avoid overflow in exp, use np.logaddexp(0, x/eps) * eps
    return np.logaddexp(0, x / eps) * eps

def objective(params, eps1, eps2, lam1, lam2):
    # params: (33,) -> 16 points (32 floats) + R (1 float)
    x = params[:32].reshape(N, D)
    R = params[32]
    
    d_sq = pairwise_dists_squared(x)
    mask = ~np.eye(N, dtype=bool)
    d_flat = d_sq[mask]
    
    # Penalty for d_sq > R
    # We want d_sq <= R. Violation is d_sq - R.
    pen1 = np.sum(softplus(d_flat - R, eps1))
    
    # Penalty for d_sq < 1
    # We want d_sq >= 1. Violation is 1 - d_sq.
    pen2 = np.sum(softplus(1 - d_flat, eps2))
    
    return R + lam1 * pen1 + lam2 * pen2

def exact_score(points):
    d_sq = pairwise_dists_squared(points)
    mask = ~np.eye(N, dtype=bool)
    d_flat = d_sq[mask]
    min_d = np.min(d_flat)
    max_d = np.max(d_flat)
    if min_d < 1e-9:
        return float('inf')
    return max_d / min_d

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin for 150s limit
    
    best_score = float('inf')
    best_points = None
    
    # Initial configurations
    inits = []
    
    # 1. Hexagonal packing
    # 16 points in hex lattice
    # 4x4 block is 16 points
    hex_points = []
    for row in range(4):
        for col in range(4):
            x = col + (row % 2) * 0.5
            y = row * np.sqrt(3)/2
            hex_points.append([x, y])
    inits.append(np.array(hex_points, dtype=float))
    
    # 2. Grid
    grid = list(itertools.product(range(4), repeat=2))
    inits.append(np.array(grid, dtype=float))
    
    # 3. Concentric circles
    # 1 center, 5 inner, 10 outer?
    # Or 4 rings?
    # Let's try 1 + 5 + 10 = 16
    pts = [[0, 0]]
    for i in range(5):
        angle = 2 * np.pi * i / 5
        pts.append([np.cos(angle), np.sin(angle)])
    for i in range(10):
        angle = 2 * np.pi * i / 10 + np.pi/10
        pts.append([2*np.cos(angle), 2*np.sin(angle)])
    inits.append(np.array(pts, dtype=float))
    
    # 4. Random
    np.random.seed(42)
    inits.append(np.random.uniform(-2, 2, (N, D)))
    
    # Optimization parameters
    # We will run a few iterations with decreasing epsilon
    eps_schedule = [1.0, 0.1, 0.01, 0.001]
    lam_schedule = [10.0, 100.0, 1000.0, 10000.0]
    
    for init in inits:
        if time.time() > deadline:
            break
            
        # Normalize initial points to roughly unit min distance
        d_sq = pairwise_dists_squared(init)
        mask = ~np.eye(N, dtype=bool)
        d_flat = d_sq[mask]
        min_d = np.min(d_flat)
        if min_d > 0:
            init = init / np.sqrt(min_d)
        
        current_params = np.concatenate([init.flatten(), [np.max(d_flat / min_d)]])
        
        for i, (eps, lam) in enumerate(zip(eps_schedule, lam_schedule)):
            if time.time() > deadline:
                break
                
            def obj(p):
                return objective(p, eps, eps, lam, lam)
            
            try:
                res = minimize(obj, current_params, method='L-BFGS-B', 
                               options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
                if res.fun < 1e10: # Check for validity
                    pts = res.x[:32].reshape(N, D)
                    score = exact_score(pts)
                    if score < best_score:
                        best_score = score
                        best_points = pts
                    current_params = res.x
            except Exception:
                pass
                
    if best_points is None:
        return np.array(inits[0], dtype=float)
    
    return best_points
