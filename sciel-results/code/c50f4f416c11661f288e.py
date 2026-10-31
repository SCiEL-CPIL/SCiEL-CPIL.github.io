"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16
D = 2

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

def _score(points):
    # points: (N, 2)
    # Center and scale
    pts = points - np.mean(points, axis=0)
    rms = np.sqrt(np.mean(np.sum(pts**2, axis=1)))
    if rms < 1e-10:
        return 1e10
    pts = pts / rms
    
    # Pairwise distances squared
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    
    # Exclude self-distances
    np.fill_diagonal(dists_sq, np.inf)
    
    min_d2 = np.min(dists_sq)
    max_d2 = np.max(dists_sq)
    
    if min_d2 < 1e-10:
        return 1e10
        
    return max_d2 / min_d2

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    # Initial guess: Hexagonal lattice 1+6+9
    pts = np.zeros((N, 2))
    idx = 1
    # Ring 1: 6 points
    for k in range(6):
        angle = 2 * np.pi * k / 6
        pts[idx] = [np.cos(angle), np.sin(angle)]
        idx += 1
    # Ring 2: 9 points
    # To maintain some symmetry, we can space them evenly
    # But 9 points on a ring of radius 2 might collide with ring 1?
    # Distance between ring 1 and ring 2 points.
    # If angle is 0, dist is 1.
    # If angle is pi/6, dist is sqrt(1^2+2^2-2*1*2*cos(pi/6)) = sqrt(5-2sqrt(3)) approx 0.51.
    # This is < 1, so they would overlap if min dist is 1.
    # So we need to adjust the radius or angles.
    # A better initial guess is to use the known optimal 16-point configuration if available,
    # or just let DE fix the overlaps.
    # Let's use a slightly larger radius for ring 2 to avoid initial overlaps, e.g., 2.2
    for k in range(9):
        angle = 2 * np.pi * k / 9
        pts[idx] = 2.2 * [np.cos(angle), np.sin(angle)]
        idx += 1
        
    x0 = pts.flatten()
    
    best_score = _score(pts)
    best_pts = pts.copy()
    
    # Differential Evolution
    bounds = [(-10, 10)] * (N * 2)
    
    try:
        res = differential_evolution(
            lambda x: _score(x.reshape(N, 2)), 
            bounds, 
            maxiter=100, 
            popsize=20, 
            tol=1e-4, 
            update_interval=5, 
            callback=lambda xk, convergence: (time.time() > deadline)
        )
        
        if res.fun < best_score:
            best_score = res.fun
            best_pts = res.x.reshape(N, 2)
            
        # Polish with Nelder-Mead
        if time.time() < deadline - 10:
            res2 = minimize(
                lambda x: _score(x.reshape(N, 2)),
                best_pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            if res2.fun < best_score:
                best_score = res2.fun
                best_pts = res2.x.reshape(N, 2)
                
    except Exception:
        pass
        
    return best_pts
# EVOLVE-BLOCK-END
