import numpy as np
from scipy.optimize import minimize
import time

def construct():
    N = 16
    start_time = time.time()
    deadline = 120.0  # Safety margin
    
    # 1. Generate initial hexagonal subset
    # Axial coordinates for hex lattice
    points_axial = []
    # Center
    points_axial.append((0, 0))
    # Ring 1
    for q, r in [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]:
        points_axial.append((q, r))
    # Ring 2
    for q, r in [(2, 0), (2, -1), (2, -2), (1, -2), (0, -2), (-1, -2), 
                 (-2, -1), (-2, 0), (-2, 1), (-1, 2), (0, 2), (1, 2)]:
        points_axial.append((q, r))
        
    # Remove 3 points to reduce diameter
    # Remove (2,0), (0,2), (2,-2)
    points_axial.remove((2, 0))
    points_axial.remove((0, 2))
    points_axial.remove((2, -2))
    
    # Convert to Cartesian
    def axial_to_cart(q, r):
        x = q + r / 2.0
        y = r * np.sqrt(3) / 2.0
        return (x, y)
        
    pts_init = np.array([axial_to_cart(q, r) for q, r in points_axial])
    
    # Calculate initial score
    def calc_ratio(pts):
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=-1))
        mask = ~np.eye(len(pts), dtype=bool)
        dists = dists[mask]
        d_max = np.max(dists)
        d_min = np.min(dists)
        if d_min < 1e-9:
            return 1e9
        return (d_max / d_min) ** 2

    best_pts = pts_init.copy()
    best_score = calc_ratio(best_pts)
    
    # 2. Optimization
    # Fix point 0 at (0,0) and point 1 at (1,0) to remove symmetries
    # Ensure pts_init[0] and pts_init[1] are distance 1 apart
    # In our hex subset, (0,0) is at (0,0).
    # Neighbors are at dist 1. Let's check which one is index 1.
    # Index 0: (0,0) -> (0,0)
    # Index 1: (1,0) -> (1, 0)
    # So pts_init[0] is (0,0) and pts_init[1] is (1,0).
    # This is perfect.
    
    # We optimize the remaining 14 points (indices 2 to 15)
    # Variables: 14 * 2 = 28
    
    def objective(x):
        # x is flat array of 28 values
        if time.time() > deadline:
            return 1e9
        pts_other = x.reshape(14, 2)
        pts = np.vstack((np.array([[0.0, 0.0], [1.0, 0.0]]), pts_other))
        return calc_ratio(pts)
        
    # Initial guess for variables
    x0 = best_pts[2:].flatten()
    
    # Use Nelder-Mead for robustness with non-smooth objective
    try:
        res = minimize(objective, x0, method='Nelder-Mead', 
                       options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6})
        
        if time.time() < deadline:
            res_pts = np.vstack((np.array([[0.0, 0.0], [1.0, 0.0]]), res.x.reshape(14, 2)))
            res_score = calc_ratio(res_pts)
            
            if res_score < best_score:
                best_pts = res_pts
                best_score = res_score
                
    except Exception:
        pass

    # Final check
    if time.time() > deadline:
        # Return best known
        pass
        
    return best_pts
