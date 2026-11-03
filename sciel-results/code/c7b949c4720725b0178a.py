import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE = 150.0 # seconds
START_TIME = time.time()

def _check_deadline():
    return time.time() - START_TIME > DEADLINE - 5.0

def construct():
    # Ring structure: 1, 6, 12, 18, 13
    rings = [1, 6, 12, 18, 13]
    
    # Initial guess for ring radii and angles
    # Radii should be increasing
    # Angles should be staggered
    # Let's estimate radii based on optimal packing
    # Optimal r ~ 0.1258
    # Distances between rings roughly 2*r
    # r1 ~ 2*r, r2 ~ 4*r, etc.
    # But they are not perfectly spaced due to geometry.
    
    # Initial parameters: [r1, r2, r3, r4, th1, th2, th3, th4]
    # r_i are radii of the rings
    # th_i are angular offsets
    
    # Heuristic initial values
    # The outermost ring is close to 1.
    # r4 ~ 0.8
    # r3 ~ 0.55
    # r2 ~ 0.35
    # r1 ~ 0.15
    
    x0 = np.array([0.15, 0.35, 0.55, 0.80, 0.0, 0.0, 0.0, 0.0])
    
    def generate_points(x):
        r1, r2, r3, r4 = x[0], x[1], x[2], x[3]
        th1, th2, th3, th4 = x[4], x[5], x[6], x[7]
        
        points = []
        # Center
        points.append([0.0, 0.0])
        
        # Ring 1
        for i in range(6):
            angle = th1 + 2 * np.pi * i / 6
            points.append([r1 * np.cos(angle), r1 * np.sin(angle)])
            
        # Ring 2
        for i in range(12):
            angle = th2 + 2 * np.pi * i / 12
            points.append([r2 * np.cos(angle), r2 * np.sin(angle)])
            
        # Ring 3
        for i in range(18):
            angle = th3 + 2 * np.pi * i / 18
            points.append([r3 * np.cos(angle), r3 * np.sin(angle)])
            
        # Ring 4
        for i in range(13):
            angle = th4 + 2 * np.pi * i / 13
            points.append([r4 * np.cos(angle), r4 * np.sin(angle)])
            
        return np.array(points)
    
    def objective(x):
        if _check_deadline():
            return 1e10
        pts = generate_points(x)
        # Calculate min distance between pairs
        # And min distance to boundary (1 - r_i)
        # We want to maximize min(dist_pairs/2, 1 - dist_center)
        # So we minimize -min(...)
        
        # Pairwise distances
        # Efficient calculation for 50 points is fine
        dists = np.linalg.norm(pts[:, np.newaxis, :] - pts[np.newaxis, :, :], axis=2)
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        
        # Distances to boundary
        radii = np.linalg.norm(pts, axis=1)
        min_boundary_dist = np.min(1.0 - radii)
        
        # The circle radius is half the min pair distance
        # And limited by the boundary distance
        r_eff = min(min_pair_dist / 2.0, min_boundary_dist)
        
        # We want to maximize r_eff, so minimize -r_eff
        return -r_eff

    # Run optimization
    best_x = x0
    best_val = objective(x0)
    
    # Try a few different initial angles/radii perturbations
    # and run Nelder-Mead
    # Nelder-Mead is slow but robust.
    # We have ~100s.
    # Let's run one main optimization.
    
    try:
        res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 10000, 'xatol': 1e-8, 'fatol': 1e-10})
        if res.fun < best_val:
            best_x = res.x
            best_val = res.fun
    except Exception:
        pass
    
    if _check_deadline():
        # Return best found so far
        pts = generate_points(best_x)
        return pts
    
    # Refine: Use the result from ring optimization as initial guess for full local search
    # This allows individual circles to break the ring symmetry if beneficial
    pts = generate_points(best_x)
    
    def objective_full(pts_flat):
        if _check_deadline():
            return 1e10
        pts = pts_flat.reshape(-1, 2)
        dists = np.linalg.norm(pts[:, np.newaxis, :] - pts[np.newaxis, :, :], axis=2)
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        radii = np.linalg.norm(pts, axis=1)
        min_boundary_dist = np.min(1.0 - radii)
        r_eff = min(min_pair_dist / 2.0, min_boundary_dist)
        return -r_eff

    x_full0 = pts.flatten()
    
    try:
        # Use L-BFGS-B for speed, but it needs gradients or finite differences.
        # Nelder-Mead on 100 vars is slow.
        # Let's try a few iterations of simple local search or L-BFGS-B.
        # L-BFGS-B with finite differences might be okay.
        res_full = minimize(objective_full, x_full0, method='L-BFGS-B', options={'maxiter': 2000, 'ftol': 1e-12})
        if res_full.fun < best_val:
            best_x_full = res_full.x
            best_val = res_full.fun
            pts = best_x_full.reshape(-1, 2)
    except Exception:
        pass

    # Final validation and return
    pts = generate_points(best_x) if not hasattr(res_full, 'x') else res_full.x.reshape(-1, 2)
    
    # Ensure validity
    # If any point is outside or overlapping, shrink?
    # The evaluator checks validity. If invalid, score 0.
    # We should ensure the returned points are strictly valid.
    # The optimization should keep them valid if initialized valid and steps are small,
    # but L-BFGS-B might step outside.
    # Let's verify.
    
    radii = np.linalg.norm(pts, axis=1)
    if np.any(radii > 1.0):
        # Scale down to fit
        max_r = np.max(radii)
        pts = pts * (0.99 * 1.0 / max_r)
        
    dists = np.linalg.norm(pts[:, np.newaxis, :] - pts[np.newaxis, :, :], axis=2)
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    r_calc = min(min_dist / 2.0, np.min(1.0 - np.linalg.norm(pts, axis=1)))
    
    # If r_calc is very small or invalid, fall back to ring solution
    if r_calc < 0.1:
         pts = generate_points(best_x)

    return pts
