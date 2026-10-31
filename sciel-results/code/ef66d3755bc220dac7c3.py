import numpy as np
import time
from scipy.optimize import minimize

N = 16
D = 2

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin

    def score_points(pts):
        """Compute (max_dist / min_dist)^2 for a set of points."""
        # Ensure distinct points
        # diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        # dists = np.sqrt(np.sum(diff**2, axis=2))
        # iu = np.triu_indices(N, 1)
        # d = dists[iu]
        # min_d = np.min(d)
        # max_d = np.max(d)
        
        # Vectorized for speed
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        iu = np.triu_indices(N, 1)
        d = dists[iu]
        min_d = np.min(d)
        max_d = np.max(d)
        if min_d < 1e-8:
            return 1e10
        return (max_d / min_d)**2

    # 1. Generate Hexagonal Lattice
    # Radius 2 hexagon has 19 points
    # Coordinates for hexagonal lattice:
    # (i + 0.5*j, sqrt(3)/2 * j)
    hex_pts = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            if max(abs(q), abs(r), abs(q+r)) <= 2:
                x = q + 0.5 * r
                y = (np.sqrt(3) / 2) * r
                hex_pts.append((x, y))
    
    hex_pts = np.array(hex_pts)
    # This gives 19 points.
    # We need 16.
    # Strategy: Try all combinations of removing 3 points from the 19.
    # There are C(19,3) = 969 combinations.
    # For each, run a quick local optimization? No, that's too many.
    # Instead, pick the 16 points that are closest to the center?
    # The 19 points are:
    # Center: (0,0)
    # Ring 1: 6 points
    # Ring 2: 12 points
    # Total 19.
    # If we keep Center + Ring 1 (7 points), we need 9 more from Ring 2.
    # Ring 2 points are at distance sqrt(3) from center? No.
    # Distances in hex lattice (unit edge):
    # (0,0) dist 0
    # (1,0) dist 1
    # (0,1) dist 1
    # (1,1) dist sqrt(3)
    # (2,0) dist 2
    # etc.
    
    # Let's just take the 16 points with smallest distance from origin.
    dists_from_origin = np.sum(hex_pts**2, axis=1)
    sorted_indices = np.argsort(dists_from_origin)
    initial_subset = sorted_indices[:N]
    best_pts = hex_pts[initial_subset]
    best_score = score_points(best_pts)
    
    # 2. Local Optimization using Nelder-Mead
    # To handle non-differentiability, Nelder-Mead is good.
    # We will run it on the best initial guess and a few others.
    
    def objective(params):
        pts = params.reshape(N, 2)
        # Centering helps with numerical stability and removes translation invariance
        pts = pts - np.mean(pts, axis=0)
        return score_points(pts)

    # Initial guess from hex lattice
    init_params = best_pts.flatten()
    
    # Run Nelder-Mead
    try:
        res = minimize(objective, init_params, method='Nelder-Mead', 
                       options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8})
        if res.fun < best_score:
            best_score = res.fun
            best_pts = res.x.reshape(N, 2)
    except Exception as e:
        pass

    # 3. Try other initial subsets
    # Remove different sets of 3 points from the 19-point hex cluster
    # The 19 points are indices 0..18 in hex_pts
    # Let's try removing 3 points from the outer ring (indices 7..18)
    # Outer ring has 12 points.
    # We need to keep 9 of them.
    # There are C(12, 9) = 220 combinations.
    # We can sample some of these.
    
    center_idx = 0 # (0,0)
    ring1_indices = [1, 2, 3, 4, 5, 6] # 6 points
    ring2_indices = list(range(7, 19)) # 12 points
    
    # We keep center + ring1 (7 points).
    # We need 9 from ring2.
    # Let's try removing 3 random points from ring2, 20 times.
    import random
    random.seed(42)
    
    for _ in range(20):
        if time.time() > deadline - 20:
            break
        # Choose 3 points to remove from ring2
        to_remove = random.sample(ring2_indices, 3)
        keep_ring2 = [i for i in ring2_indices if i not in to_remove]
        subset_indices = [center_idx] + ring1_indices + keep_ring2
        init_pts = hex_pts[subset_indices]
        
        # Quick score check
        s = score_points(init_pts)
        if s < best_score + 0.1: # Only optimize if promising
            init_params = init_pts.flatten()
            try:
                res = minimize(objective, init_params, method='Nelder-Mead', 
                               options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6})
                if res.fun < best_score:
                    best_score = res.fun
                    best_pts = res.x.reshape(N, 2)
            except Exception:
                pass

    # 4. Final Polish with L-BFGS-B on a smoothed objective?
    # Or just another Nelder-Mead run with tighter tolerance on the best result.
    if time.time() < deadline - 5:
        init_params = best_pts.flatten()
        try:
            res = minimize(objective, init_params, method='Nelder-Mead', 
                           options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10})
            if res.fun < best_score:
                best_score = res.fun
                best_pts = res.x.reshape(N, 2)
        except Exception:
            pass

    # Ensure distinct points
    # The optimization might cause points to merge.
    # If min_d is very small, the score is huge.
    # We should check if the final points are distinct.
    diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    iu = np.triu_indices(N, 1)
    d = dists[iu]
    if np.min(d) < 1e-6:
        # Fallback to hex lattice points
        best_pts = hex_pts[sorted_indices[:N]]

    return best_pts

# EVOLVE-BLOCK-END
