import numpy as np
from scipy.optimize import minimize
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    n = 8
    
    def compute_score(points):
        # points is shape (n, 2)
        # 1. Calculate pairwise distances
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        
        # Avoid log(0)
        dists = np.maximum(dists, 1e-10)
        
        # 2. Find diameter
        mask = ~np.eye(n, dtype=bool)
        max_dist = np.max(dists[mask])
        
        if max_dist == 0:
            return -np.inf
            
        # 3. Compute log of product of ordered pair distances
        # Product is over i != j
        # log(product) = sum(log(dists[i, j])) for i != j
        log_prod_ordered = np.sum(np.log(dists[mask]))
        
        # 4. Objective to maximize:
        # K = log_prod_ordered - 56 * log(max_dist)
        # We minimize -K
        return log_prod_ordered - 56 * np.log(max_dist)

    # Strategy: Exploit D4 symmetry
    # The regular octagon has D4 symmetry. 
    # A general D4-symmetric set of 8 points can be parameterized by:
    # 4 points on the x-axis, 4 on the y-axis? No, that's 8 points but specific structure.
    # General D4 orbit of a point (x, y) with x != y, x != 0, y != 0 has 8 points:
    # (±x, ±y), (±y, ±x)
    # This covers the regular octagon case where x = cos(22.5), y = sin(22.5).
    # Let's parameterize by x and y.
    # We assume x > y > 0 to avoid duplicates and ensure distinctness.
    
    def d4_points(params):
        x, y = params
        # Ensure x > y > 0 to maintain ordering and distinctness
        # If optimizer tries to cross, we penalize or project
        if x < y:
            x, y = y, x
        if y < 0:
            y = abs(y)
        if x < 0:
            x = abs(x)
            
        # Generate 8 points
        p1 = np.array([x, y])
        p2 = np.array([x, -y])
        p3 = np.array([-x, y])
        p4 = np.array([-x, -y])
        p5 = np.array([y, x])
        p6 = np.array([y, -x])
        p7 = np.array([-y, x])
        p8 = np.array([-y, -x])
        
        return np.vstack([p1, p2, p3, p4, p5, p6, p7, p8])

    def objective_d4(params):
        pts = d4_points(params)
        # Check for distinct points
        # If x == y, points 1 and 5 are same. If x == 0 or y == 0, collisions.
        # Regular octagon: x = cos(pi/8), y = sin(pi/8).
        # Let's just compute score. If points collide, dists will be 0, log(0) -> -inf, score -inf.
        s = compute_score(pts)
        if not np.isfinite(s):
            return 1e10
        return -s

    best_score = -np.inf
    best_points = None

    # Initial guesses for (x, y)
    # Regular octagon:
    ang = np.pi / 8
    x_oct = np.cos(ang)
    y_oct = np.sin(ang)
    
    candidates_xy = []
    
    # 1. Near regular octagon
    candidates_xy.append([x_oct, y_oct])
    
    # 2. Random perturbations in D4 space
    rng = np.random.default_rng(42)
    for _ in range(200):
        # x and y should be positive. 
        # Scale is arbitrary due to normalization, but let's keep them in [0.1, 2.0]
        x = rng.uniform(0.1, 2.0)
        y = rng.uniform(0.05, x) # y < x to avoid symmetry duplicates
        candidates_xy.append([x, y])
        
    # 3. Specific structured guesses
    # Maybe the optimal solution breaks the "x,y" symmetry slightly? 
    # But D4 is the natural symmetry group for n=8.
    # Let's also try points on the axes (which is a subset of D4 where y=0 or x=0)
    # If y=0, points are (±x, 0) and (0, ±x). This is a square. 
    # A square of 4 points? No, we need 8 distinct points.
    # If y=0, we have (x,0), (x,0)... duplicates. So y must be > 0 and x != y.
    
    # Let's refine the search using L-BFGS-B which handles bounds well
    bounds = [(0.01, 3.0), (0.01, 3.0)]
    
    for i, xy in enumerate(candidates_xy):
        if time.time() > deadline:
            break
            
        x0 = np.array(xy)
        # Ensure x > y for the specific parameterization used in d4_points
        # The function d4_points swaps if needed, but for optimizer, let's keep consistent
        if x0[0] < x0[1]:
            x0 = np.array([x0[1], x0[0]])
            
        try:
            # Use L-BFGS-B for speed and precision
            res = minimize(objective_d4, x0, method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-12})
            
            if res.success:
                final_pts = d4_points(res.x)
                final_score = compute_score(final_pts)
                
                if final_score > best_score:
                    best_score = final_score
                    best_points = final_pts
                    
        except Exception:
            pass

    # If D4 search didn't yield anything better than octagon, or failed, 
    # fallback to a broader search but still constrained to D4? 
    # Or maybe the optimal solution is NOT D4 symmetric?
    # The regular octagon is D4. Perturbing it usually breaks symmetry to C1 or C2.
    # However, for "maximizing product of distances", symmetry is often preserved at the optimum.
    # Let's add a few Nelder-Mead runs on the full 16D space starting from the best D4 point
    # to see if breaking symmetry helps.
    
    if best_points is not None and time.time() < deadline - 20:
        # Start from best D4 point
        x0 = best_points.flatten()
        
        def objective_full(x):
            pts = x.reshape(n, 2)
            s = compute_score(pts)
            if not np.isfinite(s):
                return 1e10
            return -s

        try:
            res = minimize(objective_full, x0, method='Nelder-Mead', 
                           options={'maxiter': 5000, 'xatol': 1e-9, 'fatol': 1e-12})
            final_pts = res.x.reshape(n, 2)
            final_score = compute_score(final_pts)
            
            if final_score > best_score:
                best_score = final_score
                best_points = final_pts
        except Exception:
            pass

    # Final check: if we didn't find anything, return regular octagon
    if best_points is None:
        angles = 2 * np.pi * np.arange(n) / n
        p_oct = np.column_stack([np.cos(angles), np.sin(angles)])
        return [[float(x), float(y)] for x, y in p_oct]
        
    return [[float(x), float(y)] for x, y in best_points]
