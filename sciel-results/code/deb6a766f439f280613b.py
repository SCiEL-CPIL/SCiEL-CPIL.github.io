"""
Strategy:
1. Fix diameter to 2 by pinning two points at (-1, 0) and (1, 0).
   This ensures the diameter is at least 2. Since the evaluator scales to diameter 2,
   this is equivalent to maximizing the product of distances for points constrained
   to the unit disk (centered at origin), with the condition that the max distance
   is exactly 2 (which is satisfied if the pinned points are the furthest, or if
   other points are inside the disk).
   Actually, if we pin points at (-1,0) and (1,0), the distance is 2.
   Any other point (x,y) must satisfy distance to (-1,0) <= 2 AND distance to (1,0) <= 2.
   This region is the intersection of two disks of radius 2 centered at (-1,0) and (1,0).
   However, the "diameter" of the set is the max distance between ANY pair.
   If we place other points inside the unit circle, their distance to the pinned points
   is <= 2 (actually <= 2 if on boundary, < 2 if inside).
   Wait, distance from (x,y) inside unit circle to (1,0) is sqrt((x-1)^2+y^2).
   Max possible is 2 (at (-1,0)).
   So if we constrain all 6 other points to the unit disk, the diameter of the set
   is exactly 2 (achieved by the pinned points).
   
   So the problem reduces to:
   Maximize product of all pairwise distances among 8 points,
   where P0=(-1,0), P1=(1,0), and P2..P7 are in the unit disk.
   Since the diameter is fixed at 2, the score is simply the product of distances
   (normalized by 8^8, but we just maximize the product).
   
2. Use a multi-resolution grid search for the 6 free points.
   The search space is (Unit Disk)^6.
   A coarse grid will find the general structure (e.g., symmetric arrangement).
   Then, refine using Nelder-Mead on the log-product objective.
   
   Why this works better than continuous optimization from random starts:
   - The grid ensures we don't miss the global basin of attraction.
   - The objective (log product of distances) is smooth and convex-like in the interior
     (repulsive forces), so local refinement is very effective once in the right basin.
   - Pinning the diameter removes the non-smooth max() constraint, making the optimization
     landscape much simpler (only inequality constraints for disk boundary, which can be
     handled by projecting or penalizing, but Nelder-Mead can handle it if we start inside).
"""

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    n = 8
    # Fixed points
    p_fixed = np.array([[-1.0, 0.0], [1.0, 0.0]])
    
    # We need to place 6 points in the unit disk.
    # Objective: Maximize sum_{i<j} log(||p_i - p_j||)
    # Let's define the objective function for the 6 free points.
    
    def log_product_obj(free_pts):
        """
        free_pts: (6, 2) array of points in unit disk.
        Returns: -log_product (to minimize)
        """
        all_pts = np.vstack([p_fixed, free_pts])
        log_prod = 0.0
        # Compute all pairwise distances
        for i in range(8):
            for j in range(i+1, 8):
                diff = all_pts[i] - all_pts[j]
                dist_sq = np.dot(diff, diff)
                if dist_sq < 1e-12:
                    return 1e9 # Penalty for collision
                log_prod += 0.5 * np.log(dist_sq) # log(dist) = 0.5 log(dist^2)
        
        # Check if any point is outside unit disk (shouldn't happen if we project, 
        # but Nelder-Mead might step out. We add a penalty).
        penalty = 0.0
        for p in free_pts:
            r_sq = np.dot(p, p)
            if r_sq > 1.0:
                # Soft penalty for being outside
                penalty += (r_sq - 1.0) * 100.0
        
        return -log_prod + penalty

    def refine(free_pts_init):
        """
        Run Nelder-Mead on the 6 free points.
        """
        x0 = free_pts_init.flatten()
        
        # Nelder-Mead options
        options = {
            'maxiter': 500,
            'xatol': 1e-6,
            'fatol': 1e-6,
            'disp': False
        }
        
        try:
            res = minimize(log_product_obj, x0, method='Nelder-Mead', options=options)
            if res.success or res.fun < 1e8:
                return res.x.reshape(6, 2)
        except Exception:
            pass
        return free_pts_init

    # Grid Search Strategy
    # The unit disk has area pi.
    # We can't do a full 12D grid.
    # However, by symmetry, we expect the solution to be symmetric with respect to the x-axis
    # (since the pinned points are on the x-axis).
    # Let's assume the 6 points come in 3 symmetric pairs: (x, y), (x, -y), (u, v), (u, -v), (a, b), (a, -b).
    # This reduces the search space from 12D to 6D (3 pairs of (x,y) coordinates).
    # We only need to search x in [-1, 1], y in [0, sqrt(1-x^2)].
    
    # Let's try a coarse grid on the 6D symmetric space.
    # x1, x2, x3 in [-1, 1]
    # y1, y2, y3 in [0, 1] (constrained by x^2+y^2 <= 1)
    
    # To be safe and fast, let's do a multi-stage approach:
    # Stage 1: Coarse grid on symmetric pairs.
    # Stage 2: Refine best candidates.
    # Stage 3: Try asymmetric perturbations if time permits.
    
    best_score = -1e9
    best_free_pts = np.zeros((6, 2))
    
    # Define grid points for x and y
    # x: -1 to 1, 5 points: -1, -0.5, 0, 0.5, 1
    # y: 0 to 1, 5 points: 0, 0.25, 0.5, 0.75, 1.0
    # But we must ensure x^2 + y^2 <= 1.
    
    x_vals = [-1.0, -0.6, 0.0, 0.6, 1.0]
    y_vals = [0.0, 0.3, 0.6, 0.9]
    
    candidates = []
    
    # Generate symmetric candidates
    # 3 pairs: (x1, y1), (x2, y2), (x3, y3)
    # We iterate over combinations. 5*4 * 5*4 * 5*4 = 20 * 20 * 20 = 8000 candidates.
    # This is small enough to evaluate quickly.
    
    for x1 in x_vals:
        for y1 in y_vals:
            if x1*x1 + y1*y1 > 1.0: continue
            for x2 in x_vals:
                for y2 in y_vals:
                    if x2*x2 + y2*y2 > 1.0: continue
                    for x3 in x_vals:
                        for y3 in y_vals:
                            if x3*x3 + y3*y3 > 1.0: continue
                            
                            # Construct the 6 points
                            # Pair 1: (x1, y1), (x1, -y1)
                            # Pair 2: (x2, y2), (x2, -y2)
                            # Pair 3: (x3, y3), (x3, -y3)
                            # Note: If y=0, the pair is identical, which is bad (collision).
                            # We should avoid y=0 unless x is distinct enough, but collisions are penalized.
                            # Actually, if y=0, we have two points at (x,0). Distance 0.
                            # So we should avoid y=0 in the grid or handle it.
                            # Let's shift y_vals to avoid 0 exactly for the grid, or just let the penalty handle it.
                            # Better: use small epsilon for y if it's 0? No, just skip if y is very small and x is same as another.
                            # For now, let the objective penalize it.
                            
                            pts = np.array([
                                [x1, y1], [x1, -y1],
                                [x2, y2], [x2, -y2],
                                [x3, y3], [x3, -y3]
                            ])
                            candidates.append(pts)
    
    # Also add some random asymmetric candidates to break symmetry if needed
    np.random.seed(42)
    for _ in range(50):
        pts = []
        for _ in range(6):
            r = np.random.uniform(0, 1)
            theta = np.random.uniform(0, 2*np.pi)
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            pts.append([x, y])
        candidates.append(np.array(pts))
    
    # Evaluate candidates
    # To save time, we will only refine the top K candidates from the grid.
    # First, compute a quick score for all candidates.
    
    def quick_score(free_pts):
        # Use the objective function
        val = log_product_obj(free_pts)
        return -val # We want to maximize, so negate the minimization objective
    
    scored_candidates = []
    for pts in candidates:
        s = quick_score(pts)
        scored_candidates.append((s, pts))
        
    # Sort by score descending
    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    
    # Refine top 20 candidates
    top_k = 20
    for s, pts in scored_candidates[:top_k]:
        if time.time() > deadline - 10:
            break
            
        refined_pts = refine(pts)
        s_ref = quick_score(refined_pts)
        
        if s_ref > best_score:
            best_score = s_ref
            best_free_pts = refined_pts
            
    # If we have time, try a few more random restarts with refinement
    time_left = deadline - time.time()
    if time_left > 5:
        # Do 10 more random restarts
        for _ in range(10):
            if time.time() > deadline - 2:
                break
            pts = []
            for _ in range(6):
                r = np.random.uniform(0, 1)
                theta = np.random.uniform(0, 2*np.pi)
                x = r * np.cos(theta)
                y = r * np.sin(theta)
                pts.append([x, y])
            pts = np.array(pts)
            refined_pts = refine(pts)
            s_ref = quick_score(refined_pts)
            if s_ref > best_score:
                best_score = s_ref
                best_free_pts = refined_pts

    # Construct final points
    all_pts = np.vstack([p_fixed, best_free_pts])
    
    # Verify diameter is 2
    # The pinned points are at distance 2.
    # All other points are in unit disk, so their distance to pinned points is <= 2.
    # Distance between any two free points is <= 2 (diameter of unit disk).
    # So diameter is exactly 2.
    
    # Scale to diameter 2 (it already is, but just in case of numerical drift)
    d = 0
    for i in range(8):
        for j in range(i+1, 8):
            dist = np.linalg.norm(all_pts[i] - all_pts[j])
            if dist > d:
                d = dist
    if d > 1e-6:
        all_pts = all_pts * (2.0 / d)
        
    return [[float(x), float(y)] for x, y in all_pts]
# EVOLVE-BLOCK-END
