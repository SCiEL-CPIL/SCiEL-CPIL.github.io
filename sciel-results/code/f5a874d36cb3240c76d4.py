import numpy as np
from scipy.spatial import KDTree
import time

N = 200

def construct():
        start_time = time.time()
        timeout = 140  # 10s margin
        
        # 1. Initialization: Hexagonal lattice
        # Approximate hex packing
        # Number of rows/cols
        # Area per circle ~ 0.0366^2 * pi? No, area fraction ~ 0.9069 for hex.
        # Total area 1. 200 circles.
        # r ~ 0.0366. Diameter ~ 0.0732.
        # Hex grid spacing: dx = 2r, dy = 2r * sqrt(3)/2 = r*sqrt(3).
        
        # Let's just generate a grid and jitter it.
        m = int(np.ceil(np.sqrt(N)))
        # m=14 for 196, m=15 for 225.
        # Use 15x15 grid, take 200.
        # Or better, generate hex grid points.
        
        # Simple robust init: Random points + repulsion?
        # Or Hex grid is better.
        
        # Generate hex grid
        points = []
        dx = 0.07 # approx diameter
        dy = dx * np.sqrt(3) / 2
        # Center the grid
        # We need to fit 200 points.
        # 15x14 = 210.
        # Let's try to generate a dense hex grid and pick the best 200?
        # Or just use a square grid and let the optimizer fix it.
        
        # Square grid init is safer to code quickly.
        m = 15
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        # Shuffle and take N
        np.random.seed(42)
        idx = np.random.choice(len(grid), N, replace=False)
        centers = grid[idx].copy()
        
        # Optimization
        # We will use a simple gradient ascent on the radius r.
        # r is determined by min dist / 2 and wall dist.
        
        # To speed up, we'll use a penalty function minimization with L-BFGS-B.
        # Objective: Minimize -r + Penalty(overlaps) + Penalty(walls)
        
        def objective(params):
            # params: N*2 coords + 1 radius r
            coords = params[:N*2].reshape(N, 2)
            r = params[N*2]
            
            # Wall constraints
            # x in [-0.5+r, 0.5-r]
            # Penalty if violated
            wall_pen = 0
            # x
            lo_x = -0.5 + r
            hi_x = 0.5 - r
            viol_x_lo = np.clip(lo_x - coords[:, 0], 0, None)
            viol_x_hi = np.clip(coords[:, 0] - hi_x, 0, None)
            wall_pen += np.sum(viol_x_lo**2) + np.sum(viol_x_hi**2)
            
            lo_y = -0.5 + r
            hi_y = 0.5 - r
            viol_y_lo = np.clip(lo_y - coords[:, 1], 0, None)
            viol_y_hi = np.clip(coords[:, 1] - hi_y, 0, None)
            wall_pen += np.sum(viol_y_lo**2) + np.sum(viol_y_hi**2)
            
            # Pairwise constraints
            # This is O(N^2). For N=200, 40000 pairs.
            # Vectorized distance calculation.
            diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=2))
            # Mask diagonal
            np.fill_diagonal(dists, np.inf)
            
            # Overlap if dist < 2r
            viol_pairs = np.clip(2*r - dists, 0, None)
            pair_pen = np.sum(viol_pairs**2)
            
            return -r + 1000 * (wall_pen + pair_pen)
            
        # This objective is non-smooth at boundaries. L-BFGS-B might struggle.
        # Let's use a smooth approximation: soft-min.
        # Or just use the penalty method carefully.
        
        # Better: Use scipy.optimize.minimize with SLSQP? Too slow.
        # Let's stick to L-BFGS-B with a smooth penalty.
        # Replace max(0, x) with log-sum-exp or similar?
        # Actually, for packing, a common trick is to minimize the sum of exp(-k * dist) or similar?
        
        # Let's try a different approach:
        # Directly optimize the coordinates to maximize the minimum distance.
        # This is a max-min problem.
        
        # I will implement a simple iterative refinement:
        # 1. Calculate nearest neighbors.
        # 2. Move points away from their nearest neighbors and walls.
        
        def refine(centers, r):
            for step in range(1000):
                if time.time() - start_time > timeout:
                    break
                # Compute distances
                # Use KDTree for efficiency
                tree = KDTree(centers)
                # Query nearest neighbors
                # We want to push points apart.
                # Gradient of min dist is sum of gradients of active constraints.
                
                # Simple force-based method:
                # Force on i from j: direction (i-j) / dist^2 * strength
                # Force from wall: push inwards.
                
                forces = np.zeros_like(centers)
                
                # Wall forces
                # If x < -0.5 + r, push right.
                # If x > 0.5 - r, push left.
                margin = r
                # x
                forces[:, 0] += np.where(centers[:, 0] < -0.5 + margin, 1, 0)
                forces[:, 0] -= np.where(centers[:, 0] > 0.5 - margin, 1, 0)
                # y
                forces[:, 1] += np.where(centers[:, 1] < -0.5 + margin, 1, 0)
                forces[:, 1] -= np.where(centers[:, 1] > 0.5 - margin, 1, 0)
                
                # Pair forces
                # Only consider close pairs
                # For each point, find neighbors within 2r + eps
                eps = 0.01
                for i in range(N):
                    # Query points within distance
                    # KDTree.query_ball_point is good
                    neighbors = tree.query_ball_point(centers[i], 2*r + eps)
                    for j in neighbors:
                        if i == j: continue
                        # Avoid double counting? Force is symmetric.
                        # We add force to i from j.
                        d = centers[i] - centers[j]
                        dist = np.linalg.norm(d)
                        if dist < 1e-6:
                            # Random nudge
                            d = np.random.randn(2)
                            dist = np.linalg.norm(d)
                        # Force magnitude: stronger if closer
                        # F ~ 1/dist or (2r - dist)
                        if dist < 2*r:
                            # Overlap, strong repulsion
                            mag = (2*r - dist) / dist
                        else:
                            # Just close, weak repulsion to pack tightly
                            mag = 0.1 * (2*r - dist + eps) / dist if dist < 2*r + eps else 0
                            
                        forces[i] += d * mag
                        # Don't update j here to avoid double loop complexity, 
                        # or do symmetric update? 
                        # For simplicity, just accumulate on i.
                        # This is not symmetric but works as a gradient step if done for all i.
                        
                # Update centers
                lr = 0.01
                centers += lr * forces
                
                # Clip to valid range? No, let forces handle it.
                # But ensure they don't fly away.
                # Re-calculate r
                # Find min dist
                # This is O(N^2) if done naively.
                # Use KDTree to find min dist
                # tree.query(centers, k=2) gives nearest neighbor distance
                # Min over all i.
                dists, _ = tree.query(centers, k=2)
                min_pair_dist = np.min(dists[:, 1])
                
                # Wall dist
                wall_dists = np.min([
                    np.abs(centers[:, 0] + 0.5),
                    np.abs(0.5 - centers[:, 0]),
                    np.abs(centers[:, 1] + 0.5),
                    np.abs(0.5 - centers[:, 1])
                ], axis=0)
                min_wall_dist = np.min(wall_dists)
                
                new_r = 0.5 * min(min_pair_dist, min_wall_dist)
                # Update r slowly to avoid oscillation?
                # Or just use new_r for next step's forces.
                r = new_r
                
            return centers, r

        # Initial r
        # Estimate from grid
        # Grid spacing 1/15 = 0.066. r ~ 0.033.
        r_init = 0.033
        centers, r_final = refine(centers, r_init)
        
        # Final polish: 
        # Calculate exact r
        tree = KDTree(centers)
        dists, _ = tree.query(centers, k=2)
        min_pair_dist = np.min(dists[:, 1])
        wall_dists = np.min([
            np.abs(centers[:, 0] + 0.5),
            np.abs(0.5 - centers[:, 0]),
            np.abs(centers[:, 1] + 0.5),
            np.abs(0.5 - centers[:, 1])
        ], axis=0)
        min_wall_dist = np.min(wall_dists)
        
        # Ensure validity: if r is too large, scale down?
        # The score is min(...). If we return centers, the evaluator calculates r.
        # We just need to return valid centers.
        # The force method should keep them valid if tuned well.
        # But to be safe, we can check and adjust.
        
        # If any constraint is violated, we might need to shrink r or move points.
        # But the evaluator calculates r from centers.
        # So we just return centers.
        
        return centers

    # Run
    # Wait, the force method might be slow in Python loops.
    # N=200, 1000 steps.
    # Inner loop: for i in 200, query_ball_point, for j in neighbors.
    # Neighbors are few.
    # Should be okay.
