"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def construct():
    """Place 200 equal circles in a unit square to maximize the minimum radius."""
    t_start = time.time()
    deadline = t_start + 110  # Safety margin
    
    # --- Initialization ---
    # Use a hexagonal grid initialization which is denser than square grid
    def hex_init(n):
        # Approximate hex packing
        # Area per point ~ s^2 * sqrt(3)/2
        # N points in area 1. s ~ sqrt(2 / (N * sqrt(3)))
        s = np.sqrt(2.0 / (n * np.sqrt(3)))
        pts = []
        y = -0.5 + s / 2
        row = 0
        while y < 0.5 and len(pts) < n:
            x_offset = (row % 2) * s / 2
            x = -0.5 + s / 2 + x_offset
            while x < 0.5 and len(pts) < n:
                # Check if inside bounds (strictly)
                if -0.5 < x < 0.5 and -0.5 < y < 0.5:
                    pts.append([x, y])
                x += s
            y += s * np.sqrt(3) / 2
            row += 1
        
        pts = np.array(pts)
        if len(pts) < n:
            # If we ran out of points in the hex grid (shouldn't happen with N=200 usually, 
            # but just in case), fill with random or grid
            # Actually, hex grid usually has enough points. 
            # Let's just truncate or pad.
            if len(pts) > 0:
                # Pad with random points near the last one? No, just random.
                remaining = n - len(pts)
                rand_pts = np.random.rand(remaining, 2) - 0.5
                pts = np.vstack([pts, rand_pts])
            else:
                pts = np.random.rand(n, 2) - 0.5
        return pts[:n]

    pts = hex_init(N).copy()
    
    # Helper functions
    def get_min_radius(p):
        """Calculate the max possible radius for the current configuration."""
        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        
        # Wall distances
        wall_dists = np.minimum(
            np.minimum(0.5 - p[:, 0], p[:, 0] + 0.5),
            np.minimum(0.5 - p[:, 1], p[:, 1] + 0.5)
        )
        min_wall_dist = np.min(wall_dists)
        
        return min(min_pair_dist, min_wall_dist) / 2

    def calc_dists(p):
        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        d = np.sqrt(np.sum(diff**2, axis=2))
        return d

    def calc_wall_dists(p):
        return np.minimum(
            np.minimum(0.5 - p[:, 0], p[:, 0] + 0.5),
            np.minimum(0.5 - p[:, 1], p[:, 1] + 0.5)
        )

    # Initial state
    dist_matrix = calc_dists(pts)
    wall_dists = calc_wall_dists(pts)
    current_r = get_min_radius(pts)
    best_r = current_r
    best_pts = pts.copy()

    # Mask for off-diagonal elements
    eye_mask = ~np.eye(N, dtype=bool)

    # --- Optimization Loop ---
    # Strategy: Hybrid of Simulated Annealing and Gradient-based local search.
    # 1. Try a gradient step (move all points away from nearest neighbors/walls).
    # 2. If that fails or stagnates, try a random single-point move (SA style).
    
    T = 0.01  # Temperature / Step size for random moves
    cooling = 0.9995
    min_T = 1e-7
    grad_step_size = 0.005
    no_improve_count = 0
    max_no_improve = 5000

    steps = 0
    while time.time() < deadline - 5:
        if T < min_T:
            # Restart with higher T if stuck
            if steps > 10000:
                break
            T = 0.01
            # Add small noise to escape local minima
            pts += np.random.randn(N, 2) * 0.005
            pts = np.clip(pts, -0.49, 0.49)
            dist_matrix = calc_dists(pts)
            wall_dists = calc_wall_dists(pts)
            current_r = get_min_radius(pts)
            if current_r > best_r:
                best_r = current_r
                best_pts = pts.copy()
            no_improve_count = 0

        # --- Phase 1: Gradient Step (Vectorized) ---
        # Objective: Maximize min distance. 
        # Approximation: Move points away from neighbors that are too close.
        # Threshold: current_r * 2 + small_epsilon. If dist < threshold, repel.
        threshold = max(current_r * 2.0, 0.01)
        
        # Calculate forces
        # W[i,j] = 1/d_ij if d_ij < threshold else 0
        active = (dist_matrix < threshold) & (dist_matrix > 1e-9)
        np.fill_diagonal(active, False)
        
        W = np.zeros_like(dist_matrix)
        np.divide(1.0, dist_matrix, out=W, where=active)
        
        row_sums = W.sum(axis=1)
        # grad[i] = row_sums[i] * pts[i] - sum_j W[i,j] * pts[j]
        grad_pairs = row_sums[:, np.newaxis] * pts - W.dot(pts)
        
        # Wall forces
        grad_walls = np.zeros_like(pts)
        x = pts[:, 0]
        y = pts[:, 1]
        
        d_right = 0.5 - x
        d_left = x + 0.5
        d_top = 0.5 - y
        d_bottom = y + 0.5
        
        # Find closest wall for each point
        # We can vectorize this
        min_wall = wall_dists
        # Determine which wall is min
        # d_right is min if d_right <= d_left AND d_right <= d_top AND d_right <= d_bottom
        # This is tricky to vectorize perfectly for ties, but let's try simple comparisons
        
        # Force x
        # If d_right is min -> push left (-1)
        # If d_left is min -> push right (+1)
        
        # Let's just loop for wall forces, it's O(N) and N=200 is small.
        for i in range(N):
            if wall_dists[i] < threshold:
                d_r = 0.5 - pts[i, 0]
                d_l = pts[i, 0] + 0.5
                d_t = 0.5 - pts[i, 1]
                d_b = pts[i, 1] + 0.5
                
                min_d = min(d_r, d_l, d_t, d_b)
                w = 1.0 / max(min_d, 1e-9)
                
                if d_r == min_d:
                    grad_walls[i, 0] -= w
                if d_l == min_d:
                    grad_walls[i, 0] += w
                if d_t == min_d:
                    grad_walls[i, 1] -= w
                if d_b == min_d:
                    grad_walls[i, 1] += w
                    
        grad = grad_pairs + grad_walls
        
        # Propose new position
        new_pts = pts + grad_step_size * grad
        new_pts = np.clip(new_pts, -0.49, 0.49)
        
        # Evaluate new radius
        new_dists = calc_dists(new_pts)
        new_walls = calc_wall_dists(new_pts)
        new_r = min(np.min(new_dists[eye_mask]), np.min(new_walls)) / 2
        
        if new_r > current_r:
            pts = new_pts
            dist_matrix = new_dists
            wall_dists = new_walls
            current_r = new_r
            no_improve_count = 0
            if current_r > best_r:
                best_r = current_r
                best_pts = pts.copy()
        else:
            no_improve_count += 1
            # If gradient step failed, try a random single-point move
            # This helps escape local minima where gradient is zero or pointing wrong way
            i = np.random.randint(N)
            delta = np.random.randn(2) * T
            old_pos = pts[i].copy()
            new_pos = old_pos + delta
            
            if -0.5 <= new_pos[0] <= 0.5 and -0.5 <= new_pos[1] <= 0.5:
                # Update dist matrix row/col
                new_dists_i = np.sqrt(np.sum((pts - new_pos)**2, axis=1))
                new_wall_i = np.minimum(
                    np.minimum(0.5 - new_pos[0], new_pos[0] + 0.5),
                    np.minimum(0.5 - new_pos[1], new_pos[1] + 0.5)
                )
                
                # Temporarily update
                old_dist_row = dist_matrix[i, :].copy()
                old_wall_i = wall_dists[i]
                
                dist_matrix[i, :] = new_dists_i
                dist_matrix[:, i] = new_dists_i
                dist_matrix[i, i] = 0
                wall_dists[i] = new_wall_i
                
                # Calculate new global min
                min_pair = np.min(dist_matrix[eye_mask])
                min_wall = np.min(wall_dists)
                candidate_r = min(min_pair, min_wall) / 2
                
                if candidate_r > current_r:
                    pts[i] = new_pos
                    current_r = candidate_r
                    no_improve_count = 0
                    if current_r > best_r:
                        best_r = current_r
                        best_pts = pts.copy()
                else:
                    # Revert
                    pts[i] = old_pos
                    dist_matrix[i, :] = old_dist_row
                    dist_matrix[:, i] = old_dist_row
                    dist_matrix[i, i] = 0
                    wall_dists[i] = old_wall_i
            
            T *= cooling

        steps += 1

    # Final check
    final_r = get_min_radius(best_pts)
    if final_r > best_r:
        best_r = final_r
        best_pts = best_pts.copy() # Just in case
    
    return best_pts

# EVOLVE-BLOCK-END
