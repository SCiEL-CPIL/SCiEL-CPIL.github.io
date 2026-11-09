"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    t_start = time.time()
    deadline = 145.0  # seconds, with safety margin before 150s kill
    
    def compute_radius(pts):
        """Compute the maximum radius for given centers."""
        if len(pts) < 2:
            return 0.5
        # Pairwise distances
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists)
        
        # Boundary distances
        b_left = pts[:, 0] + 0.5
        b_right = 0.5 - pts[:, 0]
        b_bottom = pts[:, 1] + 0.5
        b_top = 0.5 - pts[:, 1]
        min_bound = np.min([np.min(b_left), np.min(b_right), np.min(b_bottom), np.min(b_top)])
        
        return min(min_pair / 2.0, min_bound)

    def generate_hex_grid(n_points):
        """
        Generate a hexagonal grid of points that roughly fills the unit square.
        We generate a large grid and then select the best subset or center it.
        """
        # Estimate grid size. For N=200, sqrt(200) ~ 14.
        # Hex packing density is ~0.9069. Square packing is 1.0.
        # We want to fit N points in [-0.5, 0.5]^2.
        # Let's try a grid that is slightly larger than needed and center it.
        
        # Approximate spacing for N points in area 1:
        # Area per point ~ 1/N.
        # Hex area per point ~ s^2 * sqrt(3)/2.
        # s ~ sqrt(2 / (N * sqrt(3)))
        s_est = np.sqrt(2.0 / (N * np.sqrt(3.0)))
        
        # Generate a grid of points with spacing s_est
        # We need enough points to cover the square.
        # Square side is 1.
        # Rows needed ~ 1 / (s_est * sqrt(3)/2)
        # Cols needed ~ 1 / s_est
        
        n_rows = int(np.ceil(1.0 / (s_est * np.sqrt(3.0) / 2.0))) + 2
        n_cols = int(np.ceil(1.0 / s_est)) + 2
        
        points = []
        dx = s_est
        dy = s_est * np.sqrt(3.0)
        
        # Start from a bit outside to ensure coverage, then filter/center
        # Or generate centered around 0.
        
        # Let's generate a grid centered at (0,0)
        for row in range(-n_rows//2, n_rows//2 + 1):
            y = row * dy
            offset = 0 if row % 2 == 0 else dx / 2
            for col in range(-n_cols//2, n_cols//2 + 1):
                x = col * dx + offset
                points.append([x, y])
                
        points = np.array(points)
        
        # Filter points that are roughly inside the square [-0.5, 0.5]
        # But we want exactly N points.
        # Let's select the N points closest to the center? Or just the first N?
        # Better: Select points that are inside [-0.5, 0.5] and if we have more than N, 
        # we might need to adjust.
        
        # Simpler approach: Just create a grid that has exactly N points by adjusting dimensions?
        # No, hex grids don't have simple N.
        
        # Let's use a different strategy:
        # 1. Create a dense hex grid.
        # 2. Keep only those inside [-0.5, 0.5].
        # 3. If we have more than N, randomly remove some? Or take the first N?
        # 4. If we have fewer, expand the grid.
        
        # Let's try to fit N points by scaling.
        # We want the bounding box of the N points to be roughly 1x1.
        
        # Alternative: Use a square grid as fallback if hex is messy.
        # But the directive asks for hex.
        
        # Let's generate a hex grid with spacing s such that we get approx N points.
        # Then center it.
        
        # Re-evaluating s:
        # If we have a hex grid with spacing s, the density is 1 / (s^2 * sqrt(3)/2).
        # We want N points in area 1.
        # s = sqrt(2 / (N * sqrt(3)))
        
        # Let's generate a grid of size (n_rows, n_cols) that yields approx N points.
        # n_rows * n_cols ~ N
        # Let n_cols = int(np.sqrt(N * 2 / np.sqrt(3)))
        # Let n_rows = int(np.ceil(N / n_cols))
        
        n_cols = int(np.sqrt(N * 2.0 / np.sqrt(3.0)))
        n_rows = int(np.ceil(N / float(n_cols)))
        
        points = []
        dx = 1.0 / n_cols # Approximate horizontal spacing to fit width 1
        # But hex rows are offset.
        # To fit N points, we can just generate a grid and take the first N.
        
        # Let's generate a grid with spacing 1/sqrt(N) approx.
        s = 1.0 / np.sqrt(N)
        
        # Generate points
        for row in range(n_rows):
            y = -0.5 + s/2 + row * s * np.sqrt(3)
            if y > 0.5: break
            offset = 0 if row % 2 == 0 else s / 2
            for col in range(n_cols + 1):
                x = -0.5 + s/2 + offset + col * s
                if x > 0.5: break
                points.append([x, y])
                
        points = np.array(points)
        
        if len(points) < N:
            # Expand grid
            s = s * 0.9
            points = []
            n_cols = int(np.sqrt(N * 2.0 / np.sqrt(3.0)))
            n_rows = int(np.ceil(N / float(n_cols)))
            for row in range(n_rows + 2):
                y = -0.5 + s/2 + row * s * np.sqrt(3)
                if y > 0.5: break
                offset = 0 if row % 2 == 0 else s / 2
                for col in range(n_cols + 2):
                    x = -0.5 + s/2 + offset + col * s
                    if x > 0.5: break
                    points.append([x, y])
            points = np.array(points)
            
        if len(points) > N:
            # Randomly select N points? Or take the first N?
            # Taking first N might be biased to top-left.
            # Let's shuffle and take N.
            rng = np.random.RandomState(42)
            idx = rng.choice(len(points), N, replace=False)
            points = points[idx]
        elif len(points) < N:
            # Pad with random points?
            # This shouldn't happen if we expand enough.
            # Let's just fill remaining with random points in the square.
            rng = np.random.RandomState(42)
            missing = N - len(points)
            random_pts = rng.uniform(-0.4, 0.4, size=(missing, 2))
            points = np.vstack([points, random_pts])
            
        return points

    def relax_packing(pts, max_iter=1000, lr_init=0.01, lr_decay=0.999):
        """
        Force-based relaxation.
        Repulsion between all pairs.
        Attraction to keep points inside? No, just clip.
        """
        pts = pts.copy()
        lr = lr_init
        best_r = compute_radius(pts)
        best_pts = pts.copy()
        
        for it in range(max_iter):
            if time.time() - t_start > deadline - 5:
                break
            
            # Compute pairwise differences and distances
            diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diffs**2, axis=2))
            np.fill_diagonal(dists, np.inf)
            
            # Force: Repulsion if dist < threshold.
            # Threshold should be related to current min distance or target.
            # Let's use a fixed threshold based on initial estimate or current min.
            # Using current min distance as threshold makes it adaptive.
            min_d = np.min(dists)
            if min_d < 1e-6:
                min_d = 1e-6
            
            threshold = min_d * 2.0
            
            # Force magnitude: (threshold - dist) / dist^2 (inverse square repulsion)
            # Or simply (threshold - dist) for linear repulsion.
            # Linear is more stable.
            
            mask = dists < threshold
            np.fill_diagonal(mask, False)
            
            forces = np.zeros_like(pts)
            
            # Only compute forces for pairs within threshold to save time?
            # For N=200, 200x200 matrix is 40k elements. Full computation is fast enough.
            
            # Force vector from j to i is (p_i - p_j) * magnitude
            # magnitude = max(0, threshold - dist)
            
            mags = np.where(mask, threshold - dists, 0.0)
            # Normalize by distance to get direction? 
            # diffs is already p_i - p_j.
            # If we use mags * diffs, the force is proportional to (p_i - p_j) * (threshold - d).
            # This is a standard spring-like repulsion.
            
            forces = np.sum(mags[:, :, np.newaxis] * diffs, axis=1)
            
            # Boundary forces: Push back if too close to boundary
            # Target boundary distance: r_est = min_d / 2
            # But we don't know r yet. Let's just push away from walls if dist_to_wall < min_d/2
            wall_thresh = min_d / 2.0
            
            b_forces = np.zeros_like(pts)
            
            # Left
            d_left = pts[:, 0] + 0.5
            mask_l = d_left < wall_thresh
            b_forces[mask_l, 0] += (wall_thresh - d_left[mask_l])
            
            # Right
            d_right = 0.5 - pts[:, 0]
            mask_r = d_right < wall_thresh
            b_forces[mask_r, 0] -= (wall_thresh - d_right[mask_r])
            
            # Bottom
            d_bot = pts[:, 1] + 0.5
            mask_b = d_bot < wall_thresh
            b_forces[mask_b, 1] += (wall_thresh - d_bot[mask_b])
            
            # Top
            d_top = 0.5 - pts[:, 1]
            mask_t = d_top < wall_thresh
            b_forces[mask_t, 1] -= (wall_thresh - d_top[mask_t])
            
            # Update positions
            pts += lr * (forces + b_forces)
            
            # Clip to square
            pts[:, 0] = np.clip(pts[:, 0], -0.5, 0.5)
            pts[:, 1] = np.clip(pts[:, 1], -0.5, 0.5)
            
            lr *= lr_decay
            
            # Check improvement
            current_r = compute_radius(pts)
            if current_r > best_r:
                best_r = current_r
                best_pts = pts.copy()
        
        return best_pts, best_r

    # Main logic
    # 1. Generate initial hex grid
    init_pts = generate_hex_grid(N)
    
    # 2. Relax
    optimized_pts, score = relax_packing(init_pts, max_iter=2000, lr_init=0.005, lr_decay=0.9995)
    
    # 3. If time permits, try a second pass with different init or finer tuning
    if time.time() - t_start < deadline - 30:
        # Try a slightly perturbed version
        rng = np.random.RandomState(43)
        perturbed = optimized_pts + rng.normal(0, 0.001, size=(N, 2))
        perturbed[:, 0] = np.clip(perturbed[:, 0], -0.5, 0.5)
        perturbed[:, 1] = np.clip(perturbed[:, 1], -0.5, 0.5)
        
        opt2_pts, score2 = relax_packing(perturbed, max_iter=1000, lr_init=0.002, lr_decay=0.9999)
        
        if score2 > score:
            optimized_pts = opt2_pts
            score = score2

    # Ensure validity: Clip to inside
    optimized_pts = np.clip(optimized_pts, -0.5 + 1e-9, 0.5 - 1e-9)
    
    return optimized_pts

# EVOLVE-BLOCK-END
