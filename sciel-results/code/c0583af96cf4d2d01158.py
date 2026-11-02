"""
Packing 200 equal circles in a unit square using soft-contact rigid-body dynamics.
"""
import numpy as np

N = 200


def construct():
    # Initial configuration: hexagonal lattice
    def hexagonal_init(n_circles):
        """Generate hexagonal lattice points centered in unit square."""
        best_points = None
        best_r = 0
        
        # Try different numbers of rows and cols to find the densest hexagonal fit
        for rows in range(10, 25):
            for cols in range(10, 25):
                total = rows * cols
                if total < n_circles:
                    continue
                
                # Hexagonal grid dimensions
                # Horizontal spacing: 2r
                # Vertical spacing: r * sqrt(3)
                # Effective width with offset: (cols-1)*2r + r = (2*cols-1)*r
                # Height: (rows-1)*r*sqrt(3)
                
                # Fit in square [-0.5, 0.5]^2
                # (2*cols-1)*r <= 1 => r <= 1/(2*cols-1)
                # (rows-1)*r*sqrt(3) <= 1 => r <= 1/((rows-1)*sqrt(3))
                
                r_max = min(1.0 / (2*cols - 1), 1.0 / ((rows - 1) * np.sqrt(3)))
                
                if r_max > best_r:
                    best_r = r_max
                    pts = []
                    for j in range(rows):
                        for i in range(cols):
                            x = (i * 2 - (cols - 1)) * r_max
                            if j % 2 == 1:
                                x += r_max
                            y = (j - (rows - 1) / 2.0) * r_max * np.sqrt(3)
                            pts.append([x, y])
                    pts = np.array(pts[:n_circles])
                    if np.all(np.abs(pts) <= 0.5 - r_max * 0.01):
                        best_points = pts
                        best_r = r_max
        
        if best_points is not None:
            return best_points
        
        # Fallback: square grid
        m = int(np.ceil(np.sqrt(n_circles)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:n_circles]

    centers = hexagonal_init(N)
    velocities = np.zeros_like(centers)
    
    def compute_force_and_value(centers, t):
        """
        Compute the force (gradient of soft-min clearance) and the soft-min value.
        """
        # Pairwise distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        dists = np.sqrt(dists_sq)
        np.fill_diagonal(dists, np.inf)
        
        # Boundary distances (4 per particle)
        d_left = 0.5 - centers[:, 0]
        d_right = 0.5 + centers[:, 0]
        d_bottom = 0.5 - centers[:, 1]
        d_top = 0.5 + centers[:, 1]
        
        all_bound_dists = np.vstack([d_left, d_right, d_bottom, d_top])
        
        # Extract pairwise distances (upper triangle)
        triu_indices = np.triu_indices(N, k=1)
        pair_dists = dists[triu_indices] / 2.0
        
        # Find global min for stability
        min_pair = np.min(pair_dists)
        min_bound = np.min(all_bound_dists)
        global_min = min(min_pair, min_bound)
        
        # Shifted values
        shifted_pairs = pair_dists - global_min
        shifted_bounds = all_bound_dists - global_min
        
        # Exponentials
        exp_pairs = np.exp(-shifted_pairs / t)
        exp_bounds = np.exp(-shifted_bounds / t)
        
        sum_exp = np.sum(exp_pairs) + np.sum(exp_bounds)
        
        # Soft min value
        soft_min_val = global_min - t * np.log(sum_exp)
        
        # Weights
        w_pairs = exp_pairs / sum_exp
        w_bounds = exp_bounds / sum_exp
        
        # Forces
        forces = np.zeros_like(centers)
        
        # Pairwise forces
        W = np.zeros((N, N))
        W[triu_indices] = w_pairs
        W = W + W.T
        
        inv_dists = np.zeros_like(dists)
        mask = dists < np.inf
        inv_dists[mask] = 1.0 / (2.0 * dists[mask])
        
        weights_full = W * inv_dists
        pair_forces = np.sum(weights_full[:, :, np.newaxis] * diff, axis=1)
        
        # Boundary forces
        # w_bounds shape (4, N)
        # 0: left (grad -1), 1: right (grad 1)
        # 2: bottom (grad -1), 3: top (grad 1)
        bound_forces_x = -w_bounds[0, :] + w_bounds[1, :]
        bound_forces_y = -w_bounds[2, :] + w_bounds[3, :]
        
        forces[:, 0] += pair_forces[:, 0] + bound_forces_x
        forces[:, 1] += pair_forces[:, 1] + bound_forces_y
        
        return forces, soft_min_val

    # Simulation parameters
    num_steps = 5000
    damping = 0.9
    dt = 0.01
    t_start = 0.05
    t_end = 0.001
    
    best_centers = centers.copy()
    best_r = -np.inf
    
    for step in range(num_steps):
        frac = step / num_steps
        t = t_start * (1 - frac) + t_end * frac
        
        forces, val = compute_force_and_value(centers, t)
        
        velocities = velocities * damping + forces * dt
        centers = centers + velocities * dt
        
        # Clip to bounds
        centers = np.clip(centers, -0.5, 0.5)
        
        # Track best
        if step % 100 == 0:
            diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=2))
            np.fill_diagonal(dists, np.inf)
            min_pair = np.min(dists) / 2
            
            bound_dists = np.minimum(
                np.minimum(0.5 - centers[:, 0], 0.5 + centers[:, 0]),
                np.minimum(0.5 - centers[:, 1], 0.5 + centers[:, 1])
            )
            min_bound = np.min(bound_dists)
            
            r = min(min_pair, min_bound)
            
            if r > best_r:
                best_r = r
                best_centers = centers.copy()
        
        # Adaptive dt
        max_force = np.max(np.linalg.norm(forces, axis=1))
        if max_force > 100:
            dt *= 0.5
        elif max_force < 1:
            dt = min(dt * 1.1, 0.05)

    return best_centers


# Quick test
if __name__ == "__main__":
    centers = construct()
    print(f"Shape: {centers.shape}")
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2
    bound_dists = np.minimum(
        np.minimum(0.5 - centers[:, 0], 0.5 + centers[:, 0]),
        np.minimum(0.5 - centers[:, 1], 0.5 + centers[:, 1])
    )
    min_bound = np.min(bound_dists)
    print(f"Radius: {min(min_pair, min_bound):.10f}")
