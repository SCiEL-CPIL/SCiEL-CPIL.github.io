import numpy as np
import time
import math
from scipy.spatial import cKDTree

N = 200

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    def compute_radius(centers):
        """Compute the radius efficiently."""
        wall_dists = 0.5 - np.abs(centers)
        min_wall = np.min(wall_dists)
        
        tree = cKDTree(centers)
        dists, _ = tree.query(centers, k=2)
        min_pair = np.min(dists[:, 1])
        
        r = min(min_wall, min_pair / 2.0)
        return max(0.0, r)

    def repulsion_field(centers, k_neighbors=8, softness=0.01):
        """
        Compute a repulsion force field for each center.
        Force on center i is sum over neighbors j of:
          - (c_i - c_j) / (d_ij^2 + softness)  for pair repulsion
          - wall repulsion: push away from nearest wall
        """
        n = len(centers)
        forces = np.zeros_like(centers)
        tree = cKDTree(centers)
        dists, idxs = tree.query(centers, k=k_neighbors + 1)  # +1 for self
        
        for i in range(n):
            for k in range(1, k_neighbors + 1):
                j = idxs[i, k]
                d_vec = centers[i] - centers[j]
                d_sq = np.dot(d_vec, d_vec) + softness
                forces[i] += d_vec / d_sq
            
            # Wall repulsion: push inward if close to boundary
            for dim in range(2):
                if centers[i, dim] > 0:
                    wall_dist = 0.5 - centers[i, dim]
                else:
                    wall_dist = 0.5 + centers[i, dim]
                wall_force = 1.0 / (wall_dist + softness)
                forces[i, dim] += -np.sign(centers[i, dim]) * wall_force
        
        return forces

    def adaptive_local_search(centers, max_iters=500, base_step=0.005, cooling=0.995):
        """
        Local search using repulsion field with adaptive step size.
        Accept moves that improve the exact radius, with occasional
        random perturbations to escape local minima.
        """
        centers = centers.copy()
        current_r = compute_radius(centers)
        best_r = current_r
        best_centers = centers.copy()
        step = base_step
        
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
            
            # Compute repulsion field
            forces = repulsion_field(centers, k_neighbors=8, softness=0.005)
            
            # Normalize forces to unit magnitude for consistent step scaling
            force_norms = np.linalg.norm(forces, axis=1, keepdims=True)
            force_norms[force_norms < 1e-12] = 1.0
            unit_forces = forces / force_norms
            
            # Propose new positions: move along force direction with adaptive step
            # Add small random jitter to escape symmetry
            jitter = np.random.randn(len(centers), 2) * step * 0.3
            proposed = centers + unit_forces * step + jitter
            
            # Clip to bounds
            proposed = np.clip(proposed, -0.4999, 0.4999)
            
            # Evaluate proposed radius
            proposed_r = compute_radius(proposed)
            
            # Accept if improved, or with small probability if not worse (simulated annealing lite)
            if proposed_r > current_r:
                centers = proposed
                current_r = proposed_r
                if current_r > best_r:
                    best_r = current_r
                    best_centers = centers.copy()
                step *= 1.02  # Slightly increase step on success
            else:
                # Occasionally accept non-improving moves to escape local minima
                if np.random.rand() < 0.1:
                    centers = proposed
                    current_r = proposed_r
                step *= cooling  # Cool down on failure
            
            # Clamp step size
            step = np.clip(step, 1e-6, 0.02)
            
            # If no improvement for many iterations, restart with larger perturbation
            if it % 50 == 49 and current_r < best_r - 1e-6:
                # Perturb the best configuration
                centers = best_centers + np.random.randn(N, 2) * 0.01
                centers = np.clip(centers, -0.4999, 0.4999)
                current_r = compute_radius(centers)
                step = base_step
        
        return best_centers, best_r

    def hexagonal_initialization(n_rows, n_cols):
        """Create hexagonal packing layout."""
        d = 1.0 / (n_cols + 0.5)
        row_spacing = d * math.sqrt(3) / 2
        
        total_height = (n_rows - 1) * row_spacing + d
        if total_height > 1.0 + 1e-6:
            return None
        
        pts = []
        for row in range(n_rows):
            y = -0.5 + d/2 + row * row_spacing
            if row % 2 == 0:
                n_in_row = n_cols
                x_start = -0.5 + d/2
            else:
                n_in_row = n_cols - 1
                x_start = -0.5 + d
            
            for col in range(n_in_row):
                x = x_start + col * d
                pts.append([x, y])
        
        pts = np.array(pts)
        if len(pts) >= N:
            pts = pts[:N]
        elif len(pts) < N:
            return None
        
        return pts

    def square_grid_init():
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]

    # Generate diverse initial configurations
    best_centers = None
    best_radius = 0.0

    # Try multiple hexagonal layouts
    for n_rows in range(13, 17):
        n_cols = math.ceil(N / n_rows)
        pts = hexagonal_initialization(n_rows, n_cols)
        if pts is not None:
            r = compute_radius(pts)
            if r > best_radius:
                best_radius = r
                best_centers = pts.copy()

    # Also try square grid
    grid = square_grid_init()
    r_grid = compute_radius(grid)
    if r_grid > best_radius:
        best_radius = r_grid
        best_centers = grid.copy()

    # Run adaptive local search on best initial configuration
    if best_centers is not None and time.time() < deadline - 20:
        optimized, r_opt = adaptive_local_search(best_centers, max_iters=300, base_step=0.004, cooling=0.993)
        if r_opt > best_radius:
            best_radius = r_opt
            best_centers = optimized.copy()

    # Second pass with different parameters for refinement
    if best_centers is not None and time.time() < deadline - 10:
        refined, r_refined = adaptive_local_search(best_centers, max_iters=200, base_step=0.002, cooling=0.997)
        if r_refined > best_radius:
            best_radius = r_refined
            best_centers = refined.copy()

    # Final polish: very small steps for precision
    if best_centers is not None and time.time() < deadline - 3:
        polished, r_polished = adaptive_local_search(best_centers, max_iters=100, base_step=0.001, cooling=0.999)
        if r_polished > best_radius:
            best_radius = r_polished
            best_centers = polished.copy()

    if best_centers is None:
        best_centers = square_grid_init()

    return best_centers
