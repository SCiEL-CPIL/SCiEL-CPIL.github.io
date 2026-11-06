"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

    # 1. Initialize with Hexagonal Lattice
    # Re-use the logic from the seed to get a good starting point
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        points, radius = points[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
        if len(points) >= N:
            points = points[np.argsort(radius, kind="stable")[:N]]
            break
        spacing *= 0.98

    points = points.astype(np.float64)
    
    # Helper to compute the max valid radius for a given set of points
    def get_max_radius(pts):
        # Min distance between pairs
        dists = []
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                dists.append(np.linalg.norm(pts[i] - pts[j]))
        min_pair_dist = min(dists) if dists else 1.0
        # Min distance to boundary
        min_boundary_dist = min(1.0 - np.linalg.norm(p) for p in pts)
        return min(min_pair_dist / 2.0, min_boundary_dist)

    r = get_max_radius(points)
    
    # Optimization Parameters
    delta_r = 0.0005
    min_delta_r = 0.000001
    inner_steps = 50
    learning_rate = 0.001
    
    # Gradient Descent to maximize r
    # We iteratively increase r and relax positions to remove overlaps.
    
    while time.time() < deadline:
        new_r = r + delta_r
        if new_r > 0.0531542991469 + 0.001: # Stop if we significantly beat known best
            break
            
        # Attempt to find a configuration for new_r
        success = False
        current_pts = points.copy()
        
        # Run gradient descent for a fixed number of steps
        for _ in range(inner_steps):
            if time.time() > deadline:
                break
                
            # Compute forces (gradients of the penalty function)
            forces = np.zeros_like(current_pts)
            
            # Pairwise repulsion
            # Vectorized calculation for speed
            # dist_matrix[i,j] = distance between i and j
            diff = current_pts[:, np.newaxis, :] - current_pts[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=2)
            # Avoid division by zero
            dists[dist == 0] = 1e-10
            dists[np.arange(N), np.arange(N)] = 1.0 # Mask self
            
            # Penalty: max(0, 2*r - d)^2
            # Gradient w.r.t points:
            # For each pair i,j with d < 2r:
            # Force on i: 2 * (2r - d) * (p_i - p_j) / d
            # Force on j: 2 * (2r - d) * (p_j - p_i) / d
            
            overlap = np.maximum(0, 2 * new_r - dists)
            # We only want to apply force where overlap > 0
            # Note: This applies force from ALL overlapping pairs.
            
            # Calculate force vector
            # F_i = sum_j 2 * overlap_ij * (p_i - p_j) / dist_ij
            # We can compute this as:
            # F_i = sum_j 2 * overlap_ij * diff_ij / dist_ij
            
            # diff is (N, N, 2). dists is (N, N). overlap is (N, N).
            # We need to zero out the diagonal and non-overlapping parts.
            mask = overlap > 0
            forces_diff = np.zeros_like(diff)
            forces_diff[mask] = 2.0 * overlap[mask, :, :] * diff[mask] / dists[mask, :, :][:, :, np.newaxis]
            
            # Sum over j for each i
            forces += np.sum(forces_diff, axis=1)
            
            # Boundary repulsion
            # Penalty: max(0, r - (1 - |p|))^2
            # Gradient: 2 * (r - (1 - |p|)) * p / |p|
            radii = np.linalg.norm(current_pts, axis=1)
            boundary_overlap = np.maximum(0, new_r - (1.0 - radii))
            boundary_forces = np.zeros_like(current_pts)
            mask_b = boundary_overlap > 0
            # p / |p|
            unit_vectors = current_pts[mask_b] / radii[mask_b, np.newaxis]
            boundary_forces[mask_b] = 2.0 * boundary_overlap[mask_b, np.newaxis] * unit_vectors
            forces += boundary_forces
            
            # Update points
            current_pts += learning_rate * forces
            
            # Optional: Project back inside unit circle if they went out?
            # The force should pull them back, but let's be safe.
            new_radii = np.linalg.norm(current_pts, axis=1)
            if np.any(new_radii > 1.0):
                current_pts[new_radii > 1.0] *= (1.0 / new_radii[new_radii > 1.0])[:, np.newaxis]
                
        # Check if the configuration is valid for new_r
        # Compute actual max radius for current_pts
        achieved_r = get_max_radius(current_pts)
        
        if achieved_r >= new_r - 1e-5: # Small tolerance
            points = current_pts
            r = achieved_r
            delta_r = min(delta_r * 1.05, 0.001) # Increase step size slightly
        else:
            # Failed to reach new_r, reduce step size
            delta_r *= 0.5
            if delta_r < min_delta_r:
                break
            # Keep old points
            
    return points

# EVOLVE-BLOCK-END
