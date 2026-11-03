import numpy as np
import time

N = 300

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s buffer for overhead/return
    
    # 1. Initialization: Hexagonal Lattice
    # Estimate spacing. Area per circle ~ pi * r^2. Total area pi.
    # r ~ 1/sqrt(N) * constant.
    # For N=300, r ~ 0.05. Spacing ~ 2*r ~ 0.1.
    # Let's generate a grid and filter.
    
    def generate_hex_lattice(spacing):
        points = []
        # Range needs to cover [-1, 1]
        # spacing is distance between centers
        # hex grid:
        # x = i * spacing
        # y = j * spacing * sqrt(3)/2 + (i % 2) * spacing * sqrt(3)/4
        # Actually, standard hex:
        # x = (i + 0.5 * (j % 2)) * spacing
        # y = j * spacing * sqrt(3) / 2
        
        max_idx = int(2.0 / spacing) + 2
        for i in range(-max_idx, max_idx + 1):
            for j in range(-max_idx, max_idx + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if x*x + y*y <= 1.0:
                    points.append((x, y))
        return np.array(points)
    
    # Find a good initial spacing
    # We want at least N points.
    # Try spacing 0.1
    spacing = 0.1
    while True:
        pts = generate_hex_lattice(spacing)
        if len(pts) >= N:
            # Select N points. We can select those closest to origin to keep them central,
            # or just take the first N. Taking closest to origin is better for packing.
            dists = np.hypot(pts[:, 0], pts[:, 1])
            indices = np.argsort(dists)[:N]
            centers = pts[indices].copy()
            break
        spacing *= 0.95
        
    # 2. Optimization Loop
    # Current best radius
    def calc_radius(c):
        # min distance between pairs
        d = np.linalg.norm(c[:, np.newaxis, :] - c[np.newaxis, :, :], axis=2)
        np.fill_diagonal(d, np.inf)
        min_dist = np.min(d)
        # min distance to boundary
        r_bound = 1.0 - np.max(np.linalg.norm(c, axis=1))
        return min(min_dist / 2.0, r_bound)
    
    r_current = calc_radius(centers)
    best_r = r_current
    best_centers = centers.copy()
    
    # Force directed relaxation
    # Parameters
    step_size = 0.01
    num_iterations = 0
    
    while time.time() < deadline:
        num_iterations += 1
        
        # Calculate pairwise distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, np.inf)
        
        # Identify pairs closer than 2 * r_current
        # We want to push them apart.
        # Force F_i = sum_j (diff_ij / dist_ij^2) * max(0, 2*r - dist_ij)
        # Note: diff_ij is vector from j to i. So pushing i away from j is adding diff_ij.
        
        # Mask for close pairs
        close_mask = dist < 2.0 * r_current
        # Avoid division by zero (diagonal is inf, so mask is False there)
        
        # Calculate forces
        # We want to sum over j for each i.
        # diff is (N, N, 2). dist is (N, N).
        # We need to weight diff by (2*r - dist) / dist^2 for close pairs.
        
        # To avoid zero division, we can use a small epsilon or rely on mask.
        # Since diagonal is inf, close_mask is False there.
        
        weight = np.zeros_like(dist)
        weight[close_mask] = (2.0 * r_current - dist[close_mask]) / (dist[close_mask] ** 2 + 1e-12)
        
        forces = np.sum(weight[:, :, np.newaxis] * diff, axis=1)
        
        # Boundary force
        # Push points inside if they are too close to boundary.
        # r_bound_i = 1 - |c_i|
        # If r_bound_i < r_current, push towards origin.
        # Force = -c_i / |c_i| * (|c_i| - (1 - r_current))
        
        norms = np.linalg.norm(centers, axis=1)
        # Avoid div by zero
        norms_safe = np.where(norms > 1e-12, norms, 1.0)
        
        # Violation amount: how much it exceeds the allowed radius (1 - r_current)
        allowed_r = 1.0 - r_current
        violation = np.maximum(0.0, norms - allowed_r)
        
        # Direction: towards origin (-c_i / |c_i|)
        boundary_forces = -centers / norms_safe[:, np.newaxis] * violation[:, np.newaxis]
        
        total_forces = forces + boundary_forces
        
        # Update centers
        # Adaptive step size?
        # If we are making progress (r increasing), keep step size.
        # If not, maybe reduce it.
        
        centers = centers + step_size * total_forces
        
        # Check for improvement
        # To save time, we don't need to calculate exact r every single iteration,
        # but it's cheap enough (N=300, 90k ops).
        r_new = calc_radius(centers)
        
        if r_new > r_current:
            r_current = r_new
            if r_current > best_r:
                best_r = r_current
                best_centers = centers.copy()
            # Increase step size slightly? Or keep it.
            # step_size = min(step_size * 1.01, 0.05) # Optional
        else:
            # If no improvement, maybe the step was too big or we hit a local min.
            # Reduce step size to allow finer adjustment.
            step_size *= 0.99
            
    return best_centers

# EVOLVE-BLOCK-START
# The code above is the full program.
# EVOLVE-BLOCK-END
