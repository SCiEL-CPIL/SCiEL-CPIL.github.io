"""Improved packing: hexagonal lattice init + vectorized force relaxation."""
import numpy as np
import time

N = 300
DEADLINE = 145.0  # seconds, safety margin before 150s kill


def construct():
    start_time = time.time()
    
    # Generate hexagonal lattice points inside unit disk
    def hex_init():
        # Try various spacings to get at least N points
        best_points = None
        for trial in range(50):
            # Start with a reasonable spacing
            # For N=300 in unit disk, spacing around 2/sqrt(N) ~ 0.115
            # But hex packing is denser, so we can use slightly larger spacing
            base_spacing = 2.0 / (np.sqrt(N) + 1.0)
            spacing = base_spacing * (1.0 + 0.01 * trial)
            
            if spacing < 0.01:
                break
            
            # Generate hexagonal lattice points
            dx = spacing
            dy = spacing * np.sqrt(3.0) / 2.0
            
            # Number of points needed in each direction
            # Unit disk radius = 1, so we need to cover [-1, 1] in x and y
            k = int(1.0 / min(dx, dy)) + 3
            
            points = []
            for j in range(-k, k + 1):
                y = j * dy
                if abs(y) > 1.0:
                    continue
                # Offset for hexagonal arrangement
                x_offset = (0.5 * (j % 2)) * dx
                for i in range(-k, k + 1):
                    x = (i + x_offset) * dx
                    r = np.sqrt(x*x + y*y)
                    if r <= 1.0:
                        points.append([x, y])
            
            points = np.array(points)
            if len(points) == 0:
                continue
            
            # Compute radius (distance to boundary)
            radii = 1.0 - np.sqrt(points[:, 0]**2 + points[:, 1]**2)
            
            # Sort by radius (keep points closest to center first? or farthest?)
            # Actually, we want to keep points that are well inside
            # Let's keep all and select N with best properties
            
            if len(points) >= N:
                # Select N points - prefer those with larger minimum distance to others
                # For now, just take first N sorted by distance from center (innermost)
                dist_from_center = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
                indices = np.argsort(dist_from_center)[:N]
                points = points[indices]
                
                if best_points is None or len(points) > len(best_points):
                    best_points = points
            
            # Check time
            if time.time() - start_time > 10:
                break
        
        if best_points is not None:
            return best_points
        else:
            # Fallback: simple grid
            spacing = 2.0 / (np.sqrt(N) + 1.0)
            k = int(1.0 / spacing) + 2
            points = np.array([(i * spacing, j * spacing) 
                              for i in range(-k, k+1) 
                              for j in range(-k, k+1)])
            radii = 1.0 - np.sqrt(points[:, 0]**2 + points[:, 1]**2)
            indices = np.argsort(np.sqrt(points[:, 0]**2 + points[:, 1]**2))[:N]
            return points[indices]
    
    points = hex_init()
    
    # Force-directed relaxation
    def relax(points, max_time=130.0):
        """Vectorized force-directed relaxation to maximize minimum distance."""
        n = len(points)
        points = points.astype(np.float64).copy()
        
        # Initial step size
        step = 0.01
        min_dist_target = 0.05  # rough estimate
        
        while time.time() - start_time < max_time:
            # Compute pairwise distances
            diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
            dist = np.sqrt(np.sum(diff**2, axis=2))
            np.fill_diagonal(dist, np.inf)
            
            # Find minimum distance
            min_dist = np.min(dist)
            
            # If minimum distance is good enough, we can stop or reduce step
            if min_dist > 0.053:
                step *= 0.95
            
            # Compute forces: repulsion between all pairs, attraction to keep in disk
            forces = np.zeros_like(points)
            
            # Repulsive forces: for each pair, push apart if too close
            # Use a soft repulsion: F = k / d^2 * direction
            # Vectorized: for each pair i,j, force on i from j
            # Only consider pairs with dist < some threshold to save computation
            threshold = 0.2
            mask = (dist < threshold) & (dist > 1e-10)
            
            if np.any(mask):
                # Force magnitude: k / dist^2
                k_rep = 0.01
                force_mag = k_rep / (dist[mask] ** 2)
                
                # Get the pairs
                i_idx, j_idx = np.where(mask)
                
                # Direction from j to i
                dir_vec = diff[i_idx, j_idx] / dist[i_idx, j_idx, np.newaxis]
                
                # Apply forces
                for idx, jdx in zip(i_idx, j_idx):
                    forces[idx] += force_mag[idx] * dir_vec[idx]
                    forces[jdx] -= force_mag[idx] * dir_vec[idx]
            
            # Boundary force: push points inward if they're near the boundary
            r = np.sqrt(np.sum(points**2, axis=1))
            boundary_margin = 0.01
            near_boundary = r > (1.0 - boundary_margin)
            if np.any(near_boundary):
                # Force proportional to how far outside the margin
                excess = (r[near_boundary] - (1.0 - boundary_margin)) / boundary_margin
                # Direction: toward center
                center_dir = -points[near_boundary] / r[near_boundary, np.newaxis]
                forces[near_boundary] += 0.1 * excess[:, np.newaxis] * center_dir
            
            # Update positions with adaptive step
            points += step * forces
            
            # Project back into unit disk
            r = np.sqrt(np.sum(points**2, axis=1))
            outside = r > 1.0
            if np.any(outside):
                points[outside] = points[outside] / r[outside, np.newaxis]
            
            # Adaptive step size
            step *= 0.999
            if step < 1e-6:
                break
        
        return points
    
    # Run relaxation
    points = relax(points)
    
    # Final: compute score and return
    # Ensure valid output
    r = np.sqrt(np.sum(points**2, axis=1))
    valid = r <= 1.0
    if not np.all(valid):
        points = points[valid]
        if len(points) < N:
            # This shouldn't happen, but fallback
            points = hex_init()
    
    return points[:N]
