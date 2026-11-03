"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time


def construct():
    """
    Construct 300 equal circle centers inside the unit disk to maximize the minimum
    common radius. Uses vectorized repulsive force dynamics (Hertzian potential) 
    with local refinement.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    target_r = 0.0531542991469

    def _hex_init(n):
        """Generate a hexagonal lattice packing of n points in the unit disk."""
        # Estimate spacing
        s = np.sqrt(2 * np.pi / (n * np.sqrt(3))) * 0.95
        points = []
        max_coord = int(1.0 / s) + 2
        for j in range(-max_coord, max_coord + 1):
            for i in range(-max_coord, max_coord + 1):
                x = (i + 0.5 * (j % 2)) * s
                y = j * s * np.sqrt(3) / 2
                if x * x + y * y <= (1.0 - s / 2) ** 2:
                    points.append([x, y])
        points = np.array(points)
        if len(points) < n:
            s *= 0.9
            points = []
            max_coord = int(1.0 / s) + 2
            for j in range(-max_coord, max_coord + 1):
                for i in range(-max_coord, max_coord + 1):
                    x = (i + 0.5 * (j % 2)) * s
                    y = j * s * np.sqrt(3) / 2
                    if x * x + y * y <= (1.0 - s / 2) ** 2:
                        points.append([x, y])
            points = np.array(points)
        if len(points) >= n:
            dists = np.linalg.norm(points, axis=1)
            idx = np.argsort(dists)[:n]
            return points[idx]
        else:
            angles = np.random.uniform(0, 2 * np.pi, n)
            radii = np.sqrt(np.random.uniform(0, 1, n))
            return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])

    def _compute_radius(pts):
        """Compute the exact minimum common radius."""
        n = len(pts)
        # Vectorized distance calculation
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        
        norms = np.linalg.norm(pts, axis=1)
        min_boundary = np.min(1.0 - norms)
        
        return min(min_pair_dist / 2.0, min_boundary)

    def _force_optimize(pts, time_budget, repulsion_strength=1.0, center_attraction=0.1):
        """
        Optimize using repulsive forces between all pairs and attraction to center/boundary.
        """
        n = len(pts)
        best_pts = pts.copy()
        best_score = _compute_radius(pts)
        current_pts = pts.copy()
        
        step_size = 0.01
        no_improve_count = 0
        max_no_improve = 100
        
        # Pre-compute indices for upper triangle to avoid double counting forces?
        # Actually, Newton's 3rd law: F_ij = -F_ji.
        # We can compute full matrix of forces.
        
        while time.time() < time_budget:
            # Compute pairwise distances and forces
            diff = current_pts[:, np.newaxis, :] - current_pts[np.newaxis, :, :]
            dists_sq = np.sum(diff * diff, axis=2)
            dists = np.sqrt(dists_sq + 1e-12)
            np.fill_diagonal(dists, np.inf)
            np.fill_diagonal(dists_sq, np.inf)
            
            # Repulsive force: F = k / d^2 * direction
            # direction = diff / d
            # F = k * diff / d^3
            # To avoid division by zero issues, we use dists_sq * dists = d^3
            # Mask diagonal
            mask = dists > 1e-10
            forces = np.zeros_like(current_pts)
            
            # Vectorized force calculation
            # inv_d3 = 1 / d^3
            inv_d3 = np.zeros_like(dists)
            inv_d3[mask] = 1.0 / (dists[mask] ** 3)
            
            # Force on i from j is repulsion: k * (c_i - c_j) / d^3
            # Sum over j
            # forces[i] = sum_j k * diff[i,j] * inv_d3[i,j]
            # diff[i,j] = c_i - c_j
            # This is equivalent to:
            # forces = k * (diff * inv_d3[:, :, np.newaxis]).sum(axis=1)
            
            forces = repulsion_strength * (diff * inv_d3[:, :, np.newaxis]).sum(axis=1)
            
            # Add center attraction to keep points in and prevent drift
            # F_center = -center_attraction * c_i
            forces -= center_attraction * current_pts
            
            # Normalize forces to control step size
            force_norms = np.linalg.norm(forces, axis=1)
            max_force = np.max(force_norms)
            if max_force > 1e-12:
                forces = forces / max_force
            
            # Update positions
            new_pts = current_pts + step_size * forces
            
            # Project to unit disk
            norms = np.linalg.norm(new_pts, axis=1)
            mask_out = norms > 1.0
            if np.any(mask_out):
                new_pts[mask_out] = new_pts[mask_out] / norms[mask_out] * 0.999
            
            # Check score
            new_score = _compute_radius(new_pts)
            
            if new_score > best_score:
                best_score = new_score
                best_pts = new_pts.copy()
                no_improve_count = 0
                # Adapt step size: increase slightly if improving
                step_size = min(step_size * 1.05, 0.05)
            else:
                no_improve_count += 1
                # Reduce step size
                step_size *= 0.95
                if no_improve_count > max_no_improve:
                    if step_size < 0.0001:
                        break
                    no_improve_count = 0
                    # Try a random restart or perturbation if stuck?
                    # For now, just reduce step further
                    step_size *= 0.5
                
            current_pts = new_pts
            
        return best_pts, best_score

    def _tighten_min_constraint(pts, time_budget):
        """
        Final polish: iteratively push the most constrained points apart.
        """
        n = len(pts)
        best_pts = pts.copy()
        best_score = _compute_radius(pts)
        
        # Use a smaller step size for precision
        step_size = 0.001
        
        while time.time() < time_budget:
            score = _compute_radius(best_pts)
            
            # Find the binding constraint
            dist_boundary = 1.0 - np.linalg.norm(best_pts, axis=1)
            diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff * diff, axis=2) + 1e-30)
            np.fill_diagonal(dists, np.inf)
            min_pair_dist = np.min(dists)
            min_boundary = np.min(dist_boundary)
            
            if min_pair_dist / 2.0 < min_boundary:
                # Pairwise constraint is binding
                i, j = np.unravel_index(np.argmin(dists), dists.shape)
                d = dists[i, j]
                if d < 1e-10:
                    continue
                direction = (best_pts[i] - best_pts[j]) / d
                push = step_size
                new_i = best_pts[i] + direction * push
                new_j = best_pts[j] - direction * push
                
                # Check if this move improves the score
                # Temporarily apply
                old_i = best_pts[i].copy()
                old_j = best_pts[j].copy()
                best_pts[i] = new_i
                best_pts[j] = new_j
                
                # Project to disk
                norm_i = np.linalg.norm(best_pts[i])
                if norm_i > 1.0:
                    best_pts[i] = best_pts[i] / norm_i * 0.9995
                norm_j = np.linalg.norm(best_pts[j])
                if norm_j > 1.0:
                    best_pts[j] = best_pts[j] / norm_j * 0.9995
                    
                new_score = _compute_radius(best_pts)
                
                if new_score > best_score:
                    best_score = new_score
                else:
                    # Revert
                    best_pts[i] = old_i
                    best_pts[j] = old_j
                    step_size *= 0.9
                    if step_size < 1e-6:
                        break
            else:
                # Boundary constraint is binding
                i = np.argmin(dist_boundary)
                norm_i = np.linalg.norm(best_pts[i])
                if norm_i < 1e-10:
                    continue
                direction = -best_pts[i] / norm_i
                push = step_size
                new_i = best_pts[i] + direction * push
                
                old_i = best_pts[i].copy()
                best_pts[i] = new_i
                
                norm_new = np.linalg.norm(best_pts[i])
                if norm_new > 1.0:
                    best_pts[i] = best_pts[i] / norm_new * 0.9995
                    
                new_score = _compute_radius(best_pts)
                
                if new_score > best_score:
                    best_score = new_score
                else:
                    best_pts[i] = old_i
                    step_size *= 0.9
                    if step_size < 1e-6:
                        break
            
        return best_pts, best_score

    # Main: multi-start with force refinement
    best_pts = None
    best_score = 0.0
    
    # Start with hex init
    pts = _hex_init(N)
    norms = np.linalg.norm(pts, axis=1)
    mask = norms > 1.0
    if np.any(mask):
        pts[mask] = pts[mask] / norms[mask] * 0.999
    
    # First refinement pass: Force optimization
    time_budget1 = deadline - 40
    pts, score = _force_optimize(pts, time_budget1, repulsion_strength=1.0, center_attraction=0.1)
    best_pts = pts
    best_score = score
    
    # Second pass with perturbation
    if time.time() < deadline - 30 and best_score < target_r:
        pts2 = best_pts + np.random.normal(0, 0.005, size=(N, 2))
        norms2 = np.linalg.norm(pts2, axis=1)
        mask2 = norms2 > 1.0
        if np.any(mask2):
            pts2[mask2] = pts2[mask2] / norms2[mask2] * 0.999
        
        time_budget2 = deadline - 20
        pts2, score2 = _force_optimize(pts2, time_budget2, repulsion_strength=1.5, center_attraction=0.05)
        if score2 > best_score:
            best_pts = pts2
            best_score = score2
    
    # Third pass: fine polish
    if time.time() < deadline - 10 and best_score < target_r:
        time_budget3 = deadline - 5
        best_pts, best_score = _tighten_min_constraint(best_pts, time_budget3)
    
    # Final validation
    if best_pts is None:
        best_pts = _hex_init(N)
    
    norms = np.linalg.norm(best_pts, axis=1)
    mask = norms > 1.0
    if np.any(mask):
        best_pts[mask] = best_pts[mask] / norms[mask] * 0.99
    
    return best_pts
# EVOLVE-BLOCK-END
