"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import cKDTree

def _score(centers):
    """Compute the radius score: min of half min-distance and min boundary distance."""
    if centers.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Boundary constraint: centers must be inside [-0.5, 0.5]^2
    boundary_margin = 0.5 - np.max(np.abs(centers), axis=1)
    min_boundary = np.min(boundary_margin)
    if min_boundary <= 0:
        return 0.0
    # Pairwise distances
    # Use efficient computation
    dists = np.sqrt(np.sum((centers[:, np.newaxis, :] - centers[np.newaxis, :, :])**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    r = min(min_dist / 2.0, min_boundary)
    return r

def _initial_hex_grid(n):
    """Generate a high-quality hexagonal grid initialization."""
    # For 200 points, a 14x14 grid is 196, so we need 15 rows/cols roughly.
    # Hex packing is denser.
    # Let's try to generate a hex lattice that fits in [-0.5, 0.5]^2.
    
    # Optimal hex packing density is ~0.9069.
    # Area per point ~ 1/200 = 0.005.
    # Hex cell area for radius r is 2*sqrt(3)*r^2.
    # So 2*sqrt(3)*r^2 ~ 0.005 => r^2 ~ 0.005/3.464 ~ 0.00144 => r ~ 0.038.
    # This is close to the target 0.0366.
    
    # We will generate a hex lattice and then scale/shift it to fit.
    # Start with a rough estimate of spacing.
    r_est = 0.038
    dx = 2 * r_est
    dy = dx * np.sqrt(3) / 2
    
    # Estimate number of rows/cols
    # Width 1.0, Height 1.0
    # cols ~ 1.0 / dx
    # rows ~ 1.0 / dy
    
    # Let's just generate a large grid and pick the best subset or scale it.
    # A simpler approach: Generate a square grid and let the relaxation do the work.
    # But hex is a better start.
    
    # Generate hex points in a large area, then scale to fit [-0.5, 0.5]^2.
    # Let's try to fit exactly N points.
    
    # Try different row counts to find a good aspect ratio
    best_points = None
    best_score = 0
    
    for n_rows in range(12, 18):
        n_cols = int(np.ceil(n / n_rows))
        if n_rows * n_cols < n:
            continue
            
        points = []
        for row in range(n_rows):
            y = row * dy
            offset = 0 if row % 2 == 0 else dx / 2
            for col in range(n_cols):
                x = col * dx + offset
                points.append((x, y))
        
        points = np.array(points[:n])
        
        # Center the points
        center = points.mean(axis=0)
        points -= center
        
        # Scale to fit in [-0.5, 0.5]^2 with some margin
        # We want the max extent to be < 0.5
        # Find the bounding box
        max_extent = np.max(np.abs(points))
        if max_extent > 0:
            scale = 0.45 / max_extent  # Leave 5% margin
            points *= scale
            
        # Check if all inside
        if np.all(np.abs(points) <= 0.5 - 1e-10):
            score = _score(points)
            if score > best_score:
                best_score = score
                best_points = points.copy()
                
    if best_points is not None:
        return best_points
        
    # Fallback: Square grid
    m = int(np.ceil(np.sqrt(n)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:n]

def _relax(centers, max_iters=2000, deadline=None):
    """Force-directed relaxation using Lennard-Jones-like repulsion."""
    centers = centers.copy()
    current_score = _score(centers)
    best_centers = centers.copy()
    best_score = current_score
    
    # Parameters
    r0 = 0.038  # Equilibrium distance, slightly larger than current best
    k_repel = 1.0
    k_bound = 10.0
    
    # We use a simple gradient ascent on the objective:
    # Maximize min( min_ij |ci-cj|/2, min_i boundary_i )
    # This is non-smooth. Instead, we use a soft objective:
    # Sum over pairs of max(0, r0 - |ci-cj|)^2  (penalize overlaps)
    # Plus boundary penalties.
    
    # However, to maximize the MINIMUM distance, we need to push all points apart.
    # A common trick is to simulate repulsive forces.
    
    # Let's use a "push apart" algorithm which is more robust for hard constraints.
    # But first, let's try a gradient-based approach on a smooth proxy.
    
    # Proxy Objective:
    # F(x) = - sum_{i<j} max(0, d_target - |x_i - x_j|)^2 - sum_i max(0, d_bound - margin_i)^2
    # We want to minimize F.
    
    d_target = 0.037  # Slightly above current best to push for improvement
    d_bound = 0.5
    
    step_size = 0.001
    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
            
        # Compute forces
        forces = np.zeros_like(centers)
        
        # Pairwise repulsion
        # For efficiency, we can use KDTree to find neighbors within a cutoff
        # But N=200 is small enough for O(N^2) if we vectorize carefully.
        # Let's stick to vectorized O(N^2) for simplicity and reliability.
        
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        # Only consider pairs closer than d_target
        # mask = dists < d_target
        # But we want to push ALL points apart to increase the global minimum.
        # A "soft sphere" model: Force = k * (d_target - dist) if dist < d_target else 0
        # This only pushes points that are too close. Points far apart don't move.
        # This can lead to local minima where some points are far and some are close.
        
        # Better: Use a global pressure.
        # Force on i from j: direction * (d_target - dist) / dist
        # Let's use a smooth potential: V(r) = 0.5 * k * max(0, d_target - r)^2
        # Force = -dV/dr * r_hat = k * max(0, d_target - r) * r_hat
        
        overlap = np.maximum(0, d_target - dists)
        # Force magnitude
        force_mag = k_repel * overlap
        # Direction
        # diff is vector from j to i
        # dists is distance
        # Avoid division by zero
        safe_dists = np.where(dists > 1e-10, dists, 1.0)
        dir_vec = diff / safe_dists[:, :, np.newaxis]
        
        # Force on i is sum over j of force_mag[i,j] * dir_vec[i,j]
        # Note: diff[i,j] is c_i - c_j. So force is in direction of c_i - c_j.
        # This pushes i away from j.
        forces += np.sum(force_mag[:, :, np.newaxis] * dir_vec, axis=1)
        
        # Boundary forces
        # If center is too close to boundary, push inward.
        # Boundary is at +/- 0.5.
        # Margin = 0.5 - |c_k|
        # If margin < d_bound_margin (which is just the radius limit, but here we treat it as a wall),
        # actually the constraint is c_i must be in [-0.5+r, 0.5-r].
        # But we don't know r yet. We are optimizing for r.
        # Let's assume a target radius r_target = d_target / 2.
        r_target = d_target / 2
        inner_bound = 0.5 - r_target
        
        # For each coordinate k:
        # If c_i,k > inner_bound, push left (negative force)
        # If c_i,k < -inner_bound, push right (positive force)
        
        for k in range(2):
            # Upper bound violation
            viol_upper = np.maximum(0, centers[:, k] - inner_bound)
            forces[:, k] -= k_bound * viol_upper
            
            # Lower bound violation
            viol_lower = np.maximum(0, -inner_bound - centers[:, k])
            forces[:, k] += k_bound * viol_lower
            
        # Update positions
        # Use simple gradient descent on the potential (which is gradient ascent on distance)
        # centers += step_size * forces
        
        # Adaptive step size
        # If score improves, keep step. If not, reduce.
        new_centers = centers + step_size * forces
        
        # Clamp to square to prevent explosion
        new_centers = np.clip(new_centers, -0.5, 0.5)
        
        new_score = _score(new_centers)
        
        if new_score > current_score:
            current_score = new_score
            centers = new_centers
            if new_score > best_score:
                best_score = new_score
                best_centers = centers.copy()
            # Increase step size slightly if improving?
            # step_size *= 1.05
        else:
            # Reduce step size
            step_size *= 0.9
            # If step size is too small, break
            if step_size < 1e-6:
                break
                
    return best_centers, best_score

def _push_apart(centers, max_iters=500, deadline=None):
    """Iteratively push the closest pair apart."""
    centers = centers.copy()
    current_score = _score(centers)
    best_centers = centers.copy()
    best_score = current_score
    
    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
            
        # Find closest pair
        tree = cKDTree(centers)
        dists, idxs = tree.query(centers, k=2)
        # dists[:, 1] is distance to nearest neighbor
        # idxs[:, 1] is index of nearest neighbor
        
        min_dist_idx = np.argmin(dists[:, 1])
        i = min_dist_idx
        j = idxs[i, 1]
        
        d = dists[i, 1]
        
        # Also check boundary
        boundary_margin = 0.5 - np.max(np.abs(centers), axis=1)
        min_boundary_idx = np.argmin(boundary_margin)
        min_boundary = boundary_margin[min_boundary_idx]
        
        # Determine limiting factor
        r_pair = d / 2.0
        r_bound = min_boundary
        
        if r_pair < r_bound:
            # Push i and j apart
            diff = centers[i] - centers[j]
            norm = np.linalg.norm(diff)
            if norm < 1e-10:
                # Random kick
                centers[i] += np.random.normal(0, 0.001, 2)
                centers[j] -= np.random.normal(0, 0.001, 2)
            else:
                diff_unit = diff / norm
                # Move each by half the gap needed?
                # Just move a small step
                step = 0.001
                centers[i] += diff_unit * step
                centers[j] -= diff_unit * step
        else:
            # Push boundary-constrained point inward
            idx = min_boundary_idx
            for k in range(2):
                if abs(centers[idx, k]) > 0.5 - 1e-10:
                    centers[idx, k] -= np.sign(centers[idx, k]) * 0.001
                    
        # Clamp
        centers = np.clip(centers, -0.5, 0.5)
        
        new_score = _score(centers)
        if new_score > current_score:
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_centers = centers.copy()
        else:
            # If no improvement, maybe we are stuck.
            # Try a small random perturbation of the closest pair
            if np.random.random() < 0.1:
                centers[i] += np.random.normal(0, 0.005, 2)
                centers[j] += np.random.normal(0, 0.005, 2)
                centers = np.clip(centers, -0.5, 0.5)
                new_score = _score(centers)
                if new_score > current_score:
                    current_score = new_score
                    if new_score > best_score:
                        best_score = new_score
                        best_centers = centers.copy()
                else:
                    # Revert
                    pass # We already modified centers, need to be careful.
                    # Let's just break if stuck for a long time?
                    # For now, continue.
                    
    return best_centers, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin
    
    # 1. Initialize with hex grid
    best_centers = _initial_hex_grid(N)
    best_score = _score(best_centers)
    
    # 2. Force-directed relaxation
    if time.time() < deadline - 60:
        centers, score = _relax(best_centers, max_iters=1000, deadline=deadline - 60)
        if score > best_score:
            best_score = score
            best_centers = centers
    
    # 3. Push-apart refinement
    if time.time() < deadline - 20:
        centers, score = _push_apart(best_centers, max_iters=500, deadline=deadline - 20)
        if score > best_score:
            best_score = score
            best_centers = centers
            
    # 4. Final relaxation pass
    if time.time() < deadline - 10:
        centers, score = _relax(best_centers, max_iters=500, deadline=deadline - 10)
        if score > best_score:
            best_score = score
            best_centers = centers
            
    return best_centers
# EVOLVE-BLOCK-END
