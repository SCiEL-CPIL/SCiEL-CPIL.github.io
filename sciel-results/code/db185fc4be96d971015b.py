"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_radius(centers):
    """
    Computes the maximum common radius for a given set of centers.
    Centers: (N, 3) array.
    Cube: [-0.5, 0.5]^3.
    """
    if centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
        
    # Wall constraints: distance from center to nearest wall
    # min_i min_k (0.5 - |c_ik|)
    abs_centers = np.abs(centers)
    # Distance to wall for each coordinate
    wall_dists = 0.5 - abs_centers
    # The radius is limited by the minimum distance to any wall for any sphere
    min_wall_dist = np.min(wall_dists)
    
    if min_wall_dist < 0:
        return 0.0

    # Pairwise constraints
    # dists[i, j] = |c_i - c_j|
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    sq_dists = np.sum(diff**2, axis=2)
    dists = np.sqrt(sq_dists)
    
    # Set diagonal to infinity to exclude self-distance
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # The radius is half the minimum pairwise distance, limited by wall distance
    r = min(min_pair_dist / 2.0, min_wall_dist)
    return float(r)

def _energy_and_gradient(x, R):
    """
    Calculates the repulsion energy and gradient for a target radius R.
    Energy is the sum of squared violations of the distance constraints.
    Violations are:
    1. Pairwise: |c_i - c_j| < 2R
    2. Wall: 0.5 - |c_ik| < R
    """
    centers = x.reshape(-1, 3)
    n = len(centers)
    
    energy = 0.0
    grad = np.zeros_like(centers)
    
    # 1. Pairwise terms
    # Compute all pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    sq_dists = np.sum(diff**2, axis=2)
    # Avoid division by zero in gradient calculation
    dists = np.sqrt(sq_dists + 1e-15)
    
    # We only care about pairs where distance < 2R
    # Create a mask for upper triangle to avoid double counting
    i_idx, j_idx = np.triu_indices(n, k=1)
    d_ij = dists[i_idx, j_idx]
    
    # Mask for pairs closer than 2R
    mask_pairs = d_ij < 2.0 * R
    
    if np.any(mask_pairs):
        v_ij = 2.0 * R - d_ij[mask_pairs]
        d_valid = d_ij[mask_pairs]
        i_valid = i_idx[mask_pairs]
        j_valid = j_idx[mask_pairs]
        
        # Energy: sum (2R - d)^2
        energy += np.sum(v_ij**2)
        
        # Gradient of 0.5 * (2R - d)^2 w.r.t centers? 
        # Let E = sum (2R - d_ij)^2.
        # dE/dx_i = sum_j 2(2R - d_ij) * (-1) * d(d_ij)/dx_i
        # d(d_ij)/dx_i = (x_i - x_j) / d_ij
        # So dE/dx_i = -2 * sum_j (2R - d_ij) * (x_i - x_j) / d_ij
        # Wait, the standard penalty is usually 0.5 * violation^2 to cancel the 2.
        # If we use sum (2R-d)^2, the gradient is -2(2R-d) * (xi-xj)/d.
        # Let's stick to sum (2R-d)^2 for simplicity, just adjust gradient factor.
        
        factors = 2.0 * v_ij / d_valid
        vecs = centers[i_valid] - centers[j_valid]
        forces = factors[:, np.newaxis] * vecs
        
        # dE/dx_i = -2 * sum (2R-d) * (xi-xj)/d. 
        # My force vector is (2R-d)/d * (xi-xj).
        # So dE/dx_i = -2 * sum force_ij.
        # dE/dx_j = +2 * sum force_ij.
        np.add.at(grad, i_valid, -2.0 * forces)
        np.add.at(grad, j_valid, 2.0 * forces)

    # 2. Wall terms
    abs_centers = np.abs(centers)
    # Distance to nearest wall for each sphere is min_k (0.5 - |c_ik|)
    # But for gradient, we need to know which wall is limiting.
    # wall_dist_i = 0.5 - max_k |c_ik|
    max_abs = np.max(abs_centers, axis=1)
    wall_dists = 0.5 - max_abs
    
    mask_wall = wall_dists < R
    if np.any(mask_wall):
        violations_wall = R - wall_dists[mask_wall]
        i_valid_wall = np.where(mask_wall)[0]
        
        energy += np.sum(violations_wall**2)
        
        for idx in i_valid_wall:
            v = violations_wall[np.where(i_valid_wall == idx)[0][0]]
            # Find which coordinate is limiting (max abs)
            max_k = np.argmax(abs_centers[idx])
            
            # Gradient of (R - (0.5 - |x_k|))^2
            # Let y = |x_k|. d/dx_k (R - 0.5 + y)^2 = 2(R - 0.5 + y) * sign(x_k)
            # Here v = R - (0.5 - y) = R - 0.5 + y.
            grad[idx, max_k] += 2.0 * v * np.sign(centers[idx, max_k])
            
    return energy, grad.flatten()

def _minimize_energy(init_centers, target_R, max_iter=500, ftol=1e-14, gtol=1e-12):
    """
    Minimizes the energy for a given target radius.
    Returns (new_centers, final_energy, success_flag).
    """
    def objective_and_grad(x):
        val, g = _energy_and_gradient(x, target_R)
        return val, g

    bounds = [(-0.5, 0.5)] * (N * 3)
    
    try:
        res = minimize(
            objective_and_grad,
            init_centers.flatten(),
            jac=True,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol, 'disp': False}
        )
        new_centers = res.x.reshape(N, 3)
        final_energy, _ = _energy_and_gradient(new_centers, target_R)
        return new_centers, final_energy, res.success
    except Exception:
        return init_centers, np.inf, False

def _hybrid_init(n):
    """
    Hybrid initialization:
    1. 64 spheres in a 4x4x4 simple cubic lattice.
    2. 36 spheres in interstitial voids (body-centered positions of the 2x2x2 sub-cubes 
       or similar high-density spots).
    
    The 4x4x4 grid has spacing 1/4 = 0.25. 
    Radius of 4x4x4 grid: min pair dist 0.25 -> r=0.125. Wall dist: center to wall 0.5 - 0.125 = 0.375? 
    No, centers are at -0.375, -0.125, 0.125, 0.375. 
    Wall dist = 0.5 - 0.375 = 0.125. 
    So a 4x4x4 grid supports r=0.125.
    
    We need to place 36 more spheres. 
    The 'voids' in a simple cubic lattice are at the centers of the cubes formed by 8 spheres.
    For a 4x4x4 grid, there are 3x3x3 = 27 internal voids. 
    These are at positions where coordinates are midpoints of the grid lines.
    Grid: -0.375, -0.125, 0.125, 0.375.
    Midpoints: -0.25, 0.0, 0.25.
    These 27 points are at distance sqrt(3)*(0.125) from the 8 surrounding grid points.
    0.125 * 1.732 = 0.216. 
    If r=0.125, distance needed is 0.25. 0.216 < 0.25. 
    So we cannot simply drop spheres into these voids at r=0.125. 
    The optimizer will have to push the grid spheres apart or move the void spheres.
    
    To make this a better start, we can jitter the grid slightly or use a slightly smaller 
    initial grid spacing to allow the void spheres to fit easier, but the prompt asks for 
    exploiting high-density structures.
    
    Let's place the 64 grid spheres and 27 void spheres. 
    We need 36 more. We can add 9 more on the surface or use a different pattern for the remaining 9.
    Actually, 4x4x4 = 64. 
    Interstitial sites in a 4x4x4 lattice:
    Internal: 3x3x3 = 27.
    Face centers? 
    Let's just use the 27 internal voids and 9 random/surface points to reach 100.
    """
    # 4x4x4 Grid
    grid_coords = np.array([-0.375, -0.125, 0.125, 0.375])
    grid_points = np.array([(x, y, z) for z in grid_coords for y in grid_coords for x in grid_coords])
    
    # 27 Void Points (Internal)
    void_coords = np.array([-0.25, 0.0, 0.25])
    void_points = np.array([(x, y, z) for z in void_coords for y in void_coords for x in void_coords])
    
    # We have 64 + 27 = 91 points.
    # Need 9 more. 
    # Let's place them on the outer surface centers or corners? 
    # Corners are occupied by grid. 
    # Face centers of the big cube: (0, 0, 0.5) etc? No, wall distance is 0. 
    # Let's just use random points in the remaining space or duplicate void logic with slight offset?
    # Better: Use the 27 voids, and add 9 points in the "edge" voids or just randomize 9 positions 
    # near the surface but inside.
    
    rng = np.random.default_rng(42)
    # 9 points: Let's try to place them in the 12 edge midpoints? 
    # Edge midpoints of the 4x4x4 grid structure?
    # Let's just use random points in the box, but seeded, to break symmetry.
    extra_points = rng.uniform(-0.45, 0.45, size=(9, 3))
    
    all_points = np.vstack([grid_points, void_points, extra_points])
    
    # Add small jitter to break exact symmetry which can cause degenerate gradients
    all_points += rng.normal(0, 0.005, size=all_points.shape)
    
    # Clip to valid range
    all_points = np.clip(all_points, -0.49, 0.49)
    
    return all_points[:N]

def _polish_critical_pairs(centers, time_limit=5.0):
    """Final high-precision polish focusing only on critical (closest) pairs."""
    start_time = time.time()
    centers = centers.copy()
    
    for iteration in range(200):
        if time.time() - start_time > time_limit:
            break
        
        # Find the minimum radius constraint
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        min_pair_dist = np.min(dists)
        min_pair_idx = np.unravel_index(np.argmin(dists), dists.shape)
        
        wall_dists = 0.5 - np.abs(centers)
        min_wall_dist = np.min(wall_dists)
        min_wall_idx = np.unravel_index(np.argmin(wall_dists), wall_dists.shape)
        
        r = min(min_pair_dist / 2.0, min_wall_dist)
        
        # If pair distance is limiting, push those two apart
        if min_pair_dist / 2.0 < min_wall_dist:
            i, j = min_pair_idx
            if i > j:
                i, j = j, i
            d = centers[i] - centers[j]
            dist = np.linalg.norm(d)
            if dist < 1e-10:
                d = np.array([1e-6, 0, 0])
                dist = 1e-6
            # Push apart along the connecting line
            # We want to increase dist to 2*r_target? No, we just want to satisfy current r.
            # Actually, this polish is to fix small violations if any exist or to tighten.
            # Let's just push them apart by a small epsilon if they are too close.
            target_dist = min_pair_dist + 1e-6
            if dist < target_dist:
                push = (target_dist - dist) * d / dist * 0.5
                centers[i] += push
                centers[j] -= push
        else:
            # Wall is limiting, push that sphere inward
            i, k = min_wall_idx
            wall = wall_dists[i, k]
            if wall < 0:
                 # Should not happen if clipped, but just in case
                 sign = np.sign(centers[i, k])
                 if sign == 0:
                     sign = 1.0
                 centers[i, k] += (0.01) * sign
        
        # Re-clip to cube
        centers = np.clip(centers, -0.5, 0.5)
        
    return centers

def construct():
    """Construct the optimal packing of N spheres in the unit cube."""
    start_time = time.time()
    deadline = start_time + 140.0  # 10 second safety margin
    
    best_centers = None
    best_radius = 0.0
    
    # 1. Hybrid Initialization
    init_centers = _hybrid_init(N)
    
    # Initial radius estimate
    r_grid = _compute_radius(init_centers)
    best_radius = r_grid
    best_centers = init_centers.copy()
    
    # 2. Iterative Radius Increase
    # Start from a conservative radius and increase
    current_r = best_radius
    
    # We know 4x4x4 grid supports 0.125. 
    # Let's start slightly below the expected optimum to ensure convergence.
    # Optimum is ~0.111. Wait, 4x4x4 grid r=0.125 is HIGHER than 0.111?
    # Let's re-verify 4x4x4 grid radius.
    # Centers: -0.375, -0.125, 0.125, 0.375.
    # Min pair dist: 0.25. r = 0.125.
    # Min wall dist: 0.5 - 0.375 = 0.125.
    # So a 4x4x4 grid of 64 spheres has r=0.125.
    # Why is the best known 100-sphere packing only 0.111?
    # Because we have 100 spheres, not 64. The extra 36 spheres force the radius down.
    # So starting with a 4x4x4 grid + 36 spheres will have a very low initial radius 
    # due to overlaps in the voids.
    
    # Let's check the radius of our hybrid init.
    # The void spheres are at distance ~0.216 from grid spheres.
    # 2*r = 0.25. 0.216 < 0.25. Overlap.
    # So initial radius is determined by the closest pair, which will be ~0.108.
    
    # Strategy:
    # 1. Compute initial radius.
    # 2. Try to increase radius in small steps.
    
    # If initial radius is very small, we might need to start lower.
    # Let's set a target and iterate.
    
    # To ensure we don't get stuck, we will try to optimize for a sequence of radii.
    # Start with a radius that is definitely achievable (e.g., 0.05) and step up.
    
    # However, the hybrid init is geometrically strong. 
    # Let's just start with current_r = best_radius * 0.9 if best_radius > 0, else 0.05.
    if best_radius < 0.01:
        current_r = 0.05
    else:
        current_r = best_radius * 0.95
    
    step_size = 0.001
    max_failures = 5
    consecutive_failures = 0
    
    rng_search = np.random.default_rng(123)
    
    while time.time() < deadline - 20.0:
        target_r = current_r + step_size
        
        if consecutive_failures >= max_failures:
            step_size *= 0.7
            consecutive_failures = 0
            if step_size < 1e-6:
                break
        
        # Try a few starts to escape local minima
        # 1. Best current
        # 2. Best current + small jitter
        # 3. Best current + larger jitter
        
        starts = [
            best_centers,
            np.clip(best_centers + rng_search.normal(0, 0.005, size=(N, 3)), -0.49, 0.49),
            np.clip(best_centers + rng_search.normal(0, 0.02, size=(N, 3)), -0.49, 0.49)
        ]
        
        success = False
        for start_pos in starts:
            # Determine max iter based on time
            time_remaining = deadline - time.time() - 15.0
            if time_remaining < 2.0:
                break
            
            max_iter = 200
            if current_r > 0.11:
                max_iter = 400 # More precision near optimum
            
            new_pos, energy, opt_success = _minimize_energy(start_pos, target_r, max_iter=max_iter)
            
            # Check if we satisfied the constraints
            if energy < 1e-10:
                actual_r = _compute_radius(new_pos)
                if actual_r >= target_r - 1e-7:
                    best_pos = new_pos
                    best_r = actual_r
                    current_r = actual_r
                    success = True
                    consecutive_failures = 0
                    break
            else:
                # Even if energy is not zero, if we improved radius, take it?
                # No, we must maintain feasibility for the NEW radius to step up.
                # If we can't reach target_r, we might still have improved current_r?
                # _minimize_energy minimizes violations for target_r.
                # If it fails, the resulting centers might have a radius < target_r.
                # We should check if the resulting centers have a radius > current_r.
                actual_r = _compute_radius(new_pos)
                if actual_r > best_radius:
                    best_pos = new_pos
                    best_r = actual_r
                    best_radius = actual_r
                    best_centers = new_pos.copy()
                    # Update current_r to this improved value? 
                    # Be careful: if we can't reach target_r, we shouldn't assume we can reach current_r + step.
                    # But if we improved, we can try a smaller step.
                    current_r = actual_r
                    # Don't count as full success, but reset failures? 
                    # Let's reset failures if we made progress.
                    consecutive_failures = 0
                    # Don't break, try next start? No, if we improved, we can try to go higher.
                    # But we already used up the time for this start.
                    # Let's break and loop again with new current_r.
                    break
        
        if not success:
            consecutive_failures += 1
            # If we failed to increase, we might try a smaller step in the next loop
            # (handled by step_size reduction above)
            
        # Update best_centers if we found a better one in the "else" branch
        # (already done in the loop)
        
        # Safety check
        if time.time() > deadline - 25.0:
            break
    
    # 3. Final Polish
    if time.time() < deadline - 5.0:
        try:
            polished = _polish_critical_pairs(best_centers, time_limit=min(5.0, deadline - time.time() - 2.0))
            r_polished = _compute_radius(polished)
            if r_polished > best_radius:
                best_radius = r_polished
                best_centers = polished.copy()
        except Exception:
            pass
    
    # Final validation and clipping
    if best_centers is None:
        best_centers = _hybrid_init(N)
    
    best_centers = np.clip(best_centers, -0.4999999, 0.4999999)
    
    return best_centers
# EVOLVE-BLOCK-END
