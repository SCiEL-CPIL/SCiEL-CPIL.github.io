"""
Optimized packer for 100 equal spheres in a unit cube.
Combines constraint-satisfaction BFGS with incremental radius squeezing.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 100
HALF = 0.5

def _compute_radius(centers):
    """Compute the maximum common radius for given centers."""
    n = len(centers)
    if n == 0:
        return 0.0
    
    # Minimum distance to boundary
    min_boundary = HALF - np.abs(centers).max()
    if min_boundary <= 0:
        return 0.0
    
    # Minimum pairwise distance / 2
    if n < 2:
        return min_boundary
    
    # Compute all pairwise distances
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs ** 2, axis=-1))
    # Set diagonal to inf to ignore self-distances
    np.fill_diagonal(dists, np.inf)
    min_pair = dists.min() / 2.0
    
    return min(min_boundary, min_pair)

def _energy_and_grad(x, target_r):
    """
    Energy function for BFGS optimization.
    x is a flat array of length 3N.
    target_r is the desired radius.
    Returns (energy, gradient).
    Energy is sum of squared violations of the constraints:
    1. Pairwise distance >= 2 * target_r
    2. Wall distance >= target_r
    """
    c = x.reshape(N, 3)
    
    # --- Pairwise Part ---
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    # Avoid division by zero on diagonal
    np.fill_diagonal(dists, 1.0)
    
    # Active pairs: those closer than 2 * target_r
    active = dists < 2.0 * target_r
    # Ensure diagonal is not active
    active[np.diag_indices_from(active)] = False
    
    # Violations: (2r - d) for active pairs, 0 otherwise
    violations = np.where(active, 2.0 * target_r - dists, 0.0)
    
    # Energy is sum of squared violations (0.5 factor for clean gradient)
    energy_pair = 0.5 * np.sum(violations**2)
    
    # Gradient calculation for pairs
    # d/dx_i [ 0.5 * (2r - ||x_i - x_j||)^2 ]
    # = (2r - d) * d/dx_i (2r - d)
    # = (2r - d) * (-(x_i - x_j) / d)
    # = - (2r - d) / d * (x_i - x_j)
    # Note: violations[i,j] = 2r - d. So we need - violations[i,j] / d * (x_i - x_j)
    # To avoid division by zero, we use safe_dists
    safe_dists = np.where(dists > 1e-12, dists, 1.0)
    unit_vecs = diff / safe_dists[:, :, np.newaxis]
    
    # weights[i,j] = violations[i,j]
    # grad_i = - sum_j weights[i,j] * unit_vecs[i,j]
    grad_pair = -np.sum(violations[:, :, np.newaxis] * unit_vecs, axis=1)
    
    # --- Wall Part ---
    # Distance to each wall
    # Walls are at -0.5 and 0.5 for each dimension
    # dist_wall_k = 0.5 - |c_k|
    wall_dists = 0.5 - np.abs(c)
    
    # Violations: target_r - wall_dist if positive
    wall_violations = np.maximum(0.0, target_r - wall_dists)
    
    energy_wall = np.sum(wall_violations**2)
    
    # Gradient for walls
    # d/dc [ (r - (0.5 - |c|))^2 ]
    # = 2 * (r - (0.5 - |c|)) * d/dc (0.5 - |c|)
    # = 2 * wall_violation * (-sign(c))
    signs = np.sign(c)
    # Handle c=0 case: sign(0)=0, which is fine, gradient is 0 at center
    grad_wall = 2.0 * wall_violations * signs
    
    total_energy = energy_pair + energy_wall
    total_grad = (grad_pair + grad_wall).flatten()
    
    return total_energy, total_grad

def _optimize_bfgs(init_pts, target_r, max_iter=1000, gtol=1e-10, time_limit=50.0):
    """
    Run BFGS optimization with a time limit.
    Returns (optimized_pts, success).
    """
    t_start = time.time()
    
    def objective(x):
        if time.time() - t_start > time_limit:
            raise TimeoutError()
        return _energy_and_grad(x, target_r)
    
    try:
        result = minimize(
            objective,
            init_pts.flatten(),
            method='BFGS',
            jac=True,
            options={'maxiter': max_iter, 'gtol': gtol, 'disp': False}
        )
        opt_pts = result.x.reshape(N, 3)
        return opt_pts, result.success
    except TimeoutError:
        return init_pts, False
    except Exception:
        return init_pts, False

def _init_fcc_like():
    """
    Initialize points in a structure resembling FCC packing.
    FCC is dense. We can use a 5x5x4 grid or similar, but FCC has offset layers.
    For 100 points, a 5x5x4 grid is a good start.
    Let's use a standard grid first, then let the optimizer fix it.
    """
    # A 5x5x4 grid gives 100 points.
    # Spacing: 1/5 in x,y and 1/4 in z? No, that's not cubic.
    # Let's just use a regular cubic grid of 5x5x5 and take 100, or 4x5x5=100.
    # 4x5x5 = 100.
    # Spacing: x: 1/4, y: 1/5, z: 1/5.
    # Min dist is 0.2. Radius 0.1.
    
    nx, ny, nz = 4, 5, 5
    xs = np.linspace(-0.5 + 0.5/nx, 0.5 - 0.5/nx, nx)
    ys = np.linspace(-0.5 + 0.5/ny, 0.5 - 0.5/ny, ny)
    zs = np.linspace(-0.5 + 0.5/nz, 0.5 - 0.5/nz, nz)
    
    points = np.array([(x, y, z) for z in zs for y in ys for x in xs])
    return points

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Initial solution
    points = _init_fcc_like()
    best_points = points.copy()
    best_r = _compute_radius(best_points)
    
    rng = np.random.RandomState(42)
    
    # Strategy: Iteratively push the target radius higher.
    # We start from the current best configuration and try to optimize for a slightly larger radius.
    # If we succeed (violations -> 0 and radius actually increases), we accept.
    # We repeat this process multiple times.
    
    current_pts = best_points.copy()
    
    # Phase 1: Initial jump to a reasonable radius
    # Grid radius is 0.1. Let's try to get to 0.11.
    initial_targets = [0.105, 0.110, 0.112, 0.114, 0.116]
    
    for target_r in initial_targets:
        if time.time() > deadline - 10.0:
            break
            
        # Add small noise to break symmetry
        noise = rng.normal(0, 0.005, size=(N, 3))
        init_pts = np.clip(current_pts + noise, -0.49, 0.49)
        
        remaining_time = deadline - (time.time() - start_time) - 5.0
        if remaining_time < 2.0:
            break
            
        opt_pts, success = _optimize_bfgs(
            init_pts, 
            target_r, 
            max_iter=2000, 
            gtol=1e-10,
            time_limit=remaining_time
        )
        
        if np.all(np.isfinite(opt_pts)):
            # Clip to ensure validity
            opt_pts = np.clip(opt_pts, -0.5 + 1e-9, 0.5 - 1e-9)
            r_opt = _compute_radius(opt_pts)
            
            if r_opt > best_r:
                best_r = r_opt
                best_points = opt_pts.copy()
                current_pts = opt_pts.copy()
            else:
                # If we didn't improve radius, but the configuration is closer to satisfying 
                # the target, it might be a better starting point for the next, slightly larger target?
                # Usually, sticking to the best radius config is safer.
                pass

    # Phase 2: Refinement phase: "Squeeze" the radius
    # Now that we are in a good basin, try to incrementally increase the radius.
    # We will try to find a config that satisfies radius = best_r * (1 + epsilon)
    
    num_squeezes = 20
    for i in range(num_squeezes):
        if time.time() > deadline - 5.0:
            break
            
        # Try to increase radius by a small factor
        # Start with 1.001 and decrease if it fails? 
        # Or just try a fixed small increment.
        # Let's try a dynamic increment: if we succeed, keep it. If we fail, try smaller.
        
        success_improved = False
        # Try a few increments from largest to smallest
        increments = [0.002, 0.001, 0.0005, 0.0002]
        
        for inc in increments:
            if time.time() > deadline - 5.0:
                break
                
            target_r = best_r * (1.0 + inc)
            
            # Add small noise to help escape local minima of the violation landscape
            noise = rng.normal(0, 0.002, size=(N, 3))
            init_pts = np.clip(best_points + noise, -0.49, 0.49)
            
            remaining_time = deadline - (time.time() - start_time) - 2.0
            if remaining_time < 1.0:
                break
                
            opt_pts, success = _optimize_bfgs(
                init_pts, 
                target_r, 
                max_iter=1000, 
                gtol=1e-10,
                time_limit=remaining_time
            )
            
            if np.all(np.isfinite(opt_pts)):
                opt_pts = np.clip(opt_pts, -0.5 + 1e-9, 0.5 - 1e-9)
                r_opt = _compute_radius(opt_pts)
                
                if r_opt > best_r:
                    best_r = r_opt
                    best_points = opt_pts.copy()
                    success_improved = True
                    break # Move to next squeeze iteration with new base
        
        if not success_improved:
            # If we couldn't improve with these increments, try a random restart from a perturbed best
            # This helps if we are stuck in a local minimum
            if rng.random() < 0.5:
                noise = rng.normal(0, 0.01, size=(N, 3))
                init_pts = np.clip(best_points + noise, -0.49, 0.49)
                # Try the current best_r again but from a different start?
                # Or just try a slightly higher target with a bigger jump?
                target_r = best_r * 1.005
                remaining_time = deadline - (time.time() - start_time) - 2.0
                if remaining_time > 1.0:
                    opt_pts, success = _optimize_bfgs(
                        init_pts, 
                        target_r, 
                        max_iter=1000, 
                        gtol=1e-10,
                        time_limit=remaining_time
                    )
                    if np.all(np.isfinite(opt_pts)):
                        opt_pts = np.clip(opt_pts, -0.5 + 1e-9, 0.5 - 1e-9)
                        r_opt = _compute_radius(opt_pts)
                        if r_opt > best_r:
                            best_r = r_opt
                            best_points = opt_pts.copy()

    return best_points

# EVOLVE-BLOCK-END
