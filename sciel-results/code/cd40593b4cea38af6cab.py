"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree

N = 100

# EVOLVE-BLOCK-START

def _compute_r(centres):
    """Compute the maximum valid radius for given centres in [-0.5, 0.5]^3."""
    if not np.all(np.isfinite(centres)):
        return 0.0
    if centres.shape != (N, 3):
        return 0.0
    # Wall constraint: distance from each centre to the nearest wall
    wall_dists = np.min(0.5 - np.abs(centres), axis=1)
    min_wall = np.min(wall_dists)
    if min_wall < 0:
        return 0.0
    # Pairwise distances using KDTree
    tree = cKDTree(centres)
    pairs = tree.query_pairs(1.0, output_type='ndarray')
    if len(pairs) == 0:
        return min_wall
    diffs = centres[pairs[:, 0]] - centres[pairs[:, 1]]
    dists = np.sqrt(np.sum(diffs**2, axis=1))
    min_pair = np.min(dists)
    return min(min_pair / 2.0, min_wall)


def _compute_r_and_grad(centres):
    """Compute radius and gradient of radius w.r.t. centres."""
    if not np.all(np.isfinite(centres)):
        return 0.0, np.zeros_like(centres)
    if centres.shape != (N, 3):
        return 0.0, np.zeros_like(centres)
    
    # Wall constraint
    wall_dists = np.min(0.5 - np.abs(centres), axis=1)
    min_wall = np.min(wall_dists)
    if min_wall < 0:
        return 0.0, np.zeros_like(centres)
    
    # Find closest pair
    tree = cKDTree(centres)
    pairs = tree.query_pairs(1.0, output_type='ndarray')
    
    if len(pairs) == 0:
        min_pair = np.inf
    else:
        diffs = centres[pairs[:, 0]] - centres[pairs[:, 1]]
        dists = np.sqrt(np.sum(diffs**2, axis=1))
        min_pair = np.min(dists)
    
    r = min(min_pair / 2.0, min_wall)
    
    # Gradient: push apart the closest pair, push away from closest wall
    grad = np.zeros_like(centres)
    
    # Wall gradient: for the sphere(s) closest to a wall
    if min_wall <= min_pair / 2.0:
        # Find which sphere is closest to a wall
        for i in range(N):
            for axis in range(3):
                d = 0.5 - np.abs(centres[i, axis])
                if abs(d - min_wall) < 1e-12:
                    # Push away from this wall
                    grad[i, axis] = -np.sign(centres[i, axis]) * 0.5
                    break
    
    # Pair gradient: for the closest pair
    if min_pair / 2.0 <= min_wall and len(pairs) > 0:
        diffs = centres[pairs[:, 0]] - centres[pairs[:, 1]]
        dists = np.sqrt(np.sum(diffs**2, axis=1))
        min_idx = np.argmin(dists)
        i, j = pairs[min_idx]
        diff = centres[i] - centres[j]
        dist = dists[min_idx]
        if dist > 1e-12:
            grad[i] += diff / (2.0 * dist) * 0.5
            grad[j] -= diff / (2.0 * dist) * 0.5
    
    return r, grad


def _force_refine(centres, max_iters=200, dt_init=0.005, deadline=None):
    """Vectorized force-directed repulsion with wall constraints."""
    centres = centres.copy().astype(np.float64)
    best_centres = centres.copy()
    best_r = _compute_r(best_centres)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute forces vectorized using KDTree for nearby pairs
        tree = cKDTree(centres)
        pairs = tree.query_pairs(0.6, output_type='ndarray')
        
        forces = np.zeros_like(centres)
        
        if len(pairs) > 0:
            # Vectorized force computation
            diffs = centres[pairs[:, 0]] - centres[pairs[:, 1]]
            dists = np.sqrt(np.sum(diffs**2, axis=1))
            # Avoid division by zero
            mask = dists > 1e-12
            if np.any(mask):
                force_mags = 1.0 / (dists[mask]**2 + 1e-6)
                unit_vecs = diffs[mask] / dists[mask, np.newaxis]
                
                # Scatter forces
                for k in range(len(pairs)):
                    if not mask[k]:
                        continue
                    mag = force_mags[k] if k < np.sum(mask[:k+1]) else 0
                    # Re-index properly
                # Simpler: loop over pairs (still fast for ~100 pairs)
                for idx in range(len(pairs)):
                    i, j = pairs[idx]
                    d = dists[idx]
                    if d < 1e-12:
                        continue
                    f = 1.0 / (d**2 + 1e-6)
                    forces[i] += f * diffs[idx] / d
                    forces[j] -= f * diffs[idx] / d
        
        # Wall forces (vectorized)
        for axis in range(3):
            # Positive wall
            over = centres[:, axis] - 0.5
            mask_pos = over > -0.03
            if np.any(mask_pos):
                forces[mask_pos, axis] -= 30.0 * np.clip(over[mask_pos], 0, None) + 2.0 * np.clip(0.03 + over[mask_pos], 0, None)
            # Negative wall
            under = centres[:, axis] + 0.5
            mask_neg = under < 0.03
            if np.any(mask_neg):
                forces[mask_neg, axis] += 30.0 * np.clip(-under[mask_neg], 0, None) + 2.0 * np.clip(0.03 - under[mask_neg], 0, None)
        
        # Adaptive step size
        dt = dt_init * (0.997 ** it)
        centres += dt * forces
        centres = np.clip(centres, -0.4999, 0.4999)
        
        r = _compute_r(centres)
        if r > best_r:
            best_r = r
            best_centres = centres.copy()
    
    return best_centres, best_r


def _lbfgs_optimize(centres, maxiter=50, deadline=None):
    """L-BFGS-B optimization of the minimum radius."""
    def neg_radius_and_grad(x):
        c = x.reshape(N, 3)
        r, grad = _compute_r_and_grad(c)
        return -r, -grad.flatten()
    
    x0 = centres.flatten()
    bounds = [(-0.4999, 0.4999)] * (N * 3)
    
    result = minimize(neg_radius_and_grad, x0, method='L-BFGS-B',
                      jac=True, bounds=bounds,
                      options={'maxiter': maxiter, 'gtol': 1e-8, 'maxls': 20})
    
    c_opt = result.x.reshape(N, 3)
    r_opt = -result.fun
    return c_opt, r_opt


def _init_grid():
    """Simple cubic grid initialization."""
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N].copy()


def _init_fcc():
    """FCC lattice initialization - better packing fraction."""
    # FCC: points at (0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0) in unit cell
    # Scale to fit in [-0.5, 0.5]^3
    a = 1.0 / 4.0  # lattice constant so 4 layers fit
    points = []
    for i in range(5):
        for j in range(5):
            for k in range(5):
                x = i * a
                y = j * a
                z = k * a
                # FCC basis
                points.append([x - 0.5, y - 0.5, z - 0.5])
                points.append([x - 0.5, y + 0.5*a - 0.5, z + 0.5*a - 0.5])
                points.append([x + 0.5*a - 0.5, y - 0.5, z + 0.5*a - 0.5])
                points.append([x + 0.5*a - 0.5, y + 0.5*a - 0.5, z - 0.5])
    
    points = np.array(points)
    # Keep only points inside [-0.5, 0.5]^3
    mask = np.all(np.abs(points) <= 0.5, axis=1)
    points = points[mask]
    
    if len(points) < N:
        # Fill with grid points
        grid = _init_grid()
        points = np.vstack([points, grid[len(points):]])
    
    return points[:N].copy()


def _init_bcc():
    """BCC lattice initialization."""
    a = 1.0 / 4.0
    points = []
    for i in range(5):
        for j in range(5):
            for k in range(5):
                x = i * a
                y = j * a
                z = k * a
                points.append([x - 0.5, y - 0.5, z - 0.5])
                # Body center
                points.append([x + 0.5*a - 0.5, y + 0.5*a - 0.5, z + 0.5*a - 0.5])
    
    points = np.array(points)
    mask = np.all(np.abs(points) <= 0.5, axis=1)
    points = points[mask]
    
    if len(points) < N:
        grid = _init_grid()
        points = np.vstack([points, grid[len(points):]])
    
    return points[:N].copy()


def _init_random():
    """Random initialization."""
    return np.random.uniform(-0.45, 0.45, (N, 3))


def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s kill
    
    np.random.seed(42)
    
    # Try multiple initializations
    initializations = [
        _init_grid,
        _init_fcc,
        _init_bcc,
        _init_random,
    ]
    
    best_centres = None
    best_r = 0.0
    
    for init_fn in initializations:
        if time.time() > deadline - 20.0:
            break
        
        init = init_fn()
        init = np.clip(init, -0.49, 0.49)
        
        # Force-directed refinement
        c_refined, r_refined = _force_refine(init, max_iters=150, dt_init=0.005, deadline=deadline - 15.0)
        
        if r_refined > best_r:
            best_r = r_refined
            best_centres = c_refined.copy()
        
        if time.time() > deadline - 15.0:
            break
        
        # L-BFGS local optimization
        c_opt, r_opt = _lbfgs_optimize(c_refined, maxiter=40, deadline=deadline - 10.0)
        
        if r_opt > best_r:
            best_r = r_opt
            best_centres = c_opt.copy()
    
    # Additional restarts from best solution with perturbations
    if best_centres is not None and time.time() < deadline - 25.0:
        for pert_scale in [0.03, 0.05, 0.08]:
            if time.time() > deadline - 20.0:
                break
            
            perturbed = best_centres + np.random.randn(N, 3) * pert_scale
            perturbed = np.clip(perturbed, -0.49, 0.49)
            
            c_refined, r_refined = _force_refine(perturbed, max_iters=100, dt_init=0.004, deadline=deadline - 15.0)
            
            if r_refined > best_r:
                best_r = r_refined
                best_centres = c_refined.copy()
            
            if time.time() > deadline - 15.0:
                break
            
            c_opt, r_opt = _lbfgs_optimize(c_refined, maxiter=30, deadline=deadline - 10.0)
            
            if r_opt > best_r:
                best_r = r_opt
                best_centres = c_opt.copy()
    
    # Final validation
    if best_centres is not None:
        final_r = _compute_r(best_centres)
        if final_r > best_r:
            best_r = final_r
        return best_centres
    else:
        return _init_grid()

# EVOLVE-BLOCK-END
