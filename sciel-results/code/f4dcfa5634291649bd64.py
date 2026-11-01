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
    wall_dists = np.min(0.5 - np.abs(centres), axis=1)
    min_wall = np.min(wall_dists)
    if min_wall < 0:
        return 0.0
    tree = cKDTree(centres)
    pairs = tree.query_pairs(1.0, output_type='ndarray')
    if len(pairs) == 0:
        return min_wall
    diffs = centres[pairs[:, 0]] - centres[pairs[:, 1]]
    dists = np.sqrt(np.sum(diffs**2, axis=1))
    min_pair = np.min(dists)
    return min(min_pair / 2.0, min_wall)


def _feasible_radius(centres, target_r):
    """Check if all spheres of radius target_r fit without overlap and inside cube."""
    if not np.all(np.isfinite(centres)):
        return False
    if centres.shape != (N, 3):
        return False
    # Wall constraint
    wall_dists = np.min(0.5 - np.abs(centres), axis=1)
    if np.min(wall_dists) < target_r - 1e-12:
        return False
    # Pairwise constraint
    tree = cKDTree(centres)
    pairs = tree.query_pairs(2.0 * target_r + 0.01, output_type='ndarray')
    if len(pairs) == 0:
        return True
    diffs = centres[pairs[:, 0]] - centres[pairs[:, 1]]
    dists = np.sqrt(np.sum(diffs**2, axis=1))
    return np.min(dists) >= 2.0 * target_r - 1e-12


def _penalty_objective_and_grad(x, target_r, penalty_weight):
    """
    Penalty-based objective: minimize violations of radius constraints.
    We want all wall distances >= target_r and all pair distances >= 2*target_r.
    Penalty = sum of squared violations.
    """
    c = x.reshape(N, 3)
    
    # Wall violations: max(0, target_r - (0.5 - |c_i|))
    wall_dists = 0.5 - np.abs(c)  # (N, 3)
    wall_viol = np.maximum(0.0, target_r - wall_dists)  # (N, 3)
    
    # Pair violations: max(0, 2*target_r - dist_ij)
    tree = cKDTree(c)
    pairs = tree.query_pairs(2.0 * target_r + 0.05, output_type='ndarray')
    
    pair_viol = np.zeros(len(pairs))
    grad = np.zeros_like(c)
    
    # Wall gradient
    for axis in range(3):
        viol = wall_viol[:, axis]
        if np.any(viol > 0):
            # d/dx of max(0, target_r - (0.5 - |x|)) = -sign(x) when violated
            grad[:, axis] -= np.where(viol > 0, np.sign(c[:, axis]), 0.0)
    
    # Pair gradient
    if len(pairs) > 0:
        diffs = c[pairs[:, 0]] - c[pairs[:, 1]]
        dists = np.sqrt(np.sum(diffs**2, axis=1))
        mask = dists > 1e-12
        viol_mask = (2.0 * target_r - dists) > 0
        
        for idx in range(len(pairs)):
            if not (mask[idx] and viol_mask[idx]):
                continue
            i, j = pairs[idx]
            d = dists[idx]
            v = 2.0 * target_r - d
            # d/dx_i of max(0, 2r - ||c_i - c_j||) = -(c_i - c_j)/||c_i - c_j|| when violated
            grad[i] -= diffs[idx] / d
            grad[j] += diffs[idx] / d
            pair_viol[idx] = v
    
    obj = np.sum(wall_viol**2) + np.sum(pair_viol**2)
    grad *= penalty_weight
    
    return obj, grad.ravel()


def _sca_optimize(centres, target_r, deadline, max_sca_iters=15):
    """
    Successive Convex Approximation: iteratively linearize non-convex constraints.
    At each iteration, solve a QP-like problem using L-BFGS-B on the penalty objective.
    """
    c = centres.copy()
    iu = np.triu_indices(N, k=1)
    bounds = [(-0.4999, 0.4999)] * (N * 3)
    
    best_c = c.copy()
    best_obj = _penalty_objective_and_grad(c.flatten(), target_r, 1.0)[0]
    
    penalty_weights = [1.0, 5.0, 20.0, 100.0, 500.0]
    
    for it in range(max_sca_iters):
        if time.time() > deadline:
            break
        
        pw = penalty_weights[min(it, len(penalty_weights) - 1)]
        
        def obj_grad(x):
            return _penalty_objective_and_grad(x, target_r, pw)
        
        try:
            res = minimize(
                obj_grad,
                c.flatten(),
                jac=True,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 30}
            )
            
            new_c = np.clip(res.x.reshape(N, 3), -0.4999, 0.4999)
            new_obj = _penalty_objective_and_grad(new_c.flatten(), target_r, pw)[0]
            
            if new_obj < best_obj:
                best_obj = new_obj
                best_c = new_c.copy()
            
            c = new_c
        except Exception:
            break
    
    return best_c, best_obj


def _force_refine(centres, max_iters=200, dt_init=0.005, deadline=None):
    """Vectorized force-directed repulsion with wall constraints."""
    centres = centres.copy().astype(np.float64)
    best_centres = centres.copy()
    best_r = _compute_r(best_centres)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        tree = cKDTree(centres)
        pairs = tree.query_pairs(0.6, output_type='ndarray')
        
        forces = np.zeros_like(centres)
        
        if len(pairs) > 0:
            diffs = centres[pairs[:, 0]] - centres[pairs[:, 1]]
            dists = np.sqrt(np.sum(diffs**2, axis=1))
            
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
            over = centres[:, axis] - 0.5
            mask_pos = over > -0.03
            if np.any(mask_pos):
                forces[mask_pos, axis] -= 30.0 * np.clip(over[mask_pos], 0, None) + 2.0 * np.clip(0.03 + over[mask_pos], 0, None)
            under = centres[:, axis] + 0.5
            mask_neg = under < 0.03
            if np.any(mask_neg):
                forces[mask_neg, axis] += 30.0 * np.clip(-under[mask_neg], 0, None) + 2.0 * np.clip(0.03 - under[mask_neg], 0, None)
        
        dt = dt_init * (0.997 ** it)
        centres += dt * forces
        centres = np.clip(centres, -0.4999, 0.4999)
        
        r = _compute_r(centres)
        if r > best_r:
            best_r = r
            best_centres = centres.copy()
    
    return best_centres, best_r


def _binary_search_sca(centres, r_low, r_high, deadline, max_bisections=12):
    """
    Binary search on target radius r, using SCA to check feasibility.
    Returns the best centres found and the achieved radius.
    """
    best_centres = centres.copy()
    best_r = r_low
    
    for bisection in range(max_bisections):
        if time.time() > deadline:
            break
        
        r_mid = 0.5 * (r_low + r_high)
        
        # Try to find a feasible configuration for radius r_mid using SCA
        c_feas, obj = _sca_optimize(centres, r_mid, deadline - 5.0, max_sca_iters=10)
        
        # Check if the SCA result is actually feasible
        if _feasible_radius(c_feas, r_mid):
            r_low = r_mid
            best_r = r_mid
            best_centres = c_feas.copy()
        else:
            r_high = r_mid
        
        # Also try force refinement from the SCA result
        if time.time() < deadline - 10.0:
            c_ref, r_ref = _force_refine(c_feas, max_iters=50, dt_init=0.003, deadline=deadline - 5.0)
            if r_ref > best_r:
                best_r = r_ref
                best_centres = c_ref.copy()
            centres = c_ref
    
    return best_centres, best_r


def _init_grid():
    """Simple cubic grid initialization."""
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N].copy()


def _init_fcc():
    """FCC lattice initialization."""
    a = 1.0 / 4.0
    points = []
    for i in range(5):
        for j in range(5):
            for k in range(5):
                x = i * a
                y = j * a
                z = k * a
                points.append([x - 0.5, y - 0.5, z - 0.5])
                points.append([x - 0.5, y + 0.5*a - 0.5, z + 0.5*a - 0.5])
                points.append([x + 0.5*a - 0.5, y - 0.5, z + 0.5*a - 0.5])
                points.append([x + 0.5*a - 0.5, y + 0.5*a - 0.5, z - 0.5])
    
    points = np.array(points)
    mask = np.all(np.abs(points) <= 0.5, axis=1)
    points = points[mask]
    
    if len(points) < N:
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
    deadline = start_time + 135.0
    
    np.random.seed(42)
    
    initializations = [
        _init_grid,
        _init_fcc,
        _init_bcc,
        _init_random,
    ]
    
    best_centres = None
    best_r = 0.0
    
    for init_fn in initializations:
        if time.time() > deadline - 30.0:
            break
        
        init = init_fn()
        init = np.clip(init, -0.49, 0.49)
        
        # Force-directed refinement first to get a good starting point
        c_refined, r_refined = _force_refine(init, max_iters=100, dt_init=0.005, deadline=deadline - 25.0)
        
        if r_refined > best_r:
            best_r = r_refined
            best_centres = c_refined.copy()
        
        if time.time() > deadline - 25.0:
            break
        
        # Binary search + SCA to push the radius higher
        r_low = r_refined
        r_high = r_refined * 1.15  # Upper bound estimate
        c_bs, r_bs = _binary_search_sca(c_refined, r_low, r_high, deadline - 15.0, max_bisections=10)
        
        if r_bs > best_r:
            best_r = r_bs
            best_centres = c_bs.copy()
    
    # Additional restarts from best solution with perturbations
    if best_centres is not None and time.time() < deadline - 30.0:
        for pert_scale in [0.02, 0.04, 0.06]:
            if time.time() > deadline - 20.0:
                break
            
            perturbed = best_centres + np.random.randn(N, 3) * pert_scale
            perturbed = np.clip(perturbed, -0.49, 0.49)
            
            c_refined, r_refined = _force_refine(perturbed, max_iters=80, dt_init=0.004, deadline=deadline - 15.0)
            
            if r_refined > best_r:
                best_r = r_refined
                best_centres = c_refined.copy()
            
            if time.time() > deadline - 15.0:
                break
            
            # SCA refinement
            r_low = r_refined
            r_high = r_refined * 1.1
            c_bs, r_bs = _binary_search_sca(c_refined, r_low, r_high, deadline - 10.0, max_bisections=6)
            
            if r_bs > best_r:
                best_r = r_bs
                best_centres = c_bs.copy()
    
    # Final validation
    if best_centres is not None:
        final_r = _compute_r(best_centres)
        if final_r > best_r:
            best_r = final_r
        return best_centres
    else:
        return _init_grid()

# EVOLVE-BLOCK-END
