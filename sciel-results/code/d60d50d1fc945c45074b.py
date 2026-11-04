import numpy as np
import time
from scipy.optimize import minimize

N = 100
HALF = 0.5

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
        
    # Wall constraints
    max_abs = np.max(np.abs(centers), axis=1)
    wall_dists = 0.5 - max_abs
    min_wall_dist = np.min(wall_dists)
    
    if min_wall_dist < -1e-12:
        return 0.0

    # Pairwise constraints
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    r = min(min_pair_dist / 2.0, min_wall_dist)
    return r

def _soft_min_obj_grad(x, beta, wall_weight=1.0):
    """
    Computes the negative log-sum-exp of pairwise distances and wall distances.
    This is a smooth surrogate for -min(dists).
    
    x: (N*3,) array of centers
    beta: inverse temperature (higher is sharper)
    """
    centers = x.reshape(N, 3)
    
    # 1. Pairwise distances
    # Compute squared distances for numerical stability if needed, but direct dist is fine for beta < 1000
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2) + 1e-15)
    np.fill_diagonal(dists, np.inf)
    
    # We want to maximize min(dists). 
    # Objective: - (1/beta) * log( sum_{i<j} exp(-beta * d_ij) )
    # This approximates -min(d_ij) for large beta.
    
    # To avoid overflow, subtract the max (which is min of -d, so max of d? No.)
    # log-sum-exp trick: log(sum(exp(z))) = max(z) + log(sum(exp(z - max(z))))
    # Here z_ij = -beta * d_ij.
    # max(z) = -beta * min(d).
    
    d_flat = dists[np.triu_indices(N, k=1)]
    z = -beta * d_flat
    
    z_max = np.max(z)
    # log_sum_exp = z_max + log(sum(exp(z - z_max)))
    log_sum_exp = z_max + np.log(np.sum(np.exp(z - z_max)))
    
    obj_pair = log_sum_exp / beta  # This is approx -min(d)
    
    # Gradient of log_sum_exp w.r.t d_ij:
    # d(log_sum_exp)/d(z_ij) = exp(z_ij - z_max) / sum(exp(z - z_max))
    # d(z_ij)/d(d_ij) = -beta
    # So d(log_sum_exp)/d(d_ij) = -beta * weights
    # d(obj_pair)/d(d_ij) = -weights
    
    weights = np.exp(z - z_max)
    sum_weights = np.sum(weights)
    grad_d = -weights / sum_weights  # Gradient of obj w.r.t distances
    
    # Map back to gradient w.r.t centers
    grad_centers = np.zeros_like(centers)
    # d_ij = |c_i - c_j|
    # d(d_ij)/d(c_i) = (c_i - c_j) / d_ij
    # d(d_ij)/d(c_j) = (c_j - c_i) / d_ij
    
    i_idx, j_idx = np.triu_indices(N, k=1)
    vecs = centers[i_idx] - centers[j_idx]
    d_vals = d_flat
    
    # Avoid division by zero
    safe_d = np.where(d_vals < 1e-12, 1e-12, d_vals)
    
    # grad_centers[i] += grad_d * (c_i - c_j) / d
    # grad_centers[j] += grad_d * (c_j - c_i) / d = - grad_d * (c_i - c_j) / d
    
    forces = (grad_d / safe_d)[:, np.newaxis] * vecs
    
    np.add.at(grad_centers, i_idx, forces)
    np.add.at(grad_centers, j_idx, -forces)
    
    # 2. Wall distances
    # d_wall_i = 0.5 - max(|c_ix|, |c_iy|, |c_iz|)
    # We want to maximize min(d_wall).
    # Surrogate: - (1/beta) * log( sum_i exp(-beta * d_wall_i) )
    
    max_abs = np.max(np.abs(centers), axis=1)
    d_walls = 0.5 - max_abs
    z_w = -beta * d_walls
    z_w_max = np.max(z_w)
    log_sum_exp_w = z_w_max + np.log(np.sum(np.exp(z_w - z_w_max)))
    
    obj_wall = log_sum_exp_w / beta
    
    total_obj = obj_pair + wall_weight * obj_wall
    
    # Gradient for wall part
    w_weights = np.exp(z_w - z_w_max)
    sum_w_weights = np.sum(w_weights)
    grad_dw = -w_weights / sum_w_weights  # Grad of obj_wall w.r.t d_walls
    
    # d(d_wall_i)/d(c_i) is non-zero only for the max coordinate
    # d(0.5 - max(|x|,|y|,|z|))/d(x) = -sign(x) if |x| is max, else 0
    
    grad_centers_wall = np.zeros_like(centers)
    for i in range(N):
        max_val = max_abs[i]
        # Find indices where abs is close to max (handle ties)
        is_max = np.abs(centers[i]) >= max_val - 1e-9
        if np.any(is_max):
            # Subgradient: average over ties
            grad_vec = np.zeros(3)
            count = np.sum(is_max)
            for k in range(3):
                if is_max[k]:
                    # d(d_wall)/d(c_k) = -sign(c_k)
                    grad_vec[k] = -np.sign(centers[i, k]) / count
            grad_centers_wall[i] = grad_dw[i] * grad_vec
            
    total_grad = grad_centers + wall_weight * grad_centers_wall
    
    return total_obj, total_grad.flatten()

def _hard_min_obj_grad(x):
    """
    Exact gradient for the hard minimum radius.
    Used for final polishing.
    """
    centers = x.reshape(N, 3)
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    min_pair_val = np.min(dists)
    min_pair_idx = np.unravel_index(np.argmin(dists), dists.shape)
    i, j = min_pair_idx
    
    max_abs = np.max(np.abs(centers), axis=1)
    d_walls = 0.5 - max_abs
    min_wall_val = np.min(d_walls)
    min_wall_idx = np.unravel_index(np.argmin(d_walls), d_walls.shape)
    wi, wk = min_wall_idx
    
    r_pair = min_pair_val / 2.0
    r_wall = min_wall_val
    
    grad = np.zeros_like(centers)
    
    if r_pair <= r_wall:
        d = min_pair_val
        if d > 1e-12:
            vec = (centers[i] - centers[j]) / d
            grad[i] += 0.5 * vec
            grad[j] += -0.5 * vec
        return -r_pair, -grad.flatten()
    else:
        s = np.sign(centers[wi, wk])
        if s == 0:
            s = 1.0
        grad[wi, wk] = -s
        return -r_wall, -grad.flatten()

def _generate_fcc_init():
    """
    Generate an FCC lattice initialization.
    FCC has a packing density of ~74%.
    We generate a lattice slightly larger than the cube, then scale/shift to fit.
    """
    # FCC basis: (0,0,0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0) in units of 'a'
    # Lattice constant 'a' such that spheres of radius r touch.
    # Nearest neighbor distance in FCC is a / sqrt(2).
    # So 2r = a / sqrt(2) => a = 2r * sqrt(2).
    
    # We want to fit ~100 spheres.
    # Volume of cube = 1.
    # Volume of 100 spheres = 100 * (4/3) pi r^3.
    # Density ~ 0.74 => 100 * (4/3) pi r^3 * (1/0.74) ~ 1.
    # r^3 ~ 0.74 * 3 / (400 * pi) ~ 0.00176 => r ~ 0.12.
    # Current best is ~0.111.
    # Let's start with a grid of FCC points.
    
    # Number of lattice points per dimension.
    # If we use a cube of side L, number of points ~ (L/a)^3 * 4.
    # Let's try nx=5, ny=5, nz=5 lattice cells.
    # Total points ~ 4 * 5^3 = 500. Too many.
    # We need 100.
    # Maybe 4x4x4 cells? 4*64=256. Still too many.
    # We can just generate a large FCC cloud, pick the 100 closest to the center,
    # and then optimize.
    
    rng = np.random.default_rng(42)
    
    # Generate FCC points in a larger box
    n_side = 6
    a = 0.3  # Lattice constant, roughly 2*r*sqrt(2) if r~0.1
    
    points = []
    for i in range(-n_side//2, n_side//2 + 1):
        for j in range(-n_side//2, n_side//2 + 1):
            for k in range(-n_side//2, n_side//2 + 1):
                # FCC basis
                points.append([i*a, j*a, k*a])
                points.append([(i+0.5)*a, (j+0.5)*a, k*a])
                points.append([(i+0.5)*a, j*a, (k+0.5)*a])
                points.append([i*a, (j+0.5)*a, (k+0.5)*a])
    
    points = np.array(points)
    
    # Center the cloud
    points -= np.mean(points, axis=0)
    
    # Sort by distance from origin to pick the densest central cluster
    dists_to_origin = np.linalg.norm(points, axis=1)
    sorted_indices = np.argsort(dists_to_origin)
    
    # Take the first N
    init_points = points[sorted_indices[:N]]
    
    # Scale to fit inside [-0.5, 0.5]
    # Current max extent
    max_ext = np.max(np.abs(init_points))
    if max_ext > 0:
        init_points *= (0.45 / max_ext)
        
    return init_points

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # 1. Initialization
    best_pos = _generate_fcc_init()
    best_pos = np.clip(best_pos, -0.49, 0.49)
    
    # Initial polish with soft min to settle
    current_beta = 50.0
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    try:
        res = minimize(
            _soft_min_obj_grad,
            best_pos.flatten(),
            args=(current_beta,),
            jac=True,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8, 'disp': False}
        )
        best_pos = res.x.reshape(N, 3)
    except Exception:
        pass
        
    best_r = _compute_radius(best_pos)
    
    # 2. Iterative Beta Increase (Simulated Annealing on the Surrogate)
    # Increase beta to make the soft-min sharper, pushing towards the hard-min.
    beta = 50.0
    max_beta = 1000.0
    
    while time.time() < deadline - 10.0 and beta < max_beta:
        # Increase beta
        beta *= 1.5
        if beta > max_beta:
            beta = max_beta
            
        try:
            res = minimize(
                _soft_min_obj_grad,
                best_pos.flatten(),
                args=(beta,),
                jac=True,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10, 'disp': False}
            )
            new_pos = res.x.reshape(N, 3)
            
            # Check improvement
            new_r = _compute_radius(new_pos)
            if new_r > best_r:
                best_pos = new_pos
                best_r = new_r
            else:
                # If no improvement, try a small random restart from current best
                # with a slightly perturbed position
                perturbation = np.random.normal(0, 0.01, size=(N, 3))
                perturbed_pos = np.clip(best_pos + perturbation, -0.49, 0.49)
                
                res2 = minimize(
                    _soft_min_obj_grad,
                    perturbed_pos.flatten(),
                    args=(beta,),
                    jac=True,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-10, 'disp': False}
                )
                new_pos2 = res2.x.reshape(N, 3)
                new_r2 = _compute_radius(new_pos2)
                if new_r2 > best_r:
                    best_pos = new_pos2
                    best_r = new_r2
                    
        except Exception:
            pass

    # 3. Final High-Precision Polish
    # Use the hard-min objective for exact gradient steps
    if time.time() < deadline - 2.0:
        try:
            res = minimize(
                _hard_min_obj_grad,
                best_pos.flatten(),
                jac=True,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12, 'disp': False}
            )
            new_pos = res.x.reshape(N, 3)
            new_r = _compute_radius(new_pos)
            if new_r > best_r:
                best_pos = new_pos
                best_r = new_r
        except Exception:
            pass

    return best_pos

# EVOLVE-BLOCK-END
