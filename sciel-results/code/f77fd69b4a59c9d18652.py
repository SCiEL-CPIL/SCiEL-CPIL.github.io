"""Structural rewrite: SGLD repulsive-gas model with vectorized soft-min energy."""
import numpy as np
import time
from scipy.optimize import minimize

N = 100

def _compute_radius(centers):
    """Compute the maximum common radius for given centers."""
    centers = np.asarray(centers, dtype=float)
    if len(centers) < 2:
        return float(np.min(0.5 - np.abs(centers)))
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    sq_dists = np.sum(diff**2, axis=2)
    np.fill_diagonal(sq_dists, np.inf)
    min_pair_dist = np.sqrt(np.min(sq_dists))
    min_wall_dist = np.min(0.5 - np.abs(centers))
    return float(min(min_pair_dist / 2.0, min_wall_dist))

def _soft_min_energy_and_grad(centers, r_target, beta=50.0):
    """
    Soft-min energy: encourages all pair distances >= 2*r and wall distances >= r.
    E = sum_{i<j} max(0, 2r - d_ij)^2 + sum_i max(0, r - w_i)^2
    with smooth approximation using softplus or direct quadratic.
    """
    centers = np.asarray(centers, dtype=float)
    n = len(centers)
    energy = 0.0
    grad = np.zeros_like(centers)
    
    # Pairwise
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2) + 1e-15)
    np.fill_diagonal(dists, np.inf)
    
    d_min = 2.0 * r_target
    violations = np.maximum(0.0, d_min - dists)
    mask = violations > 0
    
    if np.any(mask):
        # E_pair = sum_{i<j} violations[i,j]^2  (count each pair once)
        # grad[i] += -2 * violations[i,j] * (centers[i]-centers[j]) / dists[i,j]
        coeff = 2.0 * violations / dists
        coeff = np.where(mask, coeff, 0.0)
        # Symmetrize: each pair counted twice in full matrix, so divide by 2
        coeff = 0.5 * coeff
        
        sum_coeff = np.sum(coeff, axis=1)
        sum_coeff_c = np.sum(coeff[:, :, np.newaxis] * centers[np.newaxis, :, :], axis=1)
        grad -= (sum_coeff[:, np.newaxis] * centers) - sum_coeff_c
        energy += 0.5 * np.sum(violations**2)  # factor of 0.5 because we halved coeff
    
    # Walls
    wall_dists = 0.5 - np.abs(centers)
    wall_viol = np.maximum(0.0, r_target - wall_dists)
    wall_mask = wall_viol > 0
    
    if np.any(wall_mask):
        signs = np.sign(centers)
        signs[signs == 0] = 1.0
        grad += 2.0 * wall_viol * signs
        energy += np.sum(wall_viol**2)
    
    return float(energy), grad

def _sgld_step(centers, r_target, rng, step_size, temperature, beta=50.0):
    """One SGLD step: gradient + noise, with wall clipping."""
    e, g = _soft_min_energy_and_grad(centers, r_target, beta)
    # SGLD: x_{t+1} = x_t - lr * grad + sqrt(2*lr*T) * N(0,I)
    noise_scale = np.sqrt(2.0 * step_size * temperature)
    noise = rng.normal(0, noise_scale, size=centers.shape)
    new_centers = centers - step_size * g + noise
    new_centers = np.clip(new_centers, -0.49999, 0.49999)
    return new_centers, e

def _sgld_optimize(centers, r_target, time_limit=20.0, seed=42):
    """Run SGLD with cooling schedule."""
    start = time.time()
    rng = np.random.RandomState(seed)
    c = centers.copy()
    
    # Schedule
    n_steps = int(time_limit * 200)  # ~200 steps/sec target
    T_init = 0.02
    T_final = 0.0005
    lr_init = 0.01
    lr_final = 0.001
    
    best_energy = np.inf
    best_c = c.copy()
    current_energy, _ = _soft_min_energy_and_grad(c, r_target)
    best_energy = current_energy
    
    for step in range(n_steps):
        if time.time() - start > time_limit - 1.0:
            break
        # Linear cooling
        frac = step / max(1, n_steps)
        T = T_init * (T_final / T_init) ** frac
        lr = lr_init * (lr_final / lr_init) ** frac
        
        c, e = _sgld_step(c, r_target, rng, lr, T)
        
        if e < best_energy:
            best_energy = e
            best_c = c.copy()
    
    return best_c, best_energy

def _polish_lbfgs(centers, r_target, time_limit=10.0):
    """Polish with L-BFGS-B."""
    start = time.time()
    
    def obj_and_grad(x):
        if time.time() - start > time_limit:
            return 1e10, np.zeros_like(x)
        c = x.reshape(N, 3)
        e, g = _soft_min_energy_and_grad(c, r_target)
        return e, g.flatten()
    
    x0 = centers.flatten()
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    try:
        result = minimize(
            obj_and_grad, x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 30}
        )
        return result.x.reshape(N, 3)
    except Exception:
        return centers

def _grid_init(n):
    """Grid initialization."""
    m = int(np.ceil(n ** (1.0/3.0)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:n]

def _random_init(n, seed=None):
    """Random initialization inside cube."""
    rng = np.random.RandomState(seed)
    return rng.uniform(-0.45, 0.45, size=(n, 3))

def construct():
    """Construct optimal packing of N spheres in unit cube."""
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_centers = _grid_init(N)
    best_radius = _compute_radius(best_centers)
    
    # Progressive target radii
    targets = [0.10, 0.105, 0.108, 0.110, 0.111, 0.1113, 0.11138, 0.1114, 0.1115]
    
    for t_idx, target_r in enumerate(targets):
        if time.time() > deadline - 25.0:
            break
        if target_r <= best_radius:
            continue
        
        # Multiple restarts with different inits
        n_restarts = 2 if t_idx < 3 else 1
        for restart in range(n_restarts):
            if time.time() > deadline - 20.0:
                break
            
            # Initialize: grid with jitter for first, random for second
            if restart == 0:
                rng_init = np.random.RandomState(100 + t_idx)
                init_c = best_centers + rng_init.normal(0, 0.003, size=(N, 3))
                init_c = np.clip(init_c, -0.4999, 0.4999)
            else:
                init_c = _random_init(N, seed=200 + t_idx)
            
            # SGLD phase
            sgld_time = min(15.0, deadline - time.time() - 15.0)
            if sgld_time < 2.0:
                break
            sgld_c, sgld_e = _sgld_optimize(init_c, target_r, time_limit=sgld_time, seed=500 + t_idx + restart)
            
            # Polish
            polish_time = min(8.0, deadline - time.time() - 10.0)
            if polish_time > 1.0:
                polished_c = _polish_lbfgs(sgld_c, target_r, time_limit=polish_time)
            else:
                polished_c = sgld_c
            
            r_new = _compute_radius(polished_c)
            if r_new > best_radius:
                best_radius = r_new
                best_centers = polished_c.copy()
    
    # Final polish
    if time.time() < deadline - 5.0:
        try:
            final_target = best_radius * 1.0005
            polish_time = min(10.0, deadline - time.time() - 2.0)
            if polish_time > 1.0:
                polished_c = _polish_lbfgs(best_centers, final_target, time_limit=polish_time)
                r_new = _compute_radius(polished_c)
                if r_new > best_radius:
                    best_radius = r_new
                    best_centers = polished_c.copy()
        except Exception:
            pass
    
    best_centers = np.clip(best_centers, -0.4999999, 0.4999999)
    return best_centers
