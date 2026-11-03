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
        
    # Wall constraints: distance from center to nearest wall
    max_abs = np.max(np.abs(centers), axis=1)
    wall_dists = 0.5 - max_abs
    min_wall_dist = np.min(wall_dists)
    
    if min_wall_dist < 0:
        return 0.0

    # Pairwise constraints
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    # Set diagonal to infinity to exclude self-distance
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # The radius is half the minimum pairwise distance, limited by wall distance
    r = min(min_pair_dist / 2.0, min_wall_dist)
    return r

def _energy_and_gradient(x, R):
    """
    Calculates the repulsion energy and gradient for a target radius R.
    Energy is the sum of squared violations of the distance constraints.
    """
    centers = x.reshape(-1, 3)
    n = len(centers)
    
    energy = 0.0
    grad = np.zeros_like(centers)
    
    # 1. Pairwise terms
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2) + 1e-15)
    np.fill_diagonal(dists, np.inf)
    
    # Mask for pairs closer than 2R
    mask = dists < 2.0 * R
    if np.any(mask):
        i_idx, j_idx = np.triu_indices(n, k=1)
        
        d_ij = dists[i_idx, j_idx]
        mask_pairs = d_ij < 2.0 * R
        
        if np.any(mask_pairs):
            v_ij = 2.0 * R - d_ij[mask_pairs]
            d_valid = d_ij[mask_pairs]
            i_valid = i_idx[mask_pairs]
            j_valid = j_idx[mask_pairs]
            
            energy += np.sum(v_ij**2)
            
            factors = 2.0 * v_ij / d_valid
            vecs = centers[i_valid] - centers[j_valid]
            forces = factors[:, np.newaxis] * vecs
            
            np.add.at(grad, i_valid, -forces)
            np.add.at(grad, j_valid, forces)

    # 2. Wall terms
    abs_centers = np.abs(centers)
    max_abs = np.max(abs_centers, axis=1)
    wall_dists = 0.5 - max_abs
    
    mask_wall = wall_dists < R
    if np.any(mask_wall):
        violations_wall = R - wall_dists[mask_wall]
        i_valid_wall = np.where(mask_wall)[0]
        
        energy += np.sum(violations_wall**2)
        
        for idx in i_valid_wall:
            v = violations_wall[np.where(i_valid_wall == idx)[0][0]]
            max_val = max_abs[idx]
            max_k = np.argmax(abs_centers[idx])
            
            grad[idx, max_k] += 2.0 * v * np.sign(centers[idx, max_k])
            
    return energy, grad.flatten()

def _minimize_energy(init_centers, target_R, max_iter=300, ftol=1e-12, gtol=1e-10):
    """
    Minimizes the energy for a given target radius.
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

def _polish(pos, max_iter=1000, ftol=1e-14, gtol=1e-12):
    """
    High-precision polish using the hard-min objective.
    """
    def hard_min_obj(x):
        centers = x.reshape(N, 3)
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        min_pair_val = np.min(dists)
        min_pair_idx = np.unravel_index(np.argmin(dists), dists.shape)
        i, j = min_pair_idx
        
        d_walls = HALF - np.abs(centers)
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

    bounds = [(-0.5, 0.5)] * (N * 3)
    try:
        res = minimize(
            hard_min_obj,
            pos.flatten(),
            jac=True,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol, 'disp': False}
        )
        if res.success or res.fun < 0:
            new_pos = res.x.reshape(N, 3)
            if np.all(np.isfinite(new_pos)):
                return new_pos
    except Exception:
        pass
    return pos

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # 1. Initialization: 5x4x5 Grid
    nx, ny, nz = 5, 4, 5
    xs = np.linspace(-0.5 + 0.5/nx, 0.5 - 0.5/nx, nx)
    ys = np.linspace(-0.5 + 0.5/ny, 0.5 - 0.5/ny, ny)
    zs = np.linspace(-0.5 + 0.5/nz, 0.5 - 0.5/nz, nz)
    
    grid = np.array([(x, y, z) for z in zs for y in ys for x in xs])
    best_pos = grid[:N].copy()
    
    # Initial Polish to get a good baseline radius
    best_pos = _polish(best_pos, max_iter=200)
    best_r = _compute_radius(best_pos)
    
    if best_r <= 0:
         rng = np.random.default_rng(42)
         best_pos = rng.uniform(-0.45, 0.45, size=(N, 3))
         best_r = _compute_radius(best_pos)

    # 2. Iterative Radius Increase
    current_r = best_r
    step_size = 0.001
    
    consecutive_failures = 0
    max_failures = 5
    
    rng = np.random.default_rng(123)
    
    while time.time() < deadline - 10.0:
        target_r = current_r + step_size
        
        if consecutive_failures >= max_failures:
            step_size *= 0.5
            consecutive_failures = 0
            if step_size < 1e-6:
                break
        
        candidates = []
        candidates.append(best_pos)
        candidates.append(best_pos + rng.normal(0, 0.01, size=(N, 3)))
        candidates.append(best_pos + rng.normal(0, 0.05, size=(N, 3)))
        
        success = False
        for start_pos in candidates:
            start_pos = np.clip(start_pos, -0.5, 0.5)
            
            new_pos, energy, opt_success = _minimize_energy(start_pos, target_r, max_iter=200)
            
            if energy < 1e-8:
                actual_r = _compute_radius(new_pos)
                if actual_r >= target_r - 1e-5:
                    best_pos = new_pos
                    best_r = actual_r
                    current_r = actual_r
                    success = True
                    consecutive_failures = 0
                    break
            else:
                pass
        
        if not success:
            consecutive_failures += 1
            
            if time.time() < deadline - 20.0 and consecutive_failures % 2 == 0:
                polished, _, _ = _minimize_energy(best_pos, current_r, max_iter=100)
                if _compute_radius(polished) >= current_r - 1e-5:
                    best_pos = polished
                    best_r = _compute_radius(best_pos)

    # 3. Final High-Precision Polish
    if time.time() < deadline - 2.0:
        polished_pos = _polish(best_pos, max_iter=500)
        final_r = _compute_radius(polished_pos)
        if final_r > best_r:
            best_pos = polished_pos
            best_r = final_r

    return best_pos

# EVOLVE-BLOCK-END
