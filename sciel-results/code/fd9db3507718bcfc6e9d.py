"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, minimize_scalar
from scipy.spatial.distance import cdist

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

def _energy_and_gradient(x, R, alpha=1.0):
    """
    Energy and gradient for maximizing radius R.
    x: flat array of length N*3.
    R: target radius.
    alpha: scaling factor for energy terms.
    """
    centers = x.reshape(N, 3)
    n = len(centers)
    
    energy = 0.0
    grad = np.zeros_like(centers)
    
    # Pairwise terms: penalize pairs closer than 2R
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2) + 1e-15)
    np.fill_diagonal(dists, np.inf)
    
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

    # Wall terms: penalize centers closer than R to wall
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
            max_k = np.argmax(abs_centers[idx])
            grad[idx, max_k] += 2.0 * v * np.sign(centers[idx, max_k])
    
    energy *= alpha
    grad *= alpha
    
    return energy, grad.flatten()

def _minimize_energy(init_centers, target_R, max_iter=1000, ftol=1e-14, gtol=1e-12, alpha=1.0):
    """
    Minimizes the energy for a target radius.
    """
    def objective_and_grad(x):
        return _energy_and_gradient(x, target_R, alpha)
    
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
        final_energy, _ = _energy_and_gradient(new_centers, target_R, alpha)
        return new_centers, final_energy, res.success
    except Exception:
        return init_centers, np.inf, False

def _generate_grid_init():
    """Generate initial positions on a grid."""
    nx, ny, nz = 5, 4, 5
    xs = np.linspace(-0.45, 0.45, nx)
    ys = np.linspace(-0.45, 0.45, ny)
    zs = np.linspace(-0.45, 0.45, nz)
    
    grid = np.array([(x, y, z) for z in zs for y in ys for x in xs])
    return grid[:N].copy()

def _generate_fcc_init():
    """Generate initial positions based on FCC lattice."""
    # FCC unit cell positions
    fcc_pts = np.array([
        [0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0],
        [0.5, 0.5, 0.5], [0, 0.5, 0], [0.5, 0, 0], [0, 0, 0.5]
    ])
    
    # Scale to fit in cube
    scale = 0.9
    centers = np.zeros((N, 3))
    
    # Generate FCC lattice points
    pts = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for p in fcc_pts:
                    pt = np.array([i, j, k]) * 0.3 + p * 0.3
                    if -0.45 <= pt[0] <= 0.45 and -0.45 <= pt[1] <= 0.45 and -0.45 <= pt[2] <= 0.45:
                        pts.append(pt)
    
    pts = np.array(pts)
    if len(pts) >= N:
        centers = pts[:N]
    else:
        centers = _generate_grid_init()
    
    return centers

def _generate_random_init(rng):
    """Generate random initial positions."""
    return rng.uniform(-0.45, 0.45, size=(N, 3))

def _stochastic_local_search(centers, current_r, rng, max_iters=500, time_limit=None):
    """
    Stochastic local search to improve packing.
    """
    best_centers = centers.copy()
    best_r = current_r
    start_time = time.time()
    
    temperature = 0.01
    step_size = 0.02
    
    for it in range(max_iters):
        if time_limit is not None and time.time() > time_limit:
            break
            
        # Propose a move
        move_type = rng.random()
        
        if move_type < 0.3:
            # Move one sphere
            idx = rng.integers(N)
            new_centers = best_centers.copy()
            new_centers[idx] += rng.normal(0, step_size, size=3)
        elif move_type < 0.6:
            # Move two spheres
            idx1, idx2 = rng.choice(N, size=2, replace=False)
            new_centers = best_centers.copy()
            new_centers[idx1] += rng.normal(0, step_size, size=3)
            new_centers[idx2] += rng.normal(0, step_size, size=3)
        elif move_type < 0.8:
            # Swap two spheres
            idx1, idx2 = rng.choice(N, size=2, replace=False)
            new_centers = best_centers.copy()
            new_centers[idx1], new_centers[idx2] = new_centers[idx2].copy(), new_centers[idx1].copy()
        else:
            # Random restart of a subset
            subset_size = rng.integers(5, 15)
            subset_idx = rng.choice(N, size=subset_size, replace=False)
            new_centers = best_centers.copy()
            new_centers[subset_idx] = rng.uniform(-0.45, 0.45, size=(subset_size, 3))
        
        # Clip to cube
        new_centers = np.clip(new_centers, -0.49, 0.49)
        
        # Evaluate
        new_r = _compute_radius(new_centers)
        
        # Accept if better or with probability based on temperature
        if new_r > best_r:
            best_centers = new_centers
            best_r = new_r
            temperature *= 0.99
        elif rng.random() < np.exp((new_r - best_r) / (temperature + 1e-10)):
            best_centers = new_centers
    
    return best_centers, best_r

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    rng = np.random.default_rng(42)
    
    # Try multiple initialization strategies
    inits = [
        _generate_grid_init(),
        _generate_fcc_init(),
        _generate_random_init(rng),
        _generate_random_init(rng),
        _generate_random_init(rng)
    ]
    
    best_centers = None
    best_r = 0.0
    
    # Optimize each initialization
    for init in inits:
        if time.time() > deadline - 30.0:
            break
        
        # Initial optimization to establish baseline
        initial_r = 0.08
        centers, energy, _ = _minimize_energy(init, initial_r, max_iter=200)
        r = _compute_radius(centers)
        
        if r > best_r:
            best_centers = centers.copy()
            best_r = r
    
    if best_centers is None or best_r <= 0:
        best_centers = _generate_grid_init()
        best_r = _compute_radius(best_centers)
    
    # Iterative radius increase
    current_r = best_r
    step_size = 0.0005
    consecutive_failures = 0
    max_failures = 10
    
    while time.time() < deadline - 25.0:
        target_r = current_r + step_size
        
        if consecutive_failures >= max_failures:
            step_size *= 0.7
            consecutive_failures = 0
            if step_size < 1e-7:
                break
        
        success = False
        
        # Strategy 1: Direct optimization from current best
        new_centers, energy, opt_success = _minimize_energy(
            best_centers, target_r, max_iter=500
        )
        actual_r = _compute_radius(new_centers)
        if actual_r >= target_r - 1e-6:
            best_centers = new_centers
            best_r = actual_r
            current_r = actual_r
            success = True
            consecutive_failures = 0
        elif actual_r > current_r:
            best_centers = new_centers
            best_r = actual_r
            current_r = actual_r
            consecutive_failures = 0
        
        # Strategy 2: Stochastic local search
        if not success and time.time() < deadline - 30.0:
            sls_centers, sls_r = _stochastic_local_search(
                best_centers, current_r, rng, max_iters=200, 
                time_limit=deadline - 30.0
            )
            if sls_r > current_r:
                best_centers = sls_centers
                best_r = sls_r
                current_r = sls_r
                success = True
                consecutive_failures = 0
        
        # Strategy 3: Perturbed optimization
        if not success and time.time() < deadline - 35.0:
            perturbed = best_centers + rng.normal(0, 0.005, size=(N, 3))
            perturbed = np.clip(perturbed, -0.48, 0.48)
            new_centers, energy, opt_success = _minimize_energy(
                perturbed, target_r, max_iter=400
            )
            actual_r = _compute_radius(new_centers)
            if actual_r > current_r:
                best_centers = new_centers
                best_r = actual_r
                current_r = actual_r
                success = True
                consecutive_failures = 0
        
        if not success:
            consecutive_failures += 1
            
            # Periodic polish
            if time.time() < deadline - 40.0 and consecutive_failures % 5 == 0:
                polished, _, _ = _minimize_energy(best_centers, current_r, max_iter=1000, ftol=1e-16, gtol=1e-14)
                polished_r = _compute_radius(polished)
                if polished_r > best_r:
                    best_centers = polished
                    best_r = polished_r
                    current_r = polished_r
    
    # Final high-precision polish
    if time.time() < deadline - 5.0:
        for polish_iter in range(5):
            if time.time() > deadline - 5.0:
                break
            
            target_r = current_r + 1e-5 * (polish_iter + 1)
            new_centers, _, _ = _minimize_energy(
                best_centers, target_r, max_iter=2000, ftol=1e-18, gtol=1e-16
            )
            actual_r = _compute_radius(new_centers)
            if actual_r > best_r:
                best_centers = new_centers
                best_r = actual_r
                current_r = actual_r
            else:
                new_centers, _, _ = _minimize_energy(
                    best_centers, current_r, max_iter=3000, ftol=1e-18, gtol=1e-16
                )
                actual_r = _compute_radius(new_centers)
                if actual_r > best_r:
                    best_centers = new_centers
                    best_r = actual_r
                    current_r = actual_r
                else:
                    break
    
    # Final validation
    final_pos = np.clip(best_centers, -0.5, 0.5)
    final_r = _compute_radius(final_pos)
    
    return final_pos
# EVOLVE-BLOCK-END
