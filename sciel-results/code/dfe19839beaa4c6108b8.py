"""
Sphere packing: 100 equal spheres in unit cube [-0.5, 0.5]^3.
Combined approach: iterative radius increase with energy minimization,
simulated annealing for escape, and final epigraph polish.
"""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import cKDTree

N = 100
HALF = 0.5

def _compute_radius(centers):
    """Compute maximum common radius for given centers."""
    if centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
        
    # Wall distances
    wall_dists = HALF - np.abs(centers)
    min_wall_dist = np.min(wall_dists)
    if min_wall_dist < 0:
        return 0.0
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    return float(min(min_pair_dist / 2.0, min_wall_dist))

def _energy_and_gradient(x, R):
    """
    Energy: sum of squared violations of distance constraints for target radius R.
    Returns (energy, gradient) for minimization.
    """
    centers = x.reshape(N, 3)
    energy = 0.0
    grad = np.zeros_like(centers)
    
    # Pairwise: enforce ||c_i - c_j|| >= 2R
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    sq_dists = np.sum(diff**2, axis=2)
    dists = np.sqrt(sq_dists + 1e-15)
    np.fill_diagonal(dists, np.inf)
    np.fill_diagonal(sq_dists, np.inf)
    
    mask = dists < 2.0 * R
    if np.any(mask):
        i_idx, j_idx = np.triu_indices(N, k=1)
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
    
    # Wall: enforce wall_dist >= R
    abs_centers = np.abs(centers)
    max_abs = np.max(abs_centers, axis=1)
    wall_dists = HALF - max_abs
    
    mask_wall = wall_dists < R
    if np.any(mask_wall):
        violations_wall = R - wall_dists[mask_wall]
        i_valid_wall = np.where(mask_wall)[0]
        
        energy += np.sum(violations_wall**2)
        
        for idx in i_valid_wall:
            v = violations_wall[np.where(i_valid_wall == idx)[0][0]]
            max_k = np.argmax(abs_centers[idx])
            grad[idx, max_k] += 2.0 * v * np.sign(centers[idx, max_k])
    
    return energy, grad.flatten()

def _minimize_energy(init_centers, target_R, max_iter=500, ftol=1e-14, gtol=1e-12, time_limit=10.0):
    """Minimize energy for given target radius."""
    start_time = time.time()
    
    def objective_and_grad(x):
        if time.time() - start_time > time_limit:
            return 1e10, np.zeros_like(x)
        val, g = _energy_and_gradient(x, target_R)
        if not np.all(np.isfinite(g)):
            return 1e10, np.zeros_like(x)
        return val, g

    bounds = [(-HALF, HALF)] * (N * 3)
    
    try:
        res = minimize(
            objective_and_grad,
            init_centers.flatten(),
            jac=True,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol, 'maxls': 50, 'disp': False}
        )
        new_centers = res.x.reshape(N, 3)
        final_energy, _ = _energy_and_gradient(new_centers, target_R)
        return new_centers, final_energy
    except Exception:
        return init_centers, np.inf

def _simulated_annealing(centers, time_limit=10.0, seed=42):
    """Simulated annealing to escape local minima."""
    start_time = time.time()
    centers = centers.copy()
    best_centers = centers.copy()
    best_radius = _compute_radius(centers)
    
    T0 = 0.06
    Tend = 0.0003
    n_steps = 3000
    cooling_rate = (Tend / T0) ** (1.0 / n_steps)
    
    rng = np.random.RandomState(seed)
    T = T0
    current_radius = best_radius
    
    for step in range(n_steps):
        if time.time() - start_time > time_limit:
            break
        
        # Random number of spheres to move
        if step % 7 == 0:
            n_move = 1
        elif step % 7 == 1:
            n_move = 2
        else:
            n_move = 3
            
        idx = rng.choice(N, size=n_move, replace=False)
        disp = rng.normal(0, 0.025, size=(n_move, 3))
        
        new_centers = centers.copy()
        for k, i in enumerate(idx):
            new_centers[i] += disp[k]
        new_centers = np.clip(new_centers, -0.49999, 0.49999)
        
        new_radius = _compute_radius(new_centers)
        delta = new_radius - current_radius
        
        if delta > 0 or rng.random() < np.exp(delta / max(T, 1e-10)):
            centers = new_centers
            current_radius = new_radius
            if new_radius > best_radius:
                best_radius = new_radius
                best_centers = centers.copy()
        
        T *= cooling_rate
    
    return best_centers

def _epigraph_polish(centers, time_limit=8.0):
    """High-precision polish using epigraph formulation."""
    start_time = time.time()
    centers = centers.copy()
    best_radius = _compute_radius(centers)
    
    def _epigraph_obj_and_grad(x, t, mu):
        c = x.reshape(N, 3)
        
        # Pairwise constraints: ||c_i - c_j||^2 >= 4*t^2
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        sq_dists = np.sum(diff**2, axis=2)
        np.fill_diagonal(sq_dists, np.inf)
        
        violations = np.maximum(0, 4*t*t - sq_dists)
        pair_penalty = np.sum(violations**2)
        
        # Wall constraints: wall_dist >= t
        wall_dists = HALF - np.abs(c)
        wall_violations = np.maximum(0, t - wall_dists)
        wall_penalty = np.sum(wall_violations**2)
        
        obj = -t + mu * (pair_penalty + wall_penalty)
        
        grad = np.zeros_like(c)
        
        # Pair gradient
        finite_mask = np.isfinite(sq_dists)
        if np.any(finite_mask):
            active = violations > 0
            if np.any(active):
                for i in range(N):
                    for j in range(N):
                        if i != j and active[i, j]:
                            grad[i] += -4.0 * mu * violations[i, j] * (c[i] - c[j])
        
        # Wall gradient
        active_wall = wall_violations > 0
        if np.any(active_wall):
            signs = np.sign(c)
            signs[c == 0] = 1.0
            for i in range(N):
                for k in range(3):
                    if active_wall[i, k]:
                        grad[i, k] = 2.0 * mu * wall_violations[i, k] * (-signs[i, k])
        
        return obj, grad.flatten()
    
    mu_values = [1.0, 10.0, 100.0, 1000.0]
    t = _compute_radius(centers) * 0.95
    
    for mu in mu_values:
        if time.time() - start_time > time_limit:
            break
        
        x0 = centers.flatten()
        t0 = t
        
        def obj_grad_combined(x):
            if time.time() - start_time > time_limit:
                return 1e10, np.zeros_like(x)
            val, grad = _epigraph_obj_and_grad(x, t0, mu)
            if not np.all(np.isfinite(grad)):
                return 1e10, np.zeros_like(x)
            return val, grad
        
        try:
            result = minimize(
                obj_grad_combined, x0,
                method='L-BFGS-B',
                bounds=[(-HALF, HALF)] * (N * 3),
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 40}
            )
            centers = result.x.reshape(N, 3)
            r = _compute_radius(centers)
            if r > best_radius:
                best_radius = r
            t = min(r * 1.02, HALF)
        except Exception:
            pass
    
    return centers

def _grid_init(n, nx=5, ny=4, nz=5):
    """Generate grid initial configuration."""
    xs = np.linspace(-HALF + HALF/nx, HALF - HALF/nx, nx)
    ys = np.linspace(-HALF + HALF/ny, HALF - HALF/ny, ny)
    zs = np.linspace(-HALF + HALF/nz, HALF - HALF/nz, nz)
    grid = np.array([(x, y, z) for z in zs for y in ys for x in xs])
    return grid[:n]

def _jitter_init(n, seed=None, jitter=0.015, nx=5, ny=4, nz=5):
    """Generate jittered grid initial configuration."""
    grid = _grid_init(n, nx, ny, nz)
    rng = np.random.RandomState(seed)
    noise = rng.uniform(-jitter, jitter, size=grid.shape)
    return np.clip(grid + noise, -0.49, 0.49)

def _random_init(n, seed=None):
    """Generate random initial configuration."""
    rng = np.random.RandomState(seed)
    return rng.uniform(-0.45, 0.45, size=(n, 3))

def _iterative_radius_increase(init_centers, start_r, deadline, seed=42):
    """
    Iteratively increase radius by minimizing energy at each target radius.
    Returns best centers found.
    """
    centers = init_centers.copy()
    current_r = start_r
    best_r = current_r
    best_centers = centers.copy()
    
    step_size = 0.0008
    consecutive_failures = 0
    max_failures = 4
    
    rng = np.random.RandomState(seed)
    
    while time.time() < deadline - 20.0:
        target_r = current_r + step_size
        
        if consecutive_failures >= max_failures:
            step_size *= 0.75
            consecutive_failures = 0
            if step_size < 1e-7:
                break
        
        # Try multiple starting points
        candidates = [
            centers,
            centers + rng.normal(0, 0.005, size=(N, 3)),
            centers + rng.normal(0, 0.015, size=(N, 3)),
        ]
        
        success = False
        for start_pos in candidates:
            start_pos = np.clip(start_pos, -HALF, HALF)
            
            max_iter = 400 if current_r < 0.11 else 600
            time_limit = min(8.0, deadline - time.time() - 25.0)
            if time_limit < 3.0:
                break
                
            new_pos, energy = _minimize_energy(start_pos, target_r, max_iter=max_iter, time_limit=time_limit)
            
            actual_r = _compute_radius(new_pos)
            
            if energy < 1e-10 and actual_r >= target_r - 1e-6:
                # Full success
                centers = new_pos
                current_r = actual_r
                if current_r > best_r:
                    best_r = current_r
                    best_centers = centers.copy()
                success = True
                consecutive_failures = 0
                break
            elif actual_r > current_r:
                # Partial success: improved radius
                centers = new_pos
                current_r = actual_r
                if current_r > best_r:
                    best_r = current_r
                    best_centers = centers.copy()
                # Don't count as full success for step size
                break
        
        if not success:
            consecutive_failures += 1
            
            # Try simulated annealing to escape
            if time.time() < deadline - 25.0:
                sa_time = min(6.0, deadline - time.time() - 25.0)
                sa_centers = _simulated_annealing(centers, time_limit=sa_time, seed=seed + consecutive_failures)
                sa_r = _compute_radius(sa_centers)
                if sa_r > current_r:
                    centers = sa_centers
                    current_r = sa_r
                    if current_r > best_r:
                        best_r = current_r
                        best_centers = centers.copy()
        
        # Check time
        if time.time() > deadline - 25.0:
            break
    
    return best_centers, best_r

def construct():
    """Construct optimal packing of N spheres in unit cube."""
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_centers = None
    best_radius = 0.0
    
    # Generate multiple initializations
    inits = []
    
    # Grid
    inits.append(_grid_init(N))
    
    # Jittered grids with different seeds and parameters
    for seed in range(6):
        inits.append(_jitter_init(N, seed=seed, jitter=0.015))
    
    # Different grid dimensions
    inits.append(_jitter_init(N, seed=10, jitter=0.01, nx=4, ny=5, nz=5))
    inits.append(_jitter_init(N, seed=11, jitter=0.01, nx=5, ny=5, nz=4))
    
    # Random inits
    for seed in range(4):
        inits.append(_random_init(N, seed=seed))
    
    # Optimize each initialization
    for i, init in enumerate(inits):
        if time.time() > deadline - 30.0:
            break
        
        try:
            # Get initial radius
            init_r = _compute_radius(init)
            
            # First, do a quick energy minimization to establish a baseline
            time_remaining = deadline - time.time() - 30.0
            if time_remaining < 10.0:
                break
            
            # Quick optimization to improve initial position
            quick_time = min(5.0, time_remaining * 0.3)
            if quick_time > 2.0:
                target_r_init = init_r * 1.05
                optimized_init, _ = _minimize_energy(init, target_r_init, max_iter=200, time_limit=quick_time)
                init_r = _compute_radius(optimized_init)
                init = optimized_init
            
            # Iterative radius increase
            time_remaining = deadline - time.time() - 25.0
            if time_remaining < 15.0:
                break
            
            iter_time = min(25.0, time_remaining)
            centers, r = _iterative_radius_increase(init, init_r, deadline, seed=100 + i)
            
            if r > best_radius:
                best_radius = r
                best_centers = centers.copy()
                
        except Exception:
            continue
    
    # Final polish on the best result
    if best_centers is not None and time.time() < deadline - 10.0:
        try:
            ep_time = min(8.0, deadline - time.time() - 5.0)
            ep_centers = _epigraph_polish(best_centers, time_limit=ep_time)
            r_ep = _compute_radius(ep_centers)
            if r_ep > best_radius:
                best_radius = r_ep
                best_centers = ep_centers.copy()
        except Exception:
            pass
    
    # Final validation
    if best_centers is None:
        best_centers = _grid_init(N)
    
    best_centers = np.clip(best_centers, -0.4999999, 0.4999999)
    
    return best_centers
