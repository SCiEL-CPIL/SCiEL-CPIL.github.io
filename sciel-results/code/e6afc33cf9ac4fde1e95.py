"""Iterative radius increase with smooth energy minimization for sphere packing."""
import numpy as np
import time
from scipy.optimize import minimize

N = 100
_DEADLINE = 145.0

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

def _energy_and_gradient(centers, target_r, lam=50.0):
    """
    Hard squared-violation energy with vectorized gradient.
    E = lam * sum_{i<j} max(0, 2r - d_ij)^2 + lam * sum_i max(0, r - w_i)^2
    """
    centers = np.asarray(centers, dtype=float)
    n = len(centers)
    energy = 0.0
    grad = np.zeros_like(centers)
    
    # Pair constraints
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    sq_dists = np.sum(diff**2, axis=2)
    np.fill_diagonal(sq_dists, np.inf)
    dists = np.sqrt(sq_dists)
    
    i_idx, j_idx = np.triu_indices(n, k=1)
    pair_dists = dists[i_idx, j_idx]
    violations = 2.0 * target_r - pair_dists
    
    # Hard penalty: only violated pairs contribute
    violated = violations > 0.0
    v = violations[violated]
    if len(v) > 0:
        energy += lam * np.sum(v**2)
        # dE/dv = lam * 2*v
        coeff = lam * 2.0 * v
        # d(v)/d(c_i) = -(c_i - c_j) / d_ij
        # d(v)/d(c_j) = (c_i - c_j) / d_ij
        vi = i_idx[violated]
        vj = j_idx[violated]
        d_vals = pair_dists[violated]
        # Avoid division by zero
        safe_d = np.maximum(d_vals, 1e-12)
        vecs = centers[vi] - centers[vj]
        # grad_i += coeff * (-vec/d), grad_j += coeff * (vec/d)
        factor = coeff / safe_d  # shape (m,)
        # grad[vi] += -factor[:,None] * vecs
        # grad[vj] += factor[:,None] * vecs
        np.add.at(grad, vi, -factor[:, np.newaxis] * vecs)
        np.add.at(grad, vj, factor[:, np.newaxis] * vecs)
    
    # Wall constraints
    wall_dists = 0.5 - np.abs(centers)
    min_wall = np.min(wall_dists, axis=1)
    wall_violations = target_r - min_wall
    
    wv_mask = wall_violations > 0.0
    if np.any(wv_mask):
        wv = wall_violations[wv_mask]
        energy += lam * np.sum(wv**2)
        coeff_w = lam * 2.0 * wv
        wi = np.where(wv_mask)[0]
        min_k = np.argmin(wall_dists[wi], axis=1)
        signs = np.sign(centers[wi, min_k])
        signs[signs == 0] = 1.0
        grad[wi, min_k] += coeff_w * signs
    
    return energy, grad

def _minimize_energy(centers, target_r, lam=50.0, time_limit=10.0):
    """Minimize energy for given target radius using L-BFGS-B."""
    start = time.time()
    
    def obj_and_grad(x):
        if time.time() - start > time_limit:
            return 1e10, np.zeros_like(x)
        c = x.reshape(N, 3)
        e, g = _energy_and_gradient(c, target_r, lam)
        return e, g.flatten()
    
    x0 = centers.flatten()
    bounds = [(-0.4999999, 0.4999999)] * (N * 3)
    
    try:
        result = minimize(
            obj_and_grad, x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 50, 'maxfun': 5000}
        )
        return result.x.reshape(N, 3)
    except Exception:
        return centers

def _grid_init(n, seed=42):
    """Grid initialization with small random perturbation to break symmetry."""
    rng = np.random.RandomState(seed)
    m = int(np.ceil(n ** (1.0/3.0)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    grid = grid[:n].copy()
    # Small perturbation to break degeneracy
    grid += rng.uniform(-0.005, 0.005, size=grid.shape)
    return grid

def _local_restarts(centers, target_r, lam, n_restarts=3, time_per=3.0):
    """Try small random perturbations and re-optimize to escape local minima."""
    best = centers.copy()
    best_r = _compute_radius(centers)
    rng = np.random.RandomState(int(time.time()) % 100000)
    
    for _ in range(n_restarts):
        perturbed = centers + rng.normal(0, 0.01, size=centers.shape)
        perturbed = np.clip(perturbed, -0.4999999, 0.4999999)
        result = _minimize_energy(perturbed, target_r, lam=lam, time_limit=time_per)
        result = np.clip(result, -0.4999999, 0.4999999)
        r = _compute_radius(result)
        if r > best_r:
            best_r = r
            best = result.copy()
    
    return best, best_r

def construct():
    """Construct optimal packing of N spheres in unit cube."""
    start_time = time.time()
    deadline = start_time + _DEADLINE
    
    # Start with grid initialization
    centers = _grid_init(N)
    current_r = _compute_radius(centers)
    best_centers = centers.copy()
    best_radius = current_r
    
    # Iterative radius increase strategy
    max_target = 0.112  # Slightly above known best 0.11138
    
    # Phase 1: Gradually increase target radius with local restarts
    n_increments = 40
    stall_count = 0
    for step in range(n_increments):
        if time.time() > deadline - 25.0:
            break
        
        progress = step / n_increments
        target_r = current_r + (max_target - current_r) * progress
        target_r = min(target_r, max_target)
        
        remaining = deadline - time.time() - 20.0
        if remaining < 3.0:
            break
        
        # Adaptive lambda: higher as we push harder
        lam = 40.0 + 120.0 * progress
        
        t_opt = min(remaining * 0.5, 6.0)
        centers = _minimize_energy(centers, target_r, lam=lam, time_limit=t_opt)
        centers = np.clip(centers, -0.4999999, 0.4999999)
        
        r_new = _compute_radius(centers)
        
        # Local restarts if we're stuck
        if r_new <= best_radius + 1e-9 and step > 5:
            stall_count += 1
            if stall_count >= 3 and time.time() < deadline - 20.0:
                t_restart = min(remaining * 0.3, 4.0)
                centers, r_new = _local_restarts(
                    centers, target_r, lam, 
                    n_restarts=2, time_per=t_restart
                )
                centers = np.clip(centers, -0.4999999, 0.4999999)
                stall_count = 0
        else:
            stall_count = 0
        
        if r_new > best_radius:
            best_radius = r_new
            best_centers = centers.copy()
        
        current_r = max(current_r, r_new)
        
        # If we've hit the target, stop increasing
        if r_new >= target_r - 1e-8:
            continue
    
    # Phase 2: Push slightly above best with high lambda
    if time.time() < deadline - 10.0:
        remaining = deadline - time.time() - 8.0
        if remaining > 3.0:
            target_r2 = best_radius * 1.002
            centers2 = _minimize_energy(best_centers, target_r2, lam=150.0, time_limit=min(remaining, 8.0))
            centers2 = np.clip(centers2, -0.4999999, 0.4999999)
            r2 = _compute_radius(centers2)
            if r2 > best_radius:
                best_radius = r2
                best_centers = centers2.copy()
    
    # Phase 3: Final polish with very high lambda and tight tolerance
    if time.time() < deadline - 4.0:
        remaining = deadline - time.time() - 2.0
        if remaining > 2.0:
            centers3 = _minimize_energy(best_centers, best_radius, lam=300.0, time_limit=min(remaining, 5.0))
            centers3 = np.clip(centers3, -0.4999999, 0.4999999)
            r3 = _compute_radius(centers3)
            if r3 > best_radius:
                best_radius = r3
                best_centers = centers3.copy()
    
    best_centers = np.clip(best_centers, -0.4999999, 0.4999999)
    return best_centers
