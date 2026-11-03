"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize


def _log_energy(points):
    """Compute log(1/d) energy for points on unit sphere using math.fsum."""
    n = len(points)
    iu = np.triu_indices(n, k=1)
    diff = points[iu[0]] - points[iu[1]]
    dists_sq = np.sum(diff * diff, axis=-1)
    dists = np.sqrt(np.maximum(dists_sq, 1e-30))
    log_dists = np.log(dists)
    return -math.fsum(log_dists.tolist())


def _energy_and_grad(x):
    """Energy and gradient for L-BFGS-B. x is flat array of shape (N*3,)."""
    points = x.reshape(N, 3)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    points = points / norms

    n = len(points)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(dists_sq, 1.0)
    
    inv_dists_sq = 1.0 / dists_sq
    np.fill_diagonal(inv_dists_sq, 0.0)
    
    sum_inv_dists_sq = np.sum(inv_dists_sq, axis=1)
    weighted_points = inv_dists_sq @ points
    grad = weighted_points - points * sum_inv_dists_sq[:, np.newaxis]
    
    dot_products = np.sum(grad * points, axis=1, keepdims=True)
    grad_tangent = grad - dot_products * points
    
    energy = _log_energy(points)
    
    return energy, grad_tangent.flatten()


def _project_to_sphere(points):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return points / norms


def _fibonacci_sphere(n):
    """Generate Fibonacci sphere points."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_uniform_sphere(n, rng):
    """Generate random uniform points on sphere."""
    points = rng.randn(n, 3)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    return points / norms


def _local_minimize(points, deadline, maxiter=500, ftol=1e-14, gtol=1e-14):
    """Run L-BFGS-B to find local minimum."""
    x0 = points.flatten()
    try:
        res = minimize(
            _energy_and_grad,
            x0,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': maxiter, 'maxfun': maxiter * 20, 
                     'ftol': ftol, 'gtol': gtol, 'maxls': 40}
        )
        if np.all(np.isfinite(res.x)):
            p_res = _project_to_sphere(res.x.reshape(N, 3))
            e_check = _log_energy(p_res)
            return p_res, e_check
    except Exception:
        pass
    return points, _log_energy(points)


def _basin_hopping(init_points, deadline, rng, max_iter=80):
    """Basin hopping with L-BFGS-B local minimization."""
    current_points, current_energy = _local_minimize(init_points, deadline, maxiter=300)
    best_points = current_points.copy()
    best_energy = current_energy
    
    T = 0.15
    T_min = 1e-8
    cooling = 0.96
    
    for it in range(max_iter):
        if time.time() > deadline - 3.0:
            break
        if T < T_min:
            break
        
        # Perturb a random subset of points
        n_perturb = rng.randint(3, 12)
        idxs = rng.choice(N, n_perturb, replace=False)
        
        new_points = current_points.copy()
        for idx in idxs:
            rand_vec = rng.randn(3)
            tangent = rand_vec - np.dot(rand_vec, new_points[idx]) * new_points[idx]
            norm_t = np.linalg.norm(tangent)
            if norm_t > 1e-15:
                tangent /= norm_t
            else:
                continue
            step_size = T * rng.uniform(0.5, 1.5)
            new_pt = new_points[idx] + step_size * tangent
            new_points[idx] = new_pt / np.linalg.norm(new_pt)
        
        # Local minimize
        new_points, new_energy = _local_minimize(new_points, deadline, maxiter=200)
        
        # Metropolis acceptance
        delta_e = new_energy - current_energy
        if delta_e < 0 or (T > 0 and rng.rand() < math.exp(-delta_e / T)):
            current_points = new_points
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_points = current_points.copy()
        
        T *= cooling

    return best_points, best_energy


def construct():
    """Construct 150 points on unit sphere minimizing log energy."""
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin

    rng = np.random.default_rng(42)
    
    # Generate diverse initializations
    inits = []
    
    # Fibonacci sphere
    inits.append(_fibonacci_sphere(N))
    
    # Random points with different seeds
    for seed in range(8):
        r = np.random.default_rng(seed * 1000 + 7)
        inits.append(_random_uniform_sphere(N, r))
        
    # Perturbed Fibonacci
    for k in range(6):
        base = _fibonacci_sphere(N)
        perturbed = base + rng.randn(N, 3) * 0.15
        inits.append(_project_to_sphere(perturbed))

    # Phase 1: Local minimize all inits to find good local minima
    best_points = inits[0].copy()
    best_energy = _log_energy(best_points)
    local_min_results = []
    
    for init in inits:
        if time.time() > deadline - 20.0:
            break
        p_res, e_res = _local_minimize(init, deadline, maxiter=400)
        local_min_results.append((e_res, p_res))
        if e_res < best_energy:
            best_energy = e_res
            best_points = p_res.copy()

    # Sort by energy to get the best candidates for basin hopping
    local_min_results.sort(key=lambda x: x[0])
    
    # Phase 2: Basin hopping from the top candidates
    n_bh = min(5, len(local_min_results))
    for i in range(n_bh):
        if time.time() > deadline - 5.0:
            break
        e_init, p_init = local_min_results[i]
        bh_points, bh_energy = _basin_hopping(p_init, deadline, rng, max_iter=60)
        if bh_energy < best_energy:
            best_energy = bh_energy
            best_points = bh_points.copy()

    # Phase 3: Final high-precision polish on the best configuration
    if time.time() < deadline - 3.0:
        best_points, final_energy = _local_minimize(best_points, deadline, 
                                                     maxiter=1000, ftol=1e-16, gtol=1e-16)
        if final_energy < best_energy:
            best_energy = final_energy

    # Ensure validity
    best_points = _project_to_sphere(best_points)
    
    # Verify minimum distance
    diff = best_points[:, np.newaxis, :] - best_points[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_dist = np.min(dist)
    
    if min_dist < 1e-9:
        best_points = _fibonacci_sphere(N)

    return best_points

# EVOLVE-BLOCK-END
