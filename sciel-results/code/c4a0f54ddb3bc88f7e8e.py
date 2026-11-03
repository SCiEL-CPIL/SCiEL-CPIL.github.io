"""Combined program for logarithmic energy minimization of 150 points on sphere."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize

_DEADLINE = 143.0  # Safety margin before 150s kill

def _log_energy(points):
    """Compute log(1/d) energy for points on unit sphere using math.fsum for precision."""
    n = len(points)
    if n < 2:
        return 0.0
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(dists_sq, 1.0)
    dists = np.sqrt(dists_sq)
    iu = np.triu_indices(n, k=1)
    log_dists = np.log(dists[iu])
    return -math.fsum(log_dists.tolist())

def _energy_and_grad(x):
    """
    Energy and tangential gradient for L-BFGS-B. 
    x is flat array of shape (N*3,).
    The function normalizes x to the sphere internally.
    Returns (energy, tangential_gradient_flat).
    Optimized for speed using vectorized operations.
    """
    points = x.reshape(N, 3)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    points = points / norms

    n = len(points)
    # Compute squared distances efficiently
    # |x_i - x_j|^2 = |x_i|^2 + |x_j|^2 - 2*x_i.x_j = 2 - 2*x_i.x_j (since on unit sphere)
    dots = points @ points.T
    dists_sq = 2.0 - 2.0 * dots
    np.fill_diagonal(dists_sq, 1.0)
    # Clamp to avoid negative values from numerical error
    dists_sq = np.maximum(dists_sq, 1e-15)

    # Gradient of E = -sum_{i<j} log(d_ij) = -0.5 * sum_{i!=j} log(d_ij)
    # dE/dx_i = sum_{j!=i} (x_j - x_i) / d_ij^2
    # Using matrix operations for efficiency:
    # weights[i,j] = 1/d_ij^2 for i!=j, 0 otherwise
    # grads[i] = sum_j weights[i,j] x_j - x_i * sum_j weights[i,j]
    
    # Create weight matrix
    inv_dists_sq = np.zeros_like(dists_sq)
    mask = ~np.eye(n, dtype=bool)
    inv_dists_sq[mask] = 1.0 / dists_sq[mask]
    
    # sum_weights[i] = sum_j inv_dists_sq[i,j]
    sum_weights = np.sum(inv_dists_sq, axis=1)
    # weighted_sum[i] = sum_j inv_dists_sq[i,j] x_j
    weighted_sum = inv_dists_sq @ points
    
    # Raw Euclidean gradient
    grads = weighted_sum - points * sum_weights[:, np.newaxis]
    
    # Project gradient to tangent space
    # grad_tan = grad - (grad . p) p
    dot_products = np.sum(grads * points, axis=1, keepdims=True)
    grads_tangent = grads - dot_products * points

    # Compute energy for the optimizer (using fast np.sum for speed, consistency is key)
    iu = np.triu_indices(n, k=1)
    energy = -np.sum(np.log(np.sqrt(dists_sq[iu])))

    return energy, grads_tangent.flatten()

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

def _tangent_vector(point, rng):
    """Generate a random unit tangent vector at point on sphere."""
    rand_vec = rng.randn(3)
    tangent = rand_vec - np.dot(rand_vec, point) * point
    norm = np.linalg.norm(tangent)
    if norm < 1e-15:
        rand_vec = rng.randn(3)
        tangent = rand_vec - np.dot(rand_vec, point) * point
        norm = np.linalg.norm(tangent)
    return tangent / norm

def _local_minimize(points, deadline, maxiter=500, ftol=1e-12, gtol=1e-10):
    """Run L-BFGS-B to find local minimum with time monitoring."""
    x0 = points.flatten()
    
    # Check if we have time
    if time.time() > deadline - 1.0:
        return points, _log_energy(points)

    try:
        res = minimize(
            _energy_and_grad,
            x0,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': maxiter, 'maxfun': maxiter * 20, 'ftol': ftol, 'gtol': gtol, 'maxls': 40}
        )
        # Check time again
        if time.time() > deadline - 1.0:
             return points, _log_energy(points)
             
        p_res = _project_to_sphere(res.x.reshape(N, 3))
        # Use the precise energy function for comparison
        e_check = _log_energy(p_res)
        return p_res, e_check
    except Exception:
        return points, _log_energy(points)

def _basin_hopping_single(init_points, deadline, rng, max_iter=80, T_init=0.08):
    """
    Basin hopping with adaptive perturbation:
    1. Local minimize
    2. Perturb (adaptive step size based on recent success)
    3. Local minimize
    4. Accept if better (Metropolis)
    """
    current_points, current_energy = _local_minimize(init_points, deadline, maxiter=200)
    best_points = current_points.copy()
    best_energy = current_energy

    T = T_init
    T_min = 1e-6
    cooling = 0.97
    success_count = 0
    fail_count = 0
    base_perturb_size = 0.05

    for it in range(max_iter):
        if time.time() > deadline - 3.0:
            break
        if T < T_min:
            break

        # Adaptive perturbation: increase if failing, decrease if succeeding
        if success_count > fail_count:
            perturb_scale = 0.7
        elif fail_count > success_count:
            perturb_scale = 1.3
        else:
            perturb_scale = 1.0
        
        # Perturb a random subset of points
        n_perturb = rng.randint(3, 10)
        idxs = rng.choice(N, n_perturb, replace=False)

        new_points = current_points.copy()
        for idx in idxs:
            tangent = _tangent_vector(new_points[idx], rng)
            step_size = T * perturb_scale * (0.5 + rng.rand())
            new_point = new_points[idx] + step_size * tangent
            new_point = new_point / np.linalg.norm(new_point)
            new_points[idx] = new_point

        # Local minimize the perturbed configuration with fewer iterations for speed
        new_points, new_energy = _local_minimize(new_points, deadline, maxiter=80, ftol=1e-10, gtol=1e-8)

        # Metropolis criterion
        delta_e = new_energy - current_energy
        if delta_e < 0 or (T > 0 and rng.rand() < math.exp(-delta_e / T)):
            current_points = new_points
            current_energy = new_energy
            success_count += 1
            fail_count = max(0, fail_count - 1)
            if current_energy < best_energy:
                best_energy = current_energy
                best_points = current_points.copy()
        else:
            fail_count += 1
            success_count = max(0, success_count - 1)

        T *= cooling

    return best_points, best_energy

def _polish_high_precision(pts, max_time):
    """High-precision final polish with L-BFGS at very tight tolerances."""
    pts = _project_to_sphere(np.array(pts, dtype=np.float64))
    t0 = time.time()
    
    def obj(x):
        if time.time() - t0 > max_time:
            return np.inf
        p = x.reshape(N, 3)
        p = _project_to_sphere(p)
        e, _ = _energy_and_grad(p)
        return e
        
    def grad(x):
        if time.time() - t0 > max_time:
            return np.zeros_like(x)
        p = x.reshape(N, 3)
        p = _project_to_sphere(p)
        _, g = _energy_and_grad(p)
        return g

    x0 = pts.flatten()
    try:
        # First pass: moderate tolerance
        res1 = minimize(
            obj, x0, jac=grad, method='L-BFGS-B',
            options={'maxiter': 5000, 'ftol': 1e-14, 'gtol': 1e-11, 'maxfun': 20000}
        )
        pts_mid = _project_to_sphere(res1.x.reshape(N, 3))
        
        # Check time
        if time.time() - t0 > max_time * 0.6:
            e_opt = _log_energy(pts_mid)
            return pts_mid, e_opt
        
        # Second pass: very tight tolerance from improved starting point
        res2 = minimize(
            obj, pts_mid.flatten(), jac=grad, method='L-BFGS-B',
            options={'maxiter': 10000, 'ftol': 1e-16, 'gtol': 1e-13, 'maxfun': 40000}
        )
        pts_opt = _project_to_sphere(res2.x.reshape(N, 3))
        e_opt = _log_energy(pts_opt)
        return pts_opt, e_opt
    except Exception:
        return pts, _log_energy(pts)

def construct():
    t_start = time.time()
    deadline = t_start + _DEADLINE
    
    rng = np.random.RandomState(42)

    best_x = None
    best_e = float('inf')
    good_candidates = []

    # Phase 1: Diverse Initialization and Local Minimization
    # Generate diverse initializations
    inits = []
    
    # 1. Fibonacci sphere
    inits.append(_fibonacci_sphere(N))
    
    # 2. Random uniform points with different seeds
    for seed in range(8):
        r = np.random.RandomState(seed * 137 + 42)
        inits.append(_random_uniform_sphere(N, r))
        
    # 3. Perturbed Fibonacci with varying noise levels
    for k in range(4):
        noise_level = 0.05 + 0.05 * k
        perturbed = _fibonacci_sphere(N) + rng.randn(N, 3) * noise_level
        inits.append(_project_to_sphere(perturbed))
    
    # 4. Geodesic-like initializations: cluster points then spread
    for k in range(2):
        base = _fibonacci_sphere(N // 2)
        # Create two clusters and merge
        cluster1 = base + rng.randn(N // 2, 3) * 0.3
        cluster2 = -base + rng.randn(N // 2, 3) * 0.3
        combined = np.vstack([cluster1, cluster2])
        inits.append(_project_to_sphere(combined))

    # Run local minimization on each to find good local minima
    for i, init in enumerate(inits):
        if time.time() > deadline - 30.0:
            break
            
        remaining = deadline - time.time()
        if remaining < 5.0:
            break

        # Allocate time for this local minimization
        x_res, e_res = _local_minimize(init, deadline, maxiter=300, ftol=1e-10, gtol=1e-8)
        
        if e_res < best_e:
            best_e = e_res
            best_x = x_res
        
        # Keep track of good candidates for basin hopping
        if e_res < best_e + 5.0:
            good_candidates.append((e_res, x_res.copy()))

    # Phase 2: Basin Hopping from multiple good starting points
    if best_x is not None and time.time() < deadline - 15.0:
        # Sort candidates by energy
        good_candidates.sort(key=lambda x: x[0])
        
        # Run basin hopping from top candidates
        bh_count = 0
        for e_cand, x_cand in good_candidates[:4]:
            if time.time() > deadline - 12.0:
                break
            if bh_count >= 2 and time.time() > deadline - 25.0:
                break
                
            x_bh, e_bh = _basin_hopping_single(x_cand, deadline, rng, max_iter=60, T_init=0.06)
            if e_bh < best_e:
                best_e = e_bh
                best_x = x_bh
            bh_count += 1

    # Phase 3: Final High-Precision Polish
    if best_x is not None and time.time() < deadline - 5.0:
        polish_time = deadline - time.time() - 2.0 # 2s safety
        if polish_time > 2.0:
            x_final, e_final = _polish_high_precision(best_x, polish_time)
            if e_final < best_e:
                best_x = x_final
                best_e = e_final

    # Final validation and projection
    if best_x is None:
        best_x = _fibonacci_sphere(N)
        
    best_x = _project_to_sphere(best_x)
    
    # Ensure no points are too close
    # The problem states points closer than 1e-9 are invalid.
    # Our optimization should naturally avoid this, but let's check.
    diff = best_x[:, np.newaxis, :] - best_x[np.newaxis, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, 1.0)
    iu = np.triu_indices(N, 1)
    if dist[iu].min() < 1e-9:
        # Fallback to Fibonacci if points collapsed
        best_x = _fibonacci_sphere(N)
        best_x = _project_to_sphere(best_x)
        
    return best_x
# EVOLVE-BLOCK-END
