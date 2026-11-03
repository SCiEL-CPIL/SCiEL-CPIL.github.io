"""Combined program for logarithmic energy minimization of 150 points on sphere."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize


def _log_energy(points):
    """Compute log(1/d) energy for points on unit sphere using math.fsum for precision."""
    n = len(points)
    if n < 2:
        return 0.0
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=-1)
    # Avoid division by zero / log(0)
    np.fill_diagonal(dists_sq, 1.0)
    dists = np.sqrt(dists_sq)
    iu = np.triu_indices(n, k=1)
    log_dists = np.log(dists[iu])
    # Use math.fsum for high precision summation as required by the task contract
    return -math.fsum(log_dists.tolist())


def _energy_and_grad(x):
    """
    Energy and gradient for L-BFGS-B. 
    x is flat array of shape (N*3,).
    The function normalizes x to the sphere internally.
    The returned gradient is the tangential gradient, which is the correct 
    gradient for the unconstrained variable v if we assume v stays near the sphere.
    """
    points = x.reshape(N, 3)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    points = points / norms

    n = len(points)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(dists_sq, 1.0)

    # Gradient of E = -sum_{i<j} log(d_ij)
    # dE/dx_i = sum_{j!=i} (x_j - x_i) / d_ij^2  (Note: sign check)
    # E = - sum log d. dE/dd = -1/d. dd/dx_i = (x_i - x_j)/d. 
    # So dE/dx_i = sum_j (-1/d) * (x_i - x_j)/d = sum_j (x_j - x_i)/d^2.
    
    weights = np.zeros_like(dists_sq)
    mask = ~np.eye(n, dtype=bool)
    weights[mask] = 1.0 / dists_sq[mask]

    sum_weights = np.sum(weights, axis=1)
    weighted_sum = weights @ points

    # grads[i] = sum_j w_ij x_j - x_i sum_j w_ij
    grads = weighted_sum - points * sum_weights[:, np.newaxis]

    # Project gradient to tangent space
    dot_products = np.sum(grads * points, axis=1, keepdims=True)
    grads_tangent = grads - dot_products * points

    # Compute energy for the optimizer (using fast np.sum, not math.fsum, for speed)
    # The optimizer uses this to compare steps. Precision is less critical here 
    # as long as it's consistent.
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
    r = np.sqrt(1.0 - z * z)
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


def _local_minimize(points, deadline, maxiter=500):
    """Run L-BFGS-B to find local minimum."""
    x0 = points.flatten()
    try:
        res = minimize(
            _energy_and_grad,
            x0,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': maxiter, 'maxfun': maxiter * 20, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 40}
        )
        p_res = _project_to_sphere(res.x.reshape(N, 3))
        # Use the precise energy function for comparison
        e_check = _log_energy(p_res)
        return p_res, e_check
    except Exception:
        return points, _log_energy(points)


def _basin_hopping_single(init_points, deadline, rng, max_iter=50):
    """
    Simple basin hopping:
    1. Local minimize
    2. Perturb
    3. Local minimize
    4. Accept if better
    """
    current_points, current_energy = _local_minimize(init_points, deadline)
    best_points = current_points.copy()
    best_energy = current_energy

    T = 0.05
    T_min = 1e-8
    cooling = 0.95

    for it in range(max_iter):
        if time.time() > deadline - 5.0:
            break
        if T < T_min:
            break

        # Perturb a random subset of points
        n_perturb = rng.randint(2, 6)
        idxs = rng.choice(N, n_perturb, replace=False)

        new_points = current_points.copy()
        for idx in idxs:
            tangent = _tangent_vector(new_points[idx], rng)
            step_size = T
            new_point = new_points[idx] + step_size * tangent
            new_point = new_point / np.linalg.norm(new_point)
            new_points[idx] = new_point

        # Local minimize the perturbed configuration
        new_points, new_energy = _local_minimize(new_points, deadline, maxiter=200)

        # Metropolis criterion
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
    start_time = time.time()
    # Safety margin: 150s limit. Use 140s.
    deadline = start_time + 140.0 

    rng = np.random.RandomState(42)

    best_x = None
    best_e = float('inf')

    # Generate diverse initializations
    inits = []
    
    # 1. Fibonacci sphere
    inits.append(_fibonacci_sphere(N))
    
    # 2. Random uniform points
    for seed in range(5):
        r = np.random.RandomState(seed * 100 + 1)
        inits.append(_random_uniform_sphere(N, r))
        
    # 3. Perturbed Fibonacci
    for k in range(3):
        perturbed = _fibonacci_sphere(N) + rng.randn(N, 3) * 0.1
        inits.append(_project_to_sphere(perturbed))

    # Run basin hopping from each initialization
    for i, init in enumerate(inits):
        if time.time() > deadline - 10.0:
            break
            
        # Check time for this specific restart
        remaining = deadline - time.time()
        if remaining < 15.0:
            break

        # Run basin hopping
        # Limit max_iter based on remaining time? 
        # _basin_hopping_single has an internal time check.
        x_res, e_res = _basin_hopping_single(init, deadline, rng, max_iter=30)
        
        if e_res < best_e:
            best_e = e_res
            best_x = x_res

    # If we still have time, do a final polish on the best solution
    if best_x is not None and time.time() < deadline - 5.0:
        x_final, e_final = _local_minimize(best_x, deadline, maxiter=1000)
        if e_final < best_e:
            best_x = x_final
            best_e = e_final

    # Final validation and projection
    if best_x is None:
        best_x = _fibonacci_sphere(N)
        
    best_x = _project_to_sphere(best_x)
    
    # Ensure no points are too close
    min_dist = 1e-9
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_x[i] - best_x[j])
            if d < min_dist:
                # Nudge point j slightly
                best_x[j] += 1e-10 * rng.randn(3)
                best_x = _project_to_sphere(best_x)

    # Final check for non-finite values
    if not np.all(np.isfinite(best_x)):
        best_x = _fibonacci_sphere(N)
        best_x = _project_to_sphere(best_x)
        
    return best_x
# EVOLVE-BLOCK-END
