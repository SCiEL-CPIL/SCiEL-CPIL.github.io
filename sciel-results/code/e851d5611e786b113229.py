"""Optimize 150 points on unit sphere to minimize logarithmic energy."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import math
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

_DEADLINE = 138.0  # Safety margin before 150s kill


def _energy_precise(pts):
    """Compute logarithmic energy using math.fsum for precision."""
    n = len(pts)
    if n < 2:
        return 0.0
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(dists_sq, 1.0)
    dists = np.sqrt(dists_sq)
    iu = np.triu_indices(n, k=1)
    log_dists = np.log(dists[iu])
    return -math.fsum(log_dists.tolist())


def _energy_and_grad(pts):
    """Compute logarithmic energy and gradient (fast vectorized)."""
    n = len(pts)
    dist = cdist(pts, pts, 'euclidean')
    np.fill_diagonal(dist, 1.0)

    iu = np.triu_indices(n, k=1)
    dist_upper = dist[iu]
    dist_upper = dist_upper[dist_upper > 1e-15]
    if len(dist_upper) == 0:
        energy = 0.0
    else:
        energy = -np.sum(np.log(dist_upper))

    # Gradient: dE/dx_i = sum_{j!=i} (x_j - x_i) / |x_i - x_j|^2
    dist_sq = dist ** 2
    inv_dist_sq = np.zeros_like(dist_sq)
    mask = dist_sq > 1e-30
    inv_dist_sq[mask] = 1.0 / dist_sq[mask]
    np.fill_diagonal(inv_dist_sq, 0.0)

    row_sums = np.sum(inv_dist_sq, axis=1)
    weighted_sum = inv_dist_sq @ pts
    grad = weighted_sum - pts * row_sums[:, np.newaxis]

    return energy, grad


def _energy_only(pts):
    """Energy only, for fast inner loops."""
    n = len(pts)
    dist = cdist(pts, pts, 'euclidean')
    np.fill_diagonal(dist, 1.0)
    iu = np.triu_indices(n, k=1)
    dist_upper = dist[iu]
    dist_upper = dist_upper[dist_upper > 1e-15]
    if len(dist_upper) == 0:
        return 0.0
    return -np.sum(np.log(dist_upper))


def _project_to_sphere(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms


def _fibonacci_sphere(n, seed=None):
    if seed is not None:
        rng = np.random.RandomState(seed)
        offset = rng.uniform(0, 1)
    else:
        offset = 0.0
    i = np.arange(n) + 0.5 + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return pts


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


def _lbfgs_optimize(pts, time_limit=5.0, ftol=1e-12, gtol=1e-10, maxiter=2000):
    """L-BFGS-B optimization with time limit."""
    start_time = time.time()
    pts = _project_to_sphere(pts.copy())

    def objective_flat(x):
        if time.time() - start_time > time_limit:
            return np.inf
        p = _project_to_sphere(x.reshape(-1, 3))
        e, _ = _energy_and_grad(p)
        return e

    def grad_flat(x):
        if time.time() - start_time > time_limit:
            return np.zeros(len(x))
        p = _project_to_sphere(x.reshape(-1, 3))
        _, g = _energy_and_grad(p)
        g_tangent = g - np.sum(g * p, axis=1, keepdims=True) * p
        return g_tangent.flatten()

    x0 = pts.flatten()

    result = minimize(
        objective_flat,
        x0,
        jac=grad_flat,
        method='L-BFGS-B',
        options={
            'maxiter': maxiter,
            'maxfun': 5000,
            'ftol': ftol,
            'gtol': gtol,
            'maxls': 50
        }
    )

    final_pts = _project_to_sphere(result.x.reshape(-1, 3))
    energy, _ = _energy_and_grad(final_pts)
    return final_pts, energy


def _bb_gradient_descent(points, time_limit=10.0):
    """Barzilai-Borwein step size gradient descent on the sphere."""
    start_time = time.time()
    pts = _project_to_sphere(points.copy())
    n = len(pts)

    # Compute initial gradient
    _, grad = _energy_and_grad(pts)
    dot_products = np.sum(grad * pts, axis=1, keepdims=True)
    grad_tan = grad - dot_products * pts

    # Initial step size
    alpha = 0.01

    prev_grad_tan = grad_tan.copy()
    prev_pts = pts.copy()

    best_pts = pts.copy()
    best_energy, _ = _energy_and_grad(pts)

    no_improve = 0
    patience = 500

    for it in range(10000):
        if time.time() - start_time > time_limit:
            break

        # Compute BB step size
        # alpha_k = <x_k - x_{k-1}, g_k - g_{k-1}> / ||g_k - g_{k-1}||^2
        # or alpha_k = <x_k - x_{k-1}, g_k - g_{k-1}> / <g_k - g_{k-1}, g_k - g_{k-1}>
        # Use the second form (BB2) which is often more stable
        d_x = pts - prev_pts
        d_g = grad_tan - prev_grad_tan

        dg_norm_sq = np.sum(d_g * d_g)
        if dg_norm_sq > 1e-16:
            dx_dg = np.sum(d_x * d_g)
            alpha_bb = dx_dg / dg_norm_sq
            # Clip to reasonable range
            alpha_bb = np.clip(alpha_bb, 1e-6, 1.0)
            alpha = alpha_bb

        # Take step
        pts_new = pts - alpha * grad_tan
        pts_new = _project_to_sphere(pts_new)

        # Compute new gradient
        _, grad_new = _energy_and_grad(pts_new)
        dot_products = np.sum(grad_new * pts_new, axis=1, keepdims=True)
        grad_tan_new = grad_new - dot_products * pts_new

        # Check energy
        energy, _ = _energy_and_grad(pts_new)
        if energy < best_energy:
            best_energy = energy
            best_pts = pts_new.copy()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

        # If step increased energy significantly, reduce alpha
        if energy > best_energy + 1.0:
            alpha *= 0.5

        # Update for next iteration
        prev_pts = pts
        prev_grad_tan = grad_tan
        pts = pts_new
        grad_tan = grad_tan_new

        # Check convergence
        if np.linalg.norm(grad_tan) < 1e-12:
            break

    return best_pts, best_energy


def _basin_hopping(init_points, time_limit=15.0, seed=42):
    """
    Basin hopping:
    1. Local minimize
    2. Perturb
    3. Local minimize
    4. Accept if better (Metropolis)
    """
    start_time = time.time()
    rng = np.random.RandomState(seed)

    current_points, current_energy = _lbfgs_optimize(init_points, time_limit=min(3.0, time_limit * 0.2))
    best_points = current_points.copy()
    best_energy = current_energy

    T = 0.05
    T_min = 1e-8
    cooling = 0.95

    max_iter = 100

    for it in range(max_iter):
        if time.time() - start_time > time_limit - 2.0:
            break
        if T < T_min:
            break

        # Perturb a random subset of points
        n_perturb = rng.randint(3, 10)
        idxs = rng.choice(N, n_perturb, replace=False)

        new_points = current_points.copy()
        for idx in idxs:
            tangent = _tangent_vector(new_points[idx], rng)
            step_size = T * 0.5
            new_point = new_points[idx] + step_size * tangent
            new_point = new_point / np.linalg.norm(new_point)
            new_points[idx] = new_point

        # Local minimize the perturbed configuration
        remaining = time_limit - (time.time() - start_time)
        if remaining < 2.0:
            break
        time_limit_local = min(3.0, remaining - 1.0)
        new_points, new_energy = _lbfgs_optimize(new_points, time_limit=time_limit_local)

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


def _polish(pts, time_limit=5.0):
    """Final high-precision polish with BB step size + L-BFGS."""
    start_time = time.time()

    # First: BB gradient descent for fast convergence
    remaining = time_limit - (time.time() - start_time)
    if remaining > 2.0:
        pts, energy = _bb_gradient_descent(pts, time_limit=min(3.0, remaining - 1.0))

    # Then: L-BFGS with tight tolerances
    remaining = time_limit - (time.time() - start_time)
    if remaining > 1.0:
        pts, energy = _lbfgs_optimize(pts, time_limit=remaining - 0.5, ftol=1e-15, gtol=1e-12, maxiter=5000)

    pts = _project_to_sphere(pts)
    energy, _ = _energy_and_grad(pts)
    return pts, energy


def construct():
    start_time = time.time()
    deadline = _DEADLINE

    # Phase 1: Initial Fibonacci sphere + L-BFGS
    pts = _fibonacci_sphere(N)
    best_pts = pts.copy()
    best_energy, _ = _energy_and_grad(pts)

    elapsed = time.time() - start_time
    if elapsed < deadline - 40:
        time_limit = min(10.0, deadline - elapsed - 25)
        candidate_pts, candidate_energy = _lbfgs_optimize(pts, time_limit=time_limit)
        if candidate_energy < best_energy:
            best_pts = candidate_pts
            best_energy = candidate_energy

    # Phase 2: Basin hopping with multiple seeds for exploration
    elapsed = time.time() - start_time
    if elapsed < deadline - 30:
        for seed in [42, 123]:
            remaining = deadline - (time.time() - start_time) - 8
            if remaining < 5.0:
                break
            time_limit = min(12.0, remaining)
            candidate_pts, candidate_energy = _basin_hopping(best_pts, time_limit=time_limit, seed=seed)
            if candidate_energy < best_energy:
                best_pts = candidate_pts
                best_energy = candidate_energy

    # Phase 3: Random restart + L-BFGS for diversity
    elapsed = time.time() - start_time
    if elapsed < deadline - 20:
        rng = np.random.RandomState(999)
        random_pts = _random_uniform_sphere(N, rng)
        remaining = deadline - (time.time() - start_time) - 10
        if remaining > 3.0:
            time_limit = min(8.0, remaining)
            candidate_pts, candidate_energy = _lbfgs_optimize(random_pts, time_limit=time_limit)
            if candidate_energy < best_energy:
                best_pts = candidate_pts
                best_energy = candidate_energy

    # Phase 4: Final polish with BB + L-BFGS
    elapsed = time.time() - start_time
    if elapsed < deadline - 5:
        time_limit = min(8.0, deadline - elapsed - 2)
        candidate_pts, candidate_energy = _polish(best_pts, time_limit=time_limit)
        if candidate_energy < best_energy:
            best_pts = candidate_pts
            best_energy = candidate_energy

    best_pts = _project_to_sphere(best_pts)

    # Validate: check for very close points
    diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    min_dist = np.min(dist)

    if min_dist < 1e-9:
        return _fibonacci_sphere(N)

    return best_pts
# EVOLVE-BLOCK-END
