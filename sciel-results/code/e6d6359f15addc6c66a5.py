"""Optimize 150 points on unit sphere to minimize logarithmic energy."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import math
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist


def _energy_and_grad(pts):
    """Compute logarithmic energy and gradient using squared distances (faster)."""
    n = len(pts)
    # Use squared distances to avoid sqrt
    dist_sq = np.sum((pts[:, np.newaxis, :] - pts[np.newaxis, :, :]) ** 2, axis=2)
    np.fill_diagonal(dist_sq, 1.0)

    iu = np.triu_indices(n, k=1)
    dist_sq_upper = dist_sq[iu]
    dist_sq_upper = dist_sq_upper[dist_sq_upper > 1e-30]
    if len(dist_sq_upper) == 0:
        energy = 0.0
    else:
        # log(1/d) = -log(d) = -0.5*log(d^2)
        energy = -0.5 * np.sum(np.log(dist_sq_upper))

    # Gradient: dE/dx_i = sum_{j!=i} (x_j - x_i) / |x_i - x_j|^2
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
    dist_sq = np.sum((pts[:, np.newaxis, :] - pts[np.newaxis, :, :]) ** 2, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    iu = np.triu_indices(n, k=1)
    dist_sq_upper = dist_sq[iu]
    dist_sq_upper = dist_sq_upper[dist_sq_upper > 1e-30]
    if len(dist_sq_upper) == 0:
        return 0.0
    return -0.5 * np.sum(np.log(dist_sq_upper))


def _project_to_sphere(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms


def _tangent_project(pts, grad):
    """Project gradient onto tangent space of sphere at each point."""
    g_dot_n = np.sum(grad * pts, axis=1, keepdims=True)
    return grad - g_dot_n * pts


def _exp_map(pts, tangent):
    """Exponential map: move points along geodesics from pts in direction tangent."""
    norms = np.linalg.norm(tangent, axis=1, keepdims=True)
    norms_safe = np.where(norms < 1e-15, 1.0, norms)
    sin_norms = np.sin(norms_safe)
    cos_norms = np.cos(norms_safe)
    # Avoid division by zero
    factor = np.where(norms < 1e-15, 1.0, sin_norms / norms_safe)
    return cos_norms * pts + factor * tangent


def _log_map(pts, new_pts):
    """Logarithmic map: get tangent vector from pts to new_pts."""
    cos_theta = np.sum(pts * new_pts, axis=1, keepdims=True)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    theta_safe = np.where(np.abs(theta) < 1e-15, 1.0, theta)
    sin_theta = np.sin(theta_safe)
    factor = np.where(np.abs(theta) < 1e-15, 1.0, theta_safe / sin_theta)
    return factor * (new_pts - cos_theta * pts)


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


def _icosahedron_based(n):
    """Generate points using icosahedron structure for better initial distribution."""
    t = (1.0 + np.sqrt(5.0)) / 2.0
    ico = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
    ], dtype=float)
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)

    fib = _fibonacci_sphere(n)
    result = fib.copy()
    for i in range(12):
        dists = np.linalg.norm(fib - ico[i], axis=1)
        closest = np.argmin(dists)
        result[closest] = ico[i]
    return _project_to_sphere(result)


def _random_sphere(n, rng):
    """Generate random points on unit sphere."""
    pts = rng.standard_normal((n, 3))
    return _project_to_sphere(pts)


def _lbfgs_optimize(pts, time_limit=5.0, ftol=1e-14, gtol=1e-12, maxiter=3000):
    """L-BFGS-B optimization with tight tolerances."""
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
        g_tangent = _tangent_project(p, g)
        return g_tangent.flatten()

    x0 = pts.flatten()

    result = minimize(
        objective_flat,
        x0,
        jac=grad_flat,
        method='L-BFGS-B',
        options={
            'maxiter': maxiter,
            'maxfun': 8000,
            'ftol': ftol,
            'gtol': gtol,
            'maxls': 80
        }
    )

    final_pts = _project_to_sphere(result.x.reshape(-1, 3))
    energy, _ = _energy_and_grad(final_pts)
    return final_pts, energy


def _riemannian_cg(pts, time_limit=8.0):
    """Riemannian conjugate gradient on the sphere."""
    start_time = time.time()
    pts = _project_to_sphere(pts.copy())
    n = len(pts)

    best_pts = pts.copy()
    best_energy, _ = _energy_and_grad(pts)

    # Initial gradient
    _, grad = _energy_and_grad(pts)
    g_tan = _tangent_project(pts, grad)

    # CG state
    p_tan = -g_tan.copy()
    beta = 0.0

    for it in range(5000):
        if time.time() - start_time > time_limit:
            break

        # Line search (backtracking)
        alpha = 0.01
        for _ in range(30):
            new_pts = _exp_map(pts, alpha * p_tan)
            new_energy, new_grad = _energy_and_grad(new_pts)
            if new_energy < best_energy:
                best_energy = new_energy
                best_pts = new_pts.copy()

            # Check Armijo condition
            if new_energy <= best_energy + 1e-4 * alpha * np.sum(g_tan * p_tan):
                break
            alpha *= 0.5
        else:
            # If line search fails, try smaller step
            alpha = 0.001
            new_pts = _exp_map(pts, alpha * p_tan)
            new_energy, new_grad = _energy_and_grad(new_pts)
            if new_energy < best_energy:
                best_energy = new_energy
                best_pts = new_pts.copy()

        # Update
        pts = new_pts
        g_tan_new = _tangent_project(pts, new_grad)

        # Check convergence
        if np.linalg.norm(g_tan_new) < 1e-12:
            break

        # CG update
        beta = np.sum(g_tan_new * g_tan_new) / max(np.sum(g_tan * g_tan), 1e-30)
        p_tan = -g_tan_new + beta * p_tan
        g_tan = g_tan_new

    return best_pts, best_energy


def _simulated_annealing(pts, time_limit=10.0, seed=42):
    """Simulated annealing to escape local minima with adaptive perturbations."""
    start_time = time.time()
    pts = _project_to_sphere(pts.copy())
    n = len(pts)

    current_energy, _ = _energy_and_grad(pts)
    best_pts = pts.copy()
    best_energy = current_energy

    T = 0.05
    T_min = 1e-7
    cooling = 0.997
    max_steps = 8000

    rng = np.random.RandomState(seed)

    for step in range(max_steps):
        if time.time() - start_time > time_limit:
            break

        # Adaptive perturbation scale
        noise_scale = T * 0.15
        # Move random subset of points for larger perturbations
        if step % 10 == 0 and T > 0.01:
            num_move = max(3, n // 10)
            idx = rng.choice(n, size=num_move, replace=False)
            perturbation = np.zeros((n, 3))
            perturbation[idx] = rng.randn(num_move, 3) * noise_scale * 2
        else:
            perturbation = rng.randn(n, 3) * noise_scale

        new_pts = pts + perturbation
        new_pts = _project_to_sphere(new_pts)

        new_energy = _energy_only(new_pts)

        if new_energy < current_energy or (T > T_min and rng.rand() < math.exp(-(new_energy - current_energy) / T)):
            pts = new_pts
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_pts = pts.copy()

        T *= cooling
        if T < T_min:
            break

    return best_pts, best_energy


def _polish(pts, time_limit=5.0):
    """Final high-precision polish with very tight tolerances."""
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
        g_tangent = _tangent_project(p, g)
        return g_tangent.flatten()

    x0 = pts.flatten()

    result = minimize(
        objective_flat,
        x0,
        jac=grad_flat,
        method='L-BFGS-B',
        options={
            'maxiter': 8000,
            'maxfun': 15000,
            'ftol': 1e-16,
            'gtol': 1e-14,
            'maxls': 100
        }
    )

    final_pts = _project_to_sphere(result.x.reshape(-1, 3))
    energy, _ = _energy_and_grad(final_pts)
    return final_pts, energy


def _momentum_gradient_descent(pts0, max_time):
    """Optimize using projected gradient descent with momentum and adaptive step size."""
    start_time = time.time()

    pts = _project_to_sphere(pts0.copy())
    e, g = _energy_and_grad(pts)
    best_pts = pts.copy()
    best_e = e

    velocity = np.zeros_like(pts)
    momentum = 0.9
    step_size = 0.01

    for iteration in range(100000):
        if time.time() - start_time >= max_time:
            break

        g_tan = _tangent_project(pts, g)
        g_norm = np.linalg.norm(g_tan)
        if g_norm < 1e-11:
            break

        velocity = momentum * velocity - step_size * g_tan
        new_pts = pts + velocity
        new_pts = _project_to_sphere(new_pts)
        new_e, new_g = _energy_and_grad(new_pts)

        if new_e > e + 1e-8:
            for _ in range(10):
                step_size *= 0.5
                velocity = momentum * velocity - step_size * g_tan
                new_pts = pts + velocity
                new_pts = _project_to_sphere(new_pts)
                new_e, new_g = _energy_and_grad(new_pts)
                if new_e <= e + 1e-8:
                    break
            else:
                step_size *= 0.1
                continue

        pts = new_pts
        e = new_e
        g = new_g

        if e < best_e:
            best_e = e
            best_pts = pts.copy()

        if iteration > 100 and step_size < 0.05:
            step_size *= 1.05

    return best_pts, best_e


def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin from 150s limit

    rng = np.random.default_rng(42)

    # Generate diverse starting points
    starts = []

    # 1. Fibonacci spiral
    starts.append(_fibonacci_sphere(N))

    # 2. Icosahedron-based
    starts.append(_icosahedron_based(N))

    # 3. Perturbed Fibonacci (multiple seeds)
    for seed in [1, 7, 13, 23, 37]:
        fib = _fibonacci_sphere(N, seed=seed)
        noise = 0.03 * rng.standard_normal((N, 3))
        perturbed = fib + noise
        starts.append(_project_to_sphere(perturbed))

    # 4. Random starts
    for _ in range(4):
        starts.append(_random_sphere(N, rng))

    best_pts = None
    best_energy = np.inf

    # Phase 1: Optimize each start with L-BFGS-B, keep best
    for i, start_pts in enumerate(starts):
        elapsed = time.time() - start_time
        if elapsed > deadline - 25:
            break

        time_limit = min(6.0, deadline - elapsed - 20)
        if time_limit < 2.0:
            break

        candidate_pts, candidate_energy = _lbfgs_optimize(start_pts, time_limit=time_limit)
        if candidate_energy < best_energy:
            best_energy = candidate_energy
            best_pts = candidate_pts

    # Phase 2: Riemannian CG on the best
    elapsed = time.time() - start_time
    if elapsed < deadline - 30:
        time_limit = min(10.0, deadline - elapsed - 15)
        candidate_pts, candidate_energy = _riemannian_cg(best_pts, time_limit=time_limit)
        if candidate_energy < best_energy:
            best_pts = candidate_pts
            best_energy = candidate_energy

    # Phase 3: Simulated annealing with multiple seeds
    elapsed = time.time() - start_time
    if elapsed < deadline - 20:
        for seed in [42, 123, 789, 555]:
            remaining = deadline - (time.time() - start_time) - 5
            if remaining < 3.0:
                break
            time_limit = min(6.0, remaining)
            candidate_pts, candidate_energy = _simulated_annealing(best_pts, time_limit=time_limit, seed=seed)
            if candidate_energy < best_energy:
                best_pts = candidate_pts
                best_energy = candidate_energy

    # Phase 4: L-BFGS-B re-polish after SA
    elapsed = time.time() - start_time
    if elapsed < deadline - 12:
        time_limit = min(6.0, deadline - elapsed - 8)
        candidate_pts, candidate_energy = _lbfgs_optimize(best_pts, time_limit=time_limit)
        if candidate_energy < best_energy:
            best_pts = candidate_pts
            best_energy = candidate_energy

    # Phase 5: Riemannian CG for precise convergence
    elapsed = time.time() - start_time
    if elapsed < deadline - 8:
        time_limit = min(5.0, deadline - elapsed - 4)
        candidate_pts, candidate_energy = _riemannian_cg(best_pts, time_limit=time_limit)
        if candidate_energy < best_energy:
            best_pts = candidate_pts
            best_energy = candidate_energy

    # Phase 6: Final high-precision polish
    elapsed = time.time() - start_time
    if elapsed < deadline - 2:
        time_limit = min(4.0, deadline - elapsed - 1)
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
