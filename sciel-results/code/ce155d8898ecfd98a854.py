"""Optimize 150 points on unit sphere to minimize logarithmic energy."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import math
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist


def _energy_and_grad(pts):
    """Compute logarithmic energy and gradient (fast vectorized using dot products)."""
    n = len(pts)
    # For unit vectors: ||x_i - x_j||^2 = 2 - 2*x_i.x_j
    dot = pts @ pts.T
    dist_sq = 2.0 - 2.0 * dot
    np.fill_diagonal(dist_sq, 1.0)
    dist_sq = np.maximum(dist_sq, 1e-15)
    dist = np.sqrt(dist_sq)

    iu = np.triu_indices(n, k=1)
    dist_upper = dist[iu]
    dist_upper = np.maximum(dist_upper, 1e-15)
    energy = -np.sum(np.log(dist_upper))

    inv_dist_sq = np.zeros_like(dist_sq)
    mask = dist_sq > 1e-30
    inv_dist_sq[mask] = 1.0 / dist_sq[mask]
    np.fill_diagonal(inv_dist_sq, 0.0)

    weighted_sum = inv_dist_sq @ pts
    row_sums = np.sum(inv_dist_sq, axis=1)
    grad = weighted_sum - pts * row_sums[:, np.newaxis]

    return energy, grad


def _energy_only(pts):
    """Energy only, for fast inner loops."""
    n = len(pts)
    dot = pts @ pts.T
    dist_sq = 2.0 - 2.0 * dot
    np.fill_diagonal(dist_sq, 1.0)
    dist_sq = np.maximum(dist_sq, 1e-15)
    dist = np.sqrt(dist_sq)
    iu = np.triu_indices(n, k=1)
    dist_upper = dist[iu]
    dist_upper = np.maximum(dist_upper, 1e-15)
    return -np.sum(np.log(dist_upper))


def _project_to_sphere(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms


def _tangent_project(pts, grad):
    """Project gradient onto tangent space of sphere at each point."""
    g_dot_n = np.sum(grad * pts, axis=1, keepdims=True)
    return grad - g_dot_n * pts


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


def _random_sphere(n, rng):
    """Generate random points on unit sphere."""
    pts = rng.standard_normal((n, 3))
    return _project_to_sphere(pts)


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
            'maxfun': 5000,
            'ftol': ftol,
            'gtol': gtol,
            'maxls': 50
        }
    )

    final_pts = _project_to_sphere(result.x.reshape(-1, 3))
    energy, _ = _energy_and_grad(final_pts)
    return final_pts, energy


def _adam_optimize(pts, time_limit=10.0):
    """Adam optimizer on the sphere with adaptive learning rate."""
    start_time = time.time()
    pts = _project_to_sphere(pts.copy())
    n = len(pts)

    lr_initial = 0.03
    lr_min = 0.0005
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    lr_decay = 0.998

    m = np.zeros_like(pts)
    v = np.zeros_like(pts)
    t = 0
    lr = lr_initial

    best_pts = pts.copy()
    best_energy, _ = _energy_and_grad(pts)

    no_improve_count = 0
    patience = 800

    while time.time() - start_time < time_limit:
        t += 1
        energy, grad = _energy_and_grad(pts)

        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count >= patience:
                break

        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad ** 2)

        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)

        grad_tangent = _tangent_project(pts, grad)
        m_tangent = _tangent_project(pts, m_hat)

        pts = pts - lr * m_tangent / (np.sqrt(v_hat) + eps)
        pts = _project_to_sphere(pts)

        lr = max(lr_min, lr * lr_decay)

        if t % 100 == 0 and np.linalg.norm(grad_tangent) < 1e-10:
            break

    return best_pts, best_energy


def _anneal_with_polish(pts, time_limit=30.0, seed=42):
    """Simulated Annealing with periodic L-BFGS-B polishing."""
    start_time = time.time()
    pts = _project_to_sphere(pts.copy())
    n = len(pts)

    current_energy, _ = _energy_and_grad(pts)
    best_pts = pts.copy()
    best_energy = current_energy

    T = 0.15
    T_min = 1e-6
    cooling = 0.995
    max_steps = 10000
    polish_interval = 80

    rng = np.random.RandomState(seed)

    for step in range(max_steps):
        if time.time() - start_time > time_limit:
            break

        noise_scale = T * 0.8
        # Perturb a random subset of points for larger basin hopping
        if step % 3 == 0:
            n_perturb = max(5, n // 5)
            indices = rng.choice(n, size=n_perturb, replace=False)
            perturbation = np.zeros((n, 3))
            perturbation[indices] = rng.randn(n_perturb, 3) * noise_scale
        else:
            perturbation = rng.randn(n, 3) * noise_scale * 0.3
        new_pts = pts + perturbation
        new_pts = _project_to_sphere(new_pts)

        new_energy = _energy_only(new_pts)

        if new_energy < current_energy or rng.rand() < math.exp(-(new_energy - current_energy) / T):
            pts = new_pts
            current_energy = new_energy

            if current_energy < best_energy:
                best_energy = current_energy
                best_pts = pts.copy()

        T *= cooling
        if T < T_min:
            T = T_min

        if step % polish_interval == 0 and step > 0:
            remaining = time_limit - (time.time() - start_time)
            if remaining > 2.0:
                polished_pts, polished_energy = _lbfgs_optimize(
                    pts,
                    time_limit=min(2.0, remaining - 1.0),
                    maxiter=200
                )
                if polished_energy < current_energy:
                    pts = polished_pts
                    current_energy = polished_energy
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_pts = pts.copy()

    return best_pts, best_energy


def _final_polish(pts, time_limit=5.0):
    """High-precision final polish with tight tolerances and more iterations."""
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
            'maxiter': 10000,
            'maxfun': 20000,
            'ftol': 1e-15,
            'gtol': 1e-14,
            'maxls': 100
        }
    )

    final_pts = _project_to_sphere(result.x.reshape(-1, 3))
    energy, _ = _energy_and_grad(final_pts)
    return final_pts, energy


def construct():
    start_time = time.time()
    deadline = 145.0

    rng = np.random.default_rng(42)

    starts = []

    # 1. Fibonacci spiral
    starts.append(_fibonacci_sphere(N))

    # 2. Perturbed Fibonacci
    for seed in [1, 7, 13, 42, 123]:
        fib = _fibonacci_sphere(N, seed=seed)
        noise = 0.1 * rng.standard_normal((N, 3))
        perturbed = fib + noise
        starts.append(_project_to_sphere(perturbed))

    # 3. Random starts
    for _ in range(3):
        starts.append(_random_sphere(N, rng))

    best_pts = None
    best_energy = np.inf

    # Phase 1: Optimize each start with L-BFGS-B
    for i, start_pts in enumerate(starts):
        elapsed = time.time() - start_time
        if elapsed > deadline - 30:
            break

        time_limit = min(5.0, deadline - elapsed - 20)
        if time_limit < 1.0:
            break

        candidate_pts, candidate_energy = _lbfgs_optimize(start_pts, time_limit=time_limit)
        if candidate_energy < best_energy:
            best_energy = candidate_energy
            best_pts = candidate_pts

    # Phase 2: Adam optimization on the best
    elapsed = time.time() - start_time
    if elapsed < deadline - 25:
        time_limit = min(10.0, deadline - elapsed - 15)
        if time_limit > 1.0:
            candidate_pts, candidate_energy = _adam_optimize(best_pts, time_limit=time_limit)
            if candidate_energy < best_energy:
                best_pts = candidate_pts
                best_energy = candidate_energy

    # Phase 3: Annealing with Polish
    elapsed = time.time() - start_time
    if elapsed < deadline - 20:
        time_limit = min(25.0, deadline - elapsed - 10)
        if time_limit > 2.0:
            candidate_pts, candidate_energy = _anneal_with_polish(best_pts, time_limit=time_limit, seed=42)
            if candidate_energy < best_energy:
                best_pts = candidate_pts
                best_energy = candidate_energy

    # Phase 4: Another round of annealing with different seed
    elapsed = time.time() - start_time
    if elapsed < deadline - 15:
        time_limit = min(15.0, deadline - elapsed - 8)
        if time_limit > 2.0:
            candidate_pts, candidate_energy = _anneal_with_polish(best_pts, time_limit=time_limit, seed=123)
            if candidate_energy < best_energy:
                best_pts = candidate_pts
                best_energy = candidate_energy

    # Phase 5: Final high-precision polish
    elapsed = time.time() - start_time
    if elapsed < deadline - 3:
        time_limit = min(5.0, deadline - elapsed - 1)
        if time_limit > 1.0:
            candidate_pts, candidate_energy = _final_polish(best_pts, time_limit=time_limit)
            if candidate_energy < best_energy:
                best_pts = candidate_pts
                best_energy = candidate_energy

    # Phase 6: Second final polish if time permits
    elapsed = time.time() - start_time
    if elapsed < deadline - 2:
        time_limit = min(3.0, deadline - elapsed - 0.5)
        if time_limit > 0.5:
            candidate_pts, candidate_energy = _final_polish(best_pts, time_limit=time_limit)
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
