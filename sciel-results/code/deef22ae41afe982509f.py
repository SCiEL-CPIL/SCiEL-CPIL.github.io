"""Optimize 150 points on unit sphere to minimize logarithmic energy."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import math
import time
from scipy.optimize import minimize


def _energy_and_grad(pts):
    """Compute logarithmic energy and gradient (fast vectorized using dot products)."""
    n = len(pts)
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


def _tangent_project(pts, vecs):
    """Project vectors onto tangent space of sphere at each point."""
    # vecs shape (N, 3)
    # pts shape (N, 3)
    v_dot_n = np.sum(vecs * pts, axis=1, keepdims=True)
    return vecs - v_dot_n * pts


def _exp_map(pts, tangents):
    """Exponential map on the sphere.
    pts: (N, 3)
    tangents: (N, 3) tangent vectors
    Returns: (N, 3) points on sphere
    """
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    # Avoid division by zero
    safe_norms = np.where(norms > 1e-12, norms, 1.0)
    cos_theta = np.cos(norms)
    sin_theta = np.sin(norms)
    # x_new = cos(||v||) * x + sin(||v||) / ||v|| * v
    term1 = cos_theta * pts
    term2 = (sin_theta / safe_norms) * tangents
    # For very small norms, sin(t)/t ~ 1, so term2 ~ v
    # But if norm is 0, term2 is 0.
    result = term1 + term2
    return _project_to_sphere(result)


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


def _lbfgs_polish(pts, time_limit=2.0, ftol=1e-10, gtol=1e-8, maxiter=200):
    """L-BFGS-B optimization with strict time limit."""
    start_time = time.time()
    pts = _project_to_sphere(pts.copy())

    def objective_flat(x):
        if time.time() - start_time > time_limit:
            raise ValueError("Time limit exceeded")
        p = _project_to_sphere(x.reshape(-1, 3))
        e, _ = _energy_and_grad(p)
        return e

    def grad_flat(x):
        if time.time() - start_time > time_limit:
            raise ValueError("Time limit exceeded")
        p = _project_to_sphere(x.reshape(-1, 3))
        _, g = _energy_and_grad(p)
        g_tangent = _tangent_project(p, g)
        return g_tangent.flatten()

    x0 = pts.flatten()

    try:
        result = minimize(
            objective_flat,
            x0,
            jac=grad_flat,
            method='L-BFGS-B',
            options={
                'maxiter': maxiter,
                'maxfun': 1000,
                'ftol': ftol,
                'gtol': gtol,
                'maxls': 10
            }
        )
        final_pts = _project_to_sphere(result.x.reshape(-1, 3))
    except Exception:
        final_pts = pts

    energy, _ = _energy_and_grad(final_pts)
    return final_pts, energy


def _basin_hopping(pts, time_limit=30.0, seed=42):
    """Basin Hopping with L-BFGS polishing, strictly time-bounded."""
    start_time = time.time()
    current_pts = _project_to_sphere(pts.copy())
    current_energy, _ = _energy_and_grad(current_pts)
    best_pts = current_pts.copy()
    best_energy = current_energy

    rng = np.random.RandomState(seed)
    
    # Parameters
    kick_max = 0.4
    kick_min = 0.05
    # We don't know how many steps we can afford, so we loop until time runs out
    # but we cap the number of steps to avoid infinite loops if time is mocked/buggy
    max_steps = 100 

    for i in range(max_steps):
        elapsed = time.time() - start_time
        if elapsed > time_limit - 2.0:  # Leave 2s buffer
            break
            
        # Decrease kick size based on remaining time or progress
        # Simple decay: start big, end small
        kick_size = kick_max * (0.5 + 0.5 * (1.0 - elapsed / time_limit))
        
        # Generate random tangent perturbations
        v = rng.randn(N, 3) * kick_size
        v_tangent = _tangent_project(current_pts, v)
        
        # Apply exponential map
        new_pts = _exp_map(current_pts, v_tangent)
        
        # Polish with a very tight time limit
        # Estimate remaining time for this step
        remaining_for_step = min(1.5, time_limit - (time.time() - start_time) - 1.0)
        if remaining_for_step < 0.2:
            break
            
        try:
            polished_pts, polished_energy = _lbfgs_polish(new_pts, time_limit=remaining_for_step)
        except Exception:
            polished_pts = new_pts
            polished_energy, _ = _energy_and_grad(polished_pts)
        
        if polished_energy < current_energy:
            current_pts = polished_pts
            current_energy = polished_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_pts = current_pts.copy()

    return best_pts, best_energy


def construct():
    start_time = time.time()
    deadline = 110.0  # Safe margin before 120s kill

    rng = np.random.default_rng(42)

    starts = []

    # 1. Fibonacci spiral
    starts.append(_fibonacci_sphere(N))

    # 2. Perturbed Fibonacci (fewer starts to save time)
    for seed in [1, 7, 42]:
        fib = _fibonacci_sphere(N, seed=seed)
        noise = 0.1 * rng.standard_normal((N, 3))
        perturbed = fib + noise
        starts.append(_project_to_sphere(perturbed))

    # 3. Random starts
    for _ in range(2):
        starts.append(_random_sphere(N, rng))

    best_pts = None
    best_energy = np.inf

    # Phase 1: Optimize each start with L-BFGS-B
    for i, start_pts in enumerate(starts):
        elapsed = time.time() - start_time
        # Reserve 60s for Basin Hopping and final polish
        if elapsed > deadline - 60:
            break

        time_limit = min(2.0, deadline - elapsed - 60)
        if time_limit < 0.5:
            break

        try:
            candidate_pts, candidate_energy = _lbfgs_polish(start_pts, time_limit=time_limit)
            if candidate_energy < best_energy:
                best_energy = candidate_energy
                best_pts = candidate_pts
        except Exception:
            pass

    if best_pts is None:
        best_pts = _fibonacci_sphere(N)
        best_energy, _ = _energy_and_grad(best_pts)

    # Phase 2: Basin Hopping
    elapsed = time.time() - start_time
    if elapsed < deadline - 10:
        time_limit = min(80.0, deadline - elapsed - 10)
        if time_limit > 5.0:
            try:
                candidate_pts, candidate_energy = _basin_hopping(best_pts, time_limit=time_limit, seed=42)
                if candidate_energy < best_energy:
                    best_pts = candidate_pts
                    best_energy = candidate_energy
            except Exception:
                pass

    # Phase 3: Final high-precision polish
    elapsed = time.time() - start_time
    if elapsed < deadline - 2:
        time_limit = min(5.0, deadline - elapsed - 1)
        if time_limit > 1.0:
            try:
                candidate_pts, candidate_energy = _lbfgs_polish(best_pts, time_limit=time_limit, ftol=1e-12, gtol=1e-10)
                if candidate_energy < best_energy:
                    best_pts = candidate_pts
                    best_energy = candidate_energy
            except Exception:
                pass

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
