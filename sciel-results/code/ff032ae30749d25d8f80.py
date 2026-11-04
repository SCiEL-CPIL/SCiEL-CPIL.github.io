"""Optimize 150 points on unit sphere to minimize logarithmic energy."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import math
import time
from scipy.optimize import minimize


def _energy_and_grad(pts):
    """Compute log energy sum_{i<j} log(1/d) and its gradient.
    
    dE/dx_i = -sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    """
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # diff[i,j] = x_i - x_j
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0)

    mask = dist_sq > 1e-30
    inv_dist_sq = np.zeros_like(dist_sq)
    inv_dist_sq[mask] = 1.0 / dist_sq[mask]
    np.fill_diagonal(inv_dist_sq, 0.0)

    # True gradient: dE/dx_i = -sum_{j!=i} (x_i - x_j)/d^2
    grad = -np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)

    # Energy
    iu = np.triu_indices(n, k=1)
    d = np.sqrt(dist_sq[iu])
    d = d[d > 1e-15]
    if len(d) == 0:
        energy = 0.0
    else:
        energy = math.fsum(np.log(1.0 / d).tolist())

    return energy, grad


def _energy_only(pts):
    """Compute log energy only (faster, no gradient)."""
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    iu = np.triu_indices(n, k=1)
    d = np.sqrt(dist_sq[iu])
    d = d[d > 1e-15]
    if len(d) == 0:
        return 0.0
    return math.fsum(np.log(1.0 / d).tolist())


def _project_to_sphere(pts):
    """Project points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms


def _tangent_project(grad, pts):
    """Project gradient to tangent space of sphere."""
    return grad - np.sum(grad * pts, axis=1, keepdims=True) * pts


def _exp_map(pts, vecs):
    """Exponential map on the unit sphere.
    pts: (N, 3)
    vecs: (N, 3) tangent vectors
    """
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    # Avoid division by zero
    safe_norms = np.where(norms < 1e-10, 1.0, norms)
    theta = norms
    # exp_map(x, v) = x * cos(theta) + (v / ||v||) * sin(theta)
    # If theta is small, use series expansion or just standard formula
    # sin(theta) / theta -> 1 for small theta
    factor = np.where(norms < 1e-10, 1.0, np.sin(safe_norms) / safe_norms)
    cos_factor = np.cos(safe_norms)
    
    # Handle zero vectors
    cos_factor_flat = cos_factor.flatten()
    factor_flat = factor.flatten()
    
    # x * cos(theta)
    term1 = pts * cos_factor
    # (v / ||v||) * sin(theta) = v * (sin(theta)/||v||)
    term2 = vecs * factor
    
    new_pts = term1 + term2
    return _project_to_sphere(new_pts)


def _fibonacci_sphere(n, seed=None):
    """Generate Fibonacci spiral points on sphere."""
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


def _random_sphere(n, seed):
    """Generate random points on sphere."""
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    return _project_to_sphere(pts)


def _lbfgs_optimize(pts, time_limit=5.0, tight=False):
    """Run L-BFGS-B optimization with time limit."""
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
        g_tangent = _tangent_project(g, p)
        return g_tangent.flatten()

    x0 = pts.flatten()

    if tight:
        options = {
            'maxiter': 10000,
            'maxfun': 30000,
            'ftol': 1e-16,
            'gtol': 1e-14,
            'maxls': 200
        }
    else:
        options = {
            'maxiter': 3000,
            'maxfun': 8000,
            'ftol': 1e-14,
            'gtol': 1e-12,
            'maxls': 50
        }

    result = minimize(
        objective_flat,
        x0,
        jac=grad_flat,
        method='L-BFGS-B',
        options=options
    )

    final_pts = _project_to_sphere(result.x.reshape(-1, 3))
    energy, _ = _energy_and_grad(final_pts)
    return final_pts, energy


def _riemannian_adam(pts, time_limit=10.0, lr=0.05):
    """Riemannian Adam optimizer using exponential map for updates."""
    start_time = time.time()
    pts = _project_to_sphere(pts.copy())
    n = len(pts)

    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8

    m = np.zeros_like(pts)
    v = np.zeros_like(pts)
    t = 0

    best_pts = pts.copy()
    best_energy, _ = _energy_and_grad(pts)

    while time.time() - start_time < time_limit:
        t += 1
        energy, grad = _energy_and_grad(pts)

        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()

        # Update momentum
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad ** 2)

        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)

        # Project gradient to tangent space
        m_tangent = _tangent_project(m_hat, pts)

        # Update step in tangent space
        step = lr * m_tangent / (np.sqrt(v_hat) + eps)
        
        # Apply exponential map
        pts = _exp_map(pts, -step)

        # Check convergence
        if t % 200 == 0:
            g_tangent = _tangent_project(grad, pts)
            if np.linalg.norm(g_tangent) < 1e-11:
                break

    return best_pts, best_energy


def _riemannian_sa(pts, time_limit=15.0, seed=42):
    """Riemannian Simulated Annealing using exponential map for perturbations."""
    start_time = time.time()
    pts = _project_to_sphere(pts.copy())
    n = len(pts)

    current_energy, _ = _energy_and_grad(pts)
    best_pts = pts.copy()
    best_energy = current_energy

    T = 0.05
    T_min = 1e-7
    cooling = 0.998
    max_steps = 20000

    rng = np.random.RandomState(seed)

    for step in range(max_steps):
        if time.time() - start_time > time_limit:
            break

        # Generate random perturbation in tangent space
        # Scale by temperature
        if rng.rand() < 0.05:
            noise_scale = T * 0.5
        else:
            noise_scale = T * 0.1
            
        # Random tangent vectors
        random_vecs = rng.randn(n, 3) * noise_scale
        random_vecs = _tangent_project(random_vecs, pts)
        
        new_pts = _exp_map(pts, random_vecs)
        new_energy = _energy_only(new_pts)

        # Accept or reject
        if new_energy < current_energy or (T > T_min and rng.rand() < math.exp(-(new_energy - current_energy) / T)):
            pts = new_pts
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_pts = pts.copy()

        # Cool down
        T *= cooling
        if T < T_min:
            break

    return best_pts, best_energy


def _basin_hopping(pts, time_limit=20.0, seed=42):
    """Basin hopping with Riemannian perturbations and L-BFGS refinement."""
    start_time = time.time()
    pts = _project_to_sphere(pts.copy())
    n = len(pts)

    current_energy, _ = _energy_and_grad(pts)
    best_pts = pts.copy()
    best_energy = current_energy

    T = 0.03
    T_min = 1e-6
    cooling = 0.997
    max_steps = 5000

    rng = np.random.RandomState(seed)

    for step in range(max_steps):
        if time.time() - start_time > time_limit:
            break

        # Generate Riemannian perturbation
        if rng.rand() < 0.1:
            noise_scale = T * 0.3
        else:
            noise_scale = T * 0.05
            
        random_vecs = rng.randn(n, 3) * noise_scale
        random_vecs = _tangent_project(random_vecs, pts)
        new_pts = _exp_map(pts, random_vecs)

        # Quick L-BFGS polish from perturbed point
        elapsed = time.time() - start_time
        remaining = time_limit - elapsed
        if remaining > 3.0:
            polish_limit = min(2.0, remaining * 0.3)
            polished_pts, polished_energy = _lbfgs_optimize(new_pts, time_limit=polish_limit)
        else:
            polished_pts = new_pts
            polished_energy = _energy_only(new_pts)

        # Accept or reject based on polished energy
        if polished_energy < current_energy or (T > T_min and rng.rand() < math.exp(-(polished_energy - current_energy) / T)):
            pts = polished_pts
            current_energy = polished_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_pts = pts.copy()

        # Cool down
        T *= cooling
        if T < T_min:
            break

    return best_pts, best_energy


def _final_polish(pts, time_limit=5.0):
    """High-precision final polish with tight tolerances."""
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
        g_tangent = _tangent_project(g, p)
        return g_tangent.flatten()

    x0 = pts.flatten()

    result = minimize(
        objective_flat,
        x0,
        jac=grad_flat,
        method='L-BFGS-B',
        options={
            'maxiter': 10000,
            'maxfun': 30000,
            'ftol': 1e-16,
            'gtol': 1e-14,
            'maxls': 200
        }
    )

    final_pts = _project_to_sphere(result.x.reshape(-1, 3))
    energy, _ = _energy_and_grad(final_pts)
    return final_pts, energy


def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin (evaluator kills at 150s)

    best_pts = None
    best_energy = np.inf

    # Generate diverse starting points
    starts = []
    starts.append(_fibonacci_sphere(N))
    starts.append(_fibonacci_sphere(N, seed=1))
    starts.append(_fibonacci_sphere(N, seed=7))
    for s in range(2, 8):
        starts.append(_random_sphere(N, seed=s * 100))

    # Phase 1: L-BFGS-B from each diverse start
    for i, start_pts in enumerate(starts):
        elapsed = time.time() - start_time
        if elapsed > deadline - 30:
            break
        time_limit = min(6.0, deadline - elapsed - 25)
        if time_limit < 2.0:
            break
        candidate_pts, candidate_energy = _lbfgs_optimize(start_pts, time_limit=time_limit)
        if candidate_energy < best_energy:
            best_energy = candidate_energy
            best_pts = candidate_pts.copy()

    # Phase 2: Riemannian Adam optimization from best point
    if best_pts is not None:
        elapsed = time.time() - start_time
        if elapsed < deadline - 30:
            time_limit = min(8.0, deadline - elapsed - 25)
            candidate_pts, candidate_energy = _riemannian_adam(best_pts, time_limit=time_limit)
            if candidate_energy < best_energy:
                best_energy = candidate_energy
                best_pts = candidate_pts.copy()

    # Phase 3: Riemannian Basin hopping to escape local minima
    if best_pts is not None:
        elapsed = time.time() - start_time
        if elapsed < deadline - 25:
            time_limit = min(15.0, deadline - elapsed - 15)
            candidate_pts, candidate_energy = _basin_hopping(
                best_pts, time_limit=time_limit, seed=42
            )
            if candidate_energy < best_energy:
                best_energy = candidate_energy
                best_pts = candidate_pts.copy()

    # Phase 4: Another Riemannian basin hopping round with different seed
    if best_pts is not None:
        elapsed = time.time() - start_time
        if elapsed < deadline - 15:
            time_limit = min(10.0, deadline - elapsed - 8)
            candidate_pts, candidate_energy = _basin_hopping(
                best_pts, time_limit=time_limit, seed=123
            )
            if candidate_energy < best_energy:
                best_energy = candidate_energy
                best_pts = candidate_pts.copy()

    # Phase 5: Riemannian Simulated annealing for additional exploration
    if best_pts is not None:
        elapsed = time.time() - start_time
        if elapsed < deadline - 12:
            time_limit = min(8.0, deadline - elapsed - 6)
            candidate_pts, candidate_energy = _riemannian_sa(
                best_pts, time_limit=time_limit, seed=777
            )
            if candidate_energy < best_energy:
                best_energy = candidate_energy
                best_pts = candidate_pts.copy()

    # Phase 6: L-BFGS-B polish
    if best_pts is not None:
        elapsed = time.time() - start_time
        if elapsed < deadline - 8:
            time_limit = min(5.0, deadline - elapsed - 3)
            candidate_pts, candidate_energy = _lbfgs_optimize(best_pts, time_limit=time_limit)
            if candidate_energy < best_energy:
                best_energy = candidate_energy
                best_pts = candidate_pts.copy()

    # Phase 7: High-precision final polish
    if best_pts is not None:
        elapsed = time.time() - start_time
        if elapsed < deadline - 3:
            time_limit = min(5.0, deadline - elapsed - 1)
            candidate_pts, candidate_energy = _final_polish(best_pts, time_limit=time_limit)
            if candidate_energy < best_energy:
                best_energy = candidate_energy
                best_pts = candidate_pts.copy()

    # Final projection and validation
    best_pts = _project_to_sphere(best_pts)

    # Validate: check for very close points
    diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    min_dist = np.min(dist)

    if min_dist < 1e-9:
        # Points too close, use Fibonacci instead
        return _fibonacci_sphere(N)

    return best_pts
# EVOLVE-BLOCK-END
