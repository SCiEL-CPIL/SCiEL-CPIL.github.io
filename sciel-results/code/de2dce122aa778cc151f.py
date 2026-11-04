"""Optimize 150 points on unit sphere to minimize logarithmic energy."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import math
import time
from scipy.optimize import minimize


def _energy_and_grad(pts):
    """Compute log energy sum_{i<j} log(1/d) and its gradient."""
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0)

    mask = dist_sq > 1e-30
    inv_dist_sq = np.zeros_like(dist_sq)
    inv_dist_sq[mask] = 1.0 / dist_sq[mask]
    np.fill_diagonal(inv_dist_sq, 0.0)

    grad = -np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)

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


def _icosahedron_based(n, seed=42):
    """Generate points using icosahedron vertices as anchors for better initial distribution."""
    t = (1.0 + np.sqrt(5.0)) / 2.0
    ico = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
    ], dtype=float)
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
    
    fib = _fibonacci_sphere(n, seed=seed)
    result = fib.copy()
    for i in range(12):
        dists = np.linalg.norm(fib - ico[i], axis=1)
        closest = np.argmin(dists)
        result[closest] = ico[i]
    return _project_to_sphere(result)


def _stochastic_grad(pts, batch_size, rng):
    """Compute stochastic gradient using a random batch of point pairs."""
    n = len(pts)
    # Sample random pairs
    i_idx = rng.randint(0, n, size=batch_size)
    j_idx = rng.randint(0, n, size=batch_size)
    # Ensure i != j
    mask = i_idx != j_idx
    while not np.all(mask):
        bad = np.where(~mask)[0]
        j_idx[bad] = rng.randint(0, n, size=len(bad))
        mask = i_idx != j_idx
    
    diff = pts[i_idx] - pts[j_idx]
    dist_sq = np.sum(diff * diff, axis=1)
    dist_sq = np.maximum(dist_sq, 1e-30)
    inv_dist_sq = 1.0 / dist_sq
    
    # Gradient contribution for each point in the batch
    grad = np.zeros_like(pts)
    np.add.at(grad, i_idx, -diff * inv_dist_sq[:, np.newaxis])
    np.add.at(grad, j_idx, diff * inv_dist_sq[:, np.newaxis])
    
    # Scale to estimate full gradient (batch_size pairs out of n*(n-1)/2 total)
    scale = (n * (n - 1) / 2.0) / batch_size
    grad *= scale
    
    return grad


def _riemannian_line_search(pts, direction, energy, max_iter=50):
    """Robust Riemannian line search with Armijo condition."""
    # Project direction to tangent space
    direction = _tangent_project(direction, pts)
    dir_norm = np.linalg.norm(direction)
    if dir_norm < 1e-15:
        return pts, energy, 0.0
    
    direction = direction / dir_norm
    
    alpha = 0.01
    c1 = 1e-4  # Armijo parameter
    beta = 0.5
    
    for _ in range(max_iter):
        # Retraction: exp map approximation for small alpha
        new_pts = pts + alpha * direction
        new_pts = _project_to_sphere(new_pts)
        new_energy = _energy_only(new_pts)
        
        # Armijo condition
        if new_energy <= energy + c1 * alpha * (-dir_norm):
            return new_pts, new_energy, alpha
        
        alpha *= beta
        if alpha < 1e-10:
            break
    
    return pts, energy, 0.0


def _rsgd_optimize(pts, time_limit=15.0, batch_size=100, lr_initial=0.01, seed=42):
    """Riemannian Stochastic Gradient Descent with mini-batches and line search."""
    start_time = time.time()
    pts = _project_to_sphere(pts.copy())
    n = len(pts)
    rng = np.random.RandomState(seed)
    
    current_energy, _ = _energy_and_grad(pts)
    best_pts = pts.copy()
    best_energy = current_energy
    
    lr = lr_initial
    lr_decay = 0.999
    momentum = 0.9
    velocity = np.zeros_like(pts)
    
    no_improve = 0
    patience = 200
    
    while time.time() - start_time < time_limit:
        # Stochastic gradient
        sg = _stochastic_grad(pts, batch_size, rng)
        
        # Momentum
        velocity = momentum * velocity + (1 - momentum) * sg
        velocity = _tangent_project(velocity, pts)
        
        # Line search
        new_pts, new_energy, alpha = _riemannian_line_search(pts, -velocity, current_energy)
        
        if alpha > 0:
            pts = new_pts
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_pts = pts.copy()
                no_improve = 0
            else:
                no_improve += 1
        else:
            no_improve += 1
        
        # Decay learning rate
        lr = max(0.001, lr * lr_decay)
        
        if no_improve >= patience:
            break
    
    return best_pts, best_energy


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
            'maxiter': 30000,
            'maxfun': 80000,
            'ftol': 1e-18,
            'gtol': 1e-16,
            'maxls': 300
        }
    else:
        options = {
            'maxiter': 5000,
            'maxfun': 15000,
            'ftol': 1e-15,
            'gtol': 1e-13,
            'maxls': 100
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


def _basin_hopping(pts, time_limit=15.0, seed=42):
    """Basin hopping: SA-like exploration with L-BFGS refinement."""
    start_time = time.time()
    pts = _project_to_sphere(pts.copy())
    n = len(pts)

    current_energy, _ = _energy_and_grad(pts)
    best_pts = pts.copy()
    best_energy = current_energy

    T = 0.05
    T_min = 1e-6
    cooling = 0.998
    max_steps = 5000

    rng = np.random.RandomState(seed)

    for step in range(max_steps):
        if time.time() - start_time > time_limit:
            break

        r = rng.rand()
        if r < 0.05:
            noise_scale = T * 0.5
            perturbation = rng.randn(n, 3) * noise_scale
        elif r < 0.15:
            k = rng.randint(10, 30)
            idx = rng.choice(n, size=k, replace=False)
            perturbation = np.zeros((n, 3))
            perturbation[idx] = rng.randn(k, 3) * T * 0.3
        else:
            noise_scale = T * 0.08
            perturbation = rng.randn(n, 3) * noise_scale
        
        new_pts = pts + perturbation
        new_pts = _project_to_sphere(new_pts)

        elapsed = time.time() - start_time
        remaining = time_limit - elapsed
        if remaining > 2.0:
            polish_limit = min(1.5, remaining * 0.25)
            polished_pts, polished_energy = _lbfgs_optimize(new_pts, time_limit=polish_limit)
        else:
            polished_pts = new_pts
            polished_energy = _energy_only(new_pts)

        if polished_energy < current_energy or (T > T_min and rng.rand() < math.exp(-(polished_energy - current_energy) / T)):
            pts = polished_pts
            current_energy = polished_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_pts = pts.copy()

        T *= cooling
        if T < T_min:
            break

    return best_pts, best_energy


def _final_polish(pts, time_limit=8.0):
    """High-precision final polish with very tight tolerances."""
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
            'maxiter': 50000,
            'maxfun': 100000,
            'ftol': 1e-20,
            'gtol': 1e-18,
            'maxls': 500
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
    starts.append(_icosahedron_based(N, seed=42))
    starts.append(_icosahedron_based(N, seed=99))
    for s in range(2, 8):
        starts.append(_random_sphere(N, seed=s * 100))

    # Phase 1: L-BFGS-B from each diverse start
    for i, start_pts in enumerate(starts):
        elapsed = time.time() - start_time
        if elapsed > deadline - 40:
            break
        time_limit = min(5.0, deadline - elapsed - 35)
        if time_limit < 2.0:
            break
        candidate_pts, candidate_energy = _lbfgs_optimize(start_pts, time_limit=time_limit)
        if candidate_energy < best_energy:
            best_energy = candidate_energy
            best_pts = candidate_pts.copy()

    # Phase 2: RSGD optimization from best point (new: fast stochastic method)
    if best_pts is not None:
        elapsed = time.time() - start_time
        if elapsed < deadline - 35:
            time_limit = min(15.0, deadline - elapsed - 30)
            if time_limit >= 3.0:
                candidate_pts, candidate_energy = _rsgd_optimize(
                    best_pts, time_limit=time_limit, batch_size=100, lr_initial=0.01, seed=42
                )
                if candidate_energy < best_energy:
                    best_energy = candidate_energy
                    best_pts = candidate_pts.copy()

    # Phase 3: RSGD with different parameters
    if best_pts is not None:
        elapsed = time.time() - start_time
        if elapsed < deadline - 25:
            time_limit = min(12.0, deadline - elapsed - 20)
            if time_limit >= 3.0:
                candidate_pts, candidate_energy = _rsgd_optimize(
                    best_pts, time_limit=time_limit, batch_size=50, lr_initial=0.02, seed=123
                )
                if candidate_energy < best_energy:
                    best_energy = candidate_energy
                    best_pts = candidate_pts.copy()

    # Phase 4: Basin hopping to escape local minima
    for bh_seed in [42, 123, 777]:
        if best_pts is not None:
            elapsed = time.time() - start_time
            if elapsed < deadline - 20:
                time_limit = min(12.0, deadline - elapsed - 15)
                if time_limit < 3.0:
                    break
                candidate_pts, candidate_energy = _basin_hopping(
                    best_pts, time_limit=time_limit, seed=bh_seed
                )
                if candidate_energy < best_energy:
                    best_energy = candidate_energy
                    best_pts = candidate_pts.copy()

    # Phase 5: L-BFGS-B polish
    if best_pts is not None:
        elapsed = time.time() - start_time
        if elapsed < deadline - 12:
            time_limit = min(6.0, deadline - elapsed - 8)
            if time_limit >= 2.0:
                candidate_pts, candidate_energy = _lbfgs_optimize(best_pts, time_limit=time_limit, tight=True)
                if candidate_energy < best_energy:
                    best_energy = candidate_energy
                    best_pts = candidate_pts.copy()

    # Phase 6: High-precision final polish
    if best_pts is not None:
        elapsed = time.time() - start_time
        if elapsed < deadline - 5:
            time_limit = min(8.0, deadline - elapsed - 2)
            if time_limit >= 2.0:
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
