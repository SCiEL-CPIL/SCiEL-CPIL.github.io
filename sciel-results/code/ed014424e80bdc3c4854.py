"""Direct approach: build from the Fibonacci (near-optimal) structure, optimize on the sphere."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize


def _log_energy(pts):
    """Compute logarithmic energy: sum_{i<j} log(1/|x_i - x_j|)."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)
    iu = np.triu_indices(N, k=1)
    return -math.fsum(np.log(dist[iu]).tolist())


def _energy_fast(pts):
    """Fast energy calculation using np.sum for optimization loops."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)
    iu = np.triu_indices(N, k=1)
    return -np.sum(np.log(dist[iu]))


def _energy_and_gradient(pts):
    """Energy and gradient (in ambient R^3) of log energy."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)
    iu = np.triu_indices(N, k=1)
    energy = -np.sum(np.log(dist[iu]))
    inv_dist_sq = 1.0 / dist_sq
    np.fill_diagonal(inv_dist_sq, 0.0)
    grad = -np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
    return energy, grad


def _project_to_sphere(pts):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-14] = 1.0
    return pts / norms


def _fibonacci_points(n):
    """Fibonacci spiral (low-discrepancy) points on unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _sobol_points(n, seed=12345):
    """Quasi-random (Sobol-like via scipy) points on unit sphere."""
    from scipy.stats import qmc
    sampler = qmc.Sobol(d=3, scramble=True, seed=seed)
    raw = sampler.random(n)
    pts = 2.0 * raw - 1.0
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-14] = 1.0
    return pts / norms


def _optimize_on_sphere(pts, deadline, maxiter=2000, tight=False):
    """Optimize log energy on the sphere using L-BFGS-B."""
    pts = _project_to_sphere(pts.copy())

    def fun_and_grad(flat):
        p = flat.reshape(N, 3)
        p = _project_to_sphere(p)
        e, g = _energy_and_gradient(p)
        g_tan = g - np.sum(g * p, axis=1, keepdims=True) * p
        return e, g_tan.flatten()

    x0 = pts.flatten()
    best = pts.copy()
    best_e = _log_energy(pts)
    
    # First pass: standard tolerance
    try:
        res = minimize(
            fun_and_grad, x0, method='L-BFGS-B', jac=True,
            options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-12,
                     'maxls': 40, 'maxcor': 30}
        )
        p1 = _project_to_sphere(res.x.reshape(N, 3))
        e1 = _log_energy(p1)
        if e1 < best_e:
            best, best_e = p1, e1
    except Exception:
        pass

    if time.time() > deadline:
        return best, best_e

    # Second pass: tight tolerance if requested or if time permits
    if tight or (deadline - time.time() > 2.0):
        try:
            # Use a smaller maxiter for the second pass to avoid hanging, 
            # but tighter tolerances.
            res2 = minimize(
                fun_and_grad, best.flatten(), method='L-BFGS-B', jac=True,
                options={'maxiter': max(100, maxiter // 4), 'ftol': 1e-16, 'gtol': 1e-14,
                         'maxls': 30, 'maxcor': 30}
            )
            p2 = _project_to_sphere(res2.x.reshape(N, 3))
            e2 = _log_energy(p2)
            if e2 < best_e:
                best, best_e = p2, e2
        except Exception:
            pass

    return best, best_e


def _tangent_grad_descent_polish(pts, deadline, max_iter=5000):
    """High-precision line-search gradient descent on the sphere (Riemannian)."""
    pts = _project_to_sphere(pts.copy())
    best = pts.copy()
    best_e = _log_energy(pts)
    current_e = best_e
    for it in range(max_iter):
        if time.time() > deadline:
            break
        e, g = _energy_and_gradient(pts)
        g_tan = g - np.sum(g * pts, axis=1, keepdims=True) * pts
        gnorm = np.linalg.norm(g_tan)
        if gnorm < 1e-14:
            break
        alpha = 1.0
        improved = False
        for _ in range(40):
            cand = pts - alpha * g_tan
            cand = _project_to_sphere(cand)
            ce = _energy_fast(cand)
            if ce < current_e - 1e-4 * alpha * gnorm * gnorm:
                pts = cand
                current_e = ce
                improved = True
                break
            alpha *= 0.5
        if not improved:
            break
        if current_e < best_e - 1e-15:
            best = pts.copy()
            best_e = current_e
    return best, best_e


def _simulated_annealing(pts, deadline, max_iters=500, T0=0.05, T_end=0.001):
    """Simulated annealing on the sphere for escaping local minima."""
    pts = _project_to_sphere(pts.copy())
    current_e = _log_energy(pts)
    best_pts = pts.copy()
    best_e = current_e
    n_iters = max(1, max_iters)
    for it in range(n_iters):
        if time.time() > deadline:
            break
        T = T0 * (T_end / T0) ** (it / n_iters)
        # Random perturbation
        idx = np.random.randint(N)
        noise = np.random.normal(0, 0.05, 3)
        new_pts = pts.copy()
        new_pts[idx] = _project_to_sphere(new_pts[idx] + noise)
        new_e = _log_energy(new_pts)
        delta = new_e - current_e
        if delta < 0 or np.random.rand() < math.exp(-delta / max(T, 1e-10)):
            pts = new_pts
            current_e = new_e
            if current_e < best_e:
                best_e = current_e
                best_pts = pts.copy()
    return best_pts, best_e


def _basin_hopping(pts, deadline, max_iters=10):
    """Simple basin hopping: perturb and re-optimize."""
    current_pts = pts.copy()
    current_energy = _log_energy(current_pts)
    best_pts = current_pts.copy()
    best_energy = current_energy
    
    for i in range(max_iters):
        if time.time() > deadline:
            break
        
        noise = np.random.normal(0, 0.1, current_pts.shape)
        perturbed = _project_to_sphere(current_pts + noise)
        
        opt_pts, opt_e = _optimize_on_sphere(perturbed, deadline - 5.0, maxiter=500)
        
        if opt_e < current_energy:
            current_pts = opt_pts
            current_energy = opt_e
            if current_energy < best_energy:
                best_pts = current_pts.copy()
                best_energy = current_energy
                
    return best_pts, best_energy


def _local_search_neighbor_swap(pts, deadline, max_iters=200):
    """Try swapping pairs of points with small perturbations."""
    pts = _project_to_sphere(pts.copy())
    current_e = _log_energy(pts)
    best_pts = pts.copy()
    best_e = current_e
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        i, j = np.random.choice(N, 2, replace=False)
        # Small perturbation to points i and j
        pts_new = pts.copy()
        pts_new[i] = _project_to_sphere(pts_new[i] + np.random.normal(0, 0.02, 3))
        pts_new[j] = _project_to_sphere(pts_new[j] + np.random.normal(0, 0.02, 3))
        e_new = _log_energy(pts_new)
        if e_new < current_e:
            pts = pts_new
            current_e = e_new
            if current_e < best_e:
                best_e = current_e
                best_pts = pts.copy()
    return best_pts, best_e


def construct():
    deadline = time.time() + 145.0
    best_pts = None
    best_energy = float('inf')

    # Multiple diverse starting configurations
    start_configs = [
        _fibonacci_points(N),
        _sobol_points(N, seed=12345),
        _sobol_points(N, seed=99999),
        _sobol_points(N, seed=424242),
    ]

    # Phase 1: Optimize each starting config with time budget
    time_per_config = 25.0
    for i, cfg in enumerate(start_configs):
        remaining = deadline - time.time()
        if remaining < 5.0:
            break
        alloc = min(time_per_config, remaining - 15.0)
        if alloc < 3.0:
            break
        opt_pts, opt_e = _optimize_on_sphere(cfg, time.time() + alloc, maxiter=800)
        if opt_e < best_energy:
            best_energy = opt_e
            best_pts = opt_pts.copy()

    if best_pts is None:
        best_pts = _fibonacci_points(N)
        best_energy = _log_energy(best_pts)

    # Phase 2: Basin hopping with multiple perturbations
    if time.time() < deadline - 25.0:
        bh_deadline = deadline - 20.0
        current_pts = best_pts.copy()
        current_energy = best_energy
        for i in range(8):
            if time.time() > bh_deadline:
                break
            noise_scale = 0.05 + 0.05 * (i % 3)
            noise = np.random.normal(0, noise_scale, current_pts.shape)
            perturbed = _project_to_sphere(current_pts + noise)
            
            alloc = min(15.0, bh_deadline - time.time() - 5.0)
            if alloc < 3.0:
                break
            opt_pts, opt_e = _optimize_on_sphere(perturbed, time.time() + alloc, maxiter=600)
            
            if opt_e < current_energy:
                current_pts = opt_pts
                current_energy = opt_e
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_pts = opt_pts.copy()

    # Phase 3: Polish with gradient descent
    if time.time() < deadline - 10.0:
        p, e = _tangent_grad_descent_polish(best_pts, deadline - 5.0, max_iter=8000)
        if e < best_energy:
            best_energy = e
            best_pts = p

    # Final L-BFGS tight pass
    if time.time() < deadline - 5.0:
        p, e = _optimize_on_sphere(best_pts, deadline - 2.0, maxiter=2000)
        if e < best_energy:
            best_energy = e
            best_pts = p

    best_pts = _project_to_sphere(best_pts)

    # Validate: no points closer than 1e-9, finite values
    if not np.all(np.isfinite(best_pts)) or best_pts.shape != (N, 3):
        best_pts = _project_to_sphere(_fibonacci_points(N))
    d = np.linalg.norm(best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :], axis=2)
    np.fill_diagonal(d, 1.0)
    if np.min(d) < 1e-9:
        best_pts = _project_to_sphere(_fibonacci_points(N))

    return best_pts
# EVOLVE-BLOCK-END
