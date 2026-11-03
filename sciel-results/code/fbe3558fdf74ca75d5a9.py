"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize
from scipy.spatial.distance import cdist


def _log_energy(points):
    """Compute log(1/d) energy using math.fsum for accuracy."""
    n = len(points)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(dists_sq, 1.0)
    iu = np.triu_indices(n, k=1)
    log_dists = 0.5 * np.log(dists_sq[iu])
    return -math.fsum(log_dists.tolist())


def _fibonacci_sphere(n):
    """Generate Fibonacci sphere points."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_uniform_sphere(n, seed=42):
    """Generate random uniform points on sphere."""
    rng = np.random.RandomState(seed)
    points = rng.randn(n, 3)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    return points / norms


def _project_to_sphere(points):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return points / norms


def _energy_and_grad(x):
    """Energy and gradient for L-BFGS-B. x is flat array of shape (N*3,)."""
    points = x.reshape(N, 3)
    points = _project_to_sphere(points)
    
    n = len(points)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(dists_sq, 1.0)
    
    # E = -sum_{i<j} log(d_ij)
    # dE/dx_i = sum_{j!=i} (x_j - x_i) / d_ij^2
    weights = np.zeros_like(dists_sq)
    mask = ~np.eye(n, dtype=bool)
    weights[mask] = 1.0 / dists_sq[mask]
    
    sum_weights = np.sum(weights, axis=1)
    weighted_sum = weights @ points
    
    grads = weighted_sum - points * sum_weights[:, np.newaxis]
    
    # Project gradient to tangent space
    dot_products = np.sum(grads * points, axis=1, keepdims=True)
    grads_tangent = grads - dot_products * points
    
    # Compute energy with math.fsum
    iu = np.triu_indices(n, k=1)
    log_dists = 0.5 * np.log(dists_sq[iu])
    energy = -math.fsum(log_dists.tolist())
    
    return energy, grads_tangent.flatten()


def _polish(points, max_iter=5000, ftol=1e-14, gtol=1e-10, time_limit=30.0):
    """Polish a point configuration using L-BFGS-B with tight tolerances."""
    start = time.time()
    x0 = points.flatten()
    e0 = _log_energy(points)
    try:
        res = minimize(
            _energy_and_grad,
            x0,
            method='L-BFGS-B',
            jac=True,
            options={
                'maxiter': max_iter,
                'maxfun': max_iter * 10,
                'ftol': ftol,
                'gtol': gtol,
                'maxls': 50
            }
        )
        if res.success or res.fun < e0:
            p_res = _project_to_sphere(res.x.reshape(N, 3))
            e_check = _log_energy(p_res)
            if e_check < e0 and time.time() - start < time_limit:
                return p_res
    except Exception:
        pass
    return points


def _gradient_descent_sphere(points, lr=0.01, max_iter=10000, time_limit=20.0):
    """Gradient descent on the sphere with adaptive step size."""
    start = time.time()
    pts = points.copy()
    best_e = _log_energy(pts)
    best_pts = pts.copy()
    
    for it in range(max_iter):
        if time.time() - start > time_limit:
            break
        
        # Compute gradient
        n = len(pts)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists_sq = np.sum(diff * diff, axis=-1)
        np.fill_diagonal(dists_sq, 1.0)
        
        weights = np.zeros_like(dists_sq)
        mask = ~np.eye(n, dtype=bool)
        weights[mask] = 1.0 / dists_sq[mask]
        
        sum_weights = np.sum(weights, axis=1)
        weighted_sum = weights @ pts
        grads = weighted_sum - pts * sum_weights[:, np.newaxis]
        
        # Project to tangent
        dot_products = np.sum(grads * pts, axis=1, keepdims=True)
        grads_tangent = grads - dot_products * pts
        
        # Adaptive step size
        grad_norm = np.linalg.norm(grads_tangent)
        if grad_norm < 1e-12:
            break
        
        step = lr / (1.0 + grad_norm * 0.1)
        new_pts = pts - step * grads_tangent
        new_pts = _project_to_sphere(new_pts)
        
        e_new = _log_energy(new_pts)
        if e_new < best_e:
            best_e = e_new
            best_pts = new_pts.copy()
            pts = new_pts
            lr = min(lr * 1.05, 0.1)
        else:
            lr *= 0.9
    
    return best_pts


def _simulated_annealing(points, T0=0.1, T_end=0.001, max_iter=5000, time_limit=25.0):
    """Simulated annealing to escape local minima."""
    start = time.time()
    rng = np.random.RandomState(12345)
    pts = points.copy()
    best_e = _log_energy(pts)
    best_pts = pts.copy()
    current_e = best_e
    
    for it in range(max_iter):
        if time.time() - start > time_limit:
            break
        
        # Cooling schedule
        alpha = it / max_iter
        T = T0 * (T_end / T0) ** alpha
        
        # Random perturbation
        idx = rng.randint(0, N)
        noise = rng.randn(3) * 0.1
        new_pts = pts.copy()
        new_pts[idx] = pts[idx] + noise
        new_pts = _project_to_sphere(new_pts)
        
        e_new = _log_energy(new_pts)
        delta = e_new - current_e
        
        if delta < 0 or rng.random() < math.exp(-delta / max(T, 1e-10)):
            pts = new_pts
            current_e = e_new
            if e_new < best_e:
                best_e = e_new
                best_pts = new_pts.copy()
    
    return best_pts


def _multi_start_optimize(points, time_limit=30.0):
    """Try multiple perturbation strategies and pick the best."""
    start = time.time()
    best_pts = points.copy()
    best_e = _log_energy(best_pts)
    
    # Try different perturbation scales
    for scale in [0.05, 0.1, 0.2, 0.3]:
        if time.time() - start > time_limit - 10.0:
            break
        
        rng = np.random.RandomState(int(time.time() * 1000) % 100000)
        perturbed = points + rng.randn(N, 3) * scale
        perturbed = _project_to_sphere(perturbed)
        
        # Quick gradient descent
        gd_result = _gradient_descent_sphere(perturbed, lr=0.01, max_iter=2000, 
                                              time_limit=time_limit - (time.time() - start) - 5.0)
        e_gd = _log_energy(gd_result)
        
        if e_gd < best_e:
            best_e = e_gd
            best_pts = gd_result.copy()
    
    return best_pts


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Multiple initializations - diverse starting points
    inits = [
        _fibonacci_sphere(N),
        _random_uniform_sphere(N, seed=1),
        _random_uniform_sphere(N, seed=2),
        _random_uniform_sphere(N, seed=3),
        _random_uniform_sphere(N, seed=7),
        _random_uniform_sphere(N, seed=13),
        _random_uniform_sphere(N, seed=42),
        _random_uniform_sphere(N, seed=99),
    ]
    
    best_points = inits[0].copy()
    best_energy = _log_energy(best_points)
    
    # First pass: moderate optimization on all initializations
    for idx, init in enumerate(inits):
        if time.time() > deadline - 60.0:
            break
        
        try:
            res = minimize(
                _energy_and_grad,
                init.flatten(),
                method='L-BFGS-B',
                jac=True,
                options={
                    'maxiter': 2000,
                    'maxfun': 20000,
                    'ftol': 1e-12,
                    'gtol': 1e-8,
                    'maxls': 50
                }
            )
            
            if res.fun < best_energy:
                p_res = _project_to_sphere(res.x.reshape(N, 3))
                e_check = _log_energy(p_res)
                if e_check < best_energy:
                    best_energy = e_check
                    best_points = p_res.copy()
        except Exception:
            pass
    
    # Second pass: simulated annealing to escape local minima
    if time.time() < deadline - 40.0:
        sa_result = _simulated_annealing(best_points, T0=0.1, T_end=0.001, 
                                          max_iter=5000, 
                                          time_limit=deadline - time.time() - 35.0)
        e_sa = _log_energy(sa_result)
        if e_sa < best_energy:
            best_energy = e_sa
            best_points = sa_result.copy()
    
    # Third pass: gradient descent refinement
    if time.time() < deadline - 25.0:
        gd_result = _gradient_descent_sphere(best_points, lr=0.01, max_iter=5000,
                                              time_limit=deadline - time.time() - 20.0)
        e_gd = _log_energy(gd_result)
        if e_gd < best_energy:
            best_energy = e_gd
            best_points = gd_result.copy()
    
    # Fourth pass: aggressive L-BFGS-B polishing
    if time.time() < deadline - 15.0:
        polished = _polish(best_points, max_iter=8000, ftol=1e-14, gtol=1e-10,
                           time_limit=deadline - time.time() - 10.0)
        e_polish = _log_energy(polished)
        if e_polish < best_energy:
            best_energy = e_polish
            best_points = polished.copy()
    
    # Fifth pass: multi-start with perturbations if time remains
    if time.time() < deadline - 10.0:
        multi_result = _multi_start_optimize(best_points, 
                                              time_limit=deadline - time.time() - 5.0)
        e_multi = _log_energy(multi_result)
        if e_multi < best_energy:
            best_energy = e_multi
            best_points = multi_result.copy()
    
    # Final ultra-tight polish
    if time.time() < deadline - 5.0:
        final_polished = _polish(best_points, max_iter=10000, ftol=1e-15, gtol=1e-12,
                                 time_limit=deadline - time.time() - 3.0)
        e_final = _log_energy(final_polished)
        if e_final < best_energy:
            best_points = final_polished.copy()
    
    best_points = _project_to_sphere(best_points)
    
    if not np.all(np.isfinite(best_points)):
        return _fibonacci_sphere(N)
    
    # Check for points too close
    diff = best_points[:, np.newaxis, :] - best_points[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=-1)
    iu = np.triu_indices(N, k=1)
    if np.min(dists_sq[iu]) < 1e-18:
        return _fibonacci_sphere(N)
    
    return best_points
# EVOLVE-BLOCK-END
