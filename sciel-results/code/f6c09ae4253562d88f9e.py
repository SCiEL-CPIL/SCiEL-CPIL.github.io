"""
Combined strategy:
1. Use L-BFGS-B on spherical coordinates (robust, handles constraint naturally).
2. Use fast vectorized energy/gradient (np.sum) for the optimization loop.
3. Implement a multi-phase search:
   a. Fibonacci start.
   b. Multiple random restarts.
   c. Perturbation of the best solution (local search around the best global candidate).
4. Strict time management.
"""
import numpy as np
import math
import time
from scipy.optimize import minimize

N = 150

# EVOLVE-BLOCK-START

def _points_to_spherical(pts):
    """Convert Cartesian points to spherical coordinates (theta, phi)."""
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    r = np.sqrt(x**2 + y**2 + z**2)
    r = np.maximum(r, 1e-15)
    theta = np.arccos(np.clip(z / r, -1.0, 1.0))
    phi = np.arctan2(y, x)
    # Ensure phi is in [0, 2pi]
    phi = np.where(phi < 0, phi + 2 * np.pi, phi)
    return theta, phi

def _spherical_to_points(theta, phi):
    """Convert spherical coordinates to Cartesian points on unit sphere."""
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return np.column_stack([x, y, z])

def _energy_and_grad_spherical(params):
    """
    Compute logarithmic energy and gradient with respect to spherical coordinates.
    params: array of shape (2N,). First N are theta, next N are phi.
    Uses np.sum for speed.
    """
    N_local = len(params) // 2
    theta = params[:N_local]
    phi = params[N_local:]
    
    pts = _spherical_to_points(theta, phi)
    
    # Compute squared distances
    dots = pts @ pts.T
    sq_dists = 2.0 - 2.0 * dots
    np.fill_diagonal(sq_dists, 1.0)
    # Clip for numerical stability
    sq_dists = np.maximum(sq_dists, 1e-15)
    
    # Energy: sum_{i<j} log(1/d) = -0.5 * sum_{i<j} log(d^2)
    iu = np.triu_indices(N_local, k=1)
    log_dists_sq = np.log(sq_dists[iu])
    # Using np.sum for speed in optimization loop
    energy = -0.5 * np.sum(log_dists_sq)
    
    # Gradient in Cartesian space
    # grad_xi = sum_{j != i} (xj - xi) / d_ij^2
    weights = 1.0 / sq_dists
    np.fill_diagonal(weights, 0.0)
    
    diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    # grad_cart[i] = - sum_j diffs[i, j] * weights[i, j]
    grad_cart = -np.sum(diffs * weights[..., np.newaxis], axis=1)
    
    # Project Cartesian gradient onto spherical basis vectors
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    
    basis_theta = np.column_stack([
        cos_theta * cos_phi,
        cos_theta * sin_phi,
        -sin_theta
    ])
    basis_phi = np.column_stack([
        -sin_theta * sin_phi,
        sin_theta * cos_phi,
        np.zeros(N_local)
    ])
    
    grad_theta = np.sum(grad_cart * basis_theta, axis=1)
    grad_phi = np.sum(grad_cart * basis_phi, axis=1)
    
    grad = np.concatenate([grad_theta, grad_phi])
    
    return energy, grad

def _optimize_spherical(theta, phi, max_iter=500, time_limit=None, start_time=None):
    """
    Optimize using L-BFGS-B on spherical coordinates.
    """
    if start_time is None:
        start_time = time.time()
        
    params = np.concatenate([theta, phi])
    
    # Bounds: theta in [1e-6, pi - 1e-6], phi in [0, 2pi]
    eps = 1e-6
    bounds = [(eps, np.pi - eps)] * N + [(0, 2 * np.pi)] * N
    
    def obj_and_grad(p):
        e, g = _energy_and_grad_spherical(p)
        return e, g
        
    # L-BFGS-B
    try:
        res = minimize(
            obj_and_grad,
            params,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
        )
        return res.x, res.fun
    except Exception:
        return params, _energy_and_grad_spherical(params)[0]

def _initial_fibonacci(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return pts

def _initial_random(n, seed=42):
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms

def _project_to_sphere(pts):
    """Project points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def _perturb_points(pts, sigma=0.1, seed=None):
    """Add Gaussian noise to points and project back to sphere."""
    rng = np.random.RandomState(seed)
    noise = rng.randn(*pts.shape) * sigma
    new_pts = pts + noise
    return _project_to_sphere(new_pts)

def construct():
    deadline = time.time() + 145  # Leave 5s margin for 150s limit
    
    best_pts = None
    best_energy = float('inf')
    
    def try_optimize(pts_init, max_iter, label=""):
        nonlocal best_pts, best_energy
        t0 = time.time()
        if time.time() > deadline - 5:
            return
        theta, phi = _points_to_spherical(pts_init)
        try:
            x_opt, e_opt = _optimize_spherical(theta, phi, max_iter=max_iter, time_limit=deadline - time.time() - 2, start_time=t0)
            if e_opt < best_energy:
                best_energy = e_opt
                theta_opt = x_opt[:N]
                phi_opt = x_opt[N:]
                best_pts = _spherical_to_points(theta_opt, phi_opt)
        except Exception:
            pass

    # Phase 1: Fibonacci start
    pts = _initial_fibonacci(N)
    try_optimize(pts, max_iter=300, label="Fibonacci")
    
    # Phase 2: Random restarts
    # Estimate time per restart. L-BFGS-B is fast.
    num_restarts = 0
    while time.time() < deadline - 20:  # Leave 20s for perturbation phase
        seed = 42 + num_restarts * 1000
        pts_r = _initial_random(N, seed=seed)
        try_optimize(pts_r, max_iter=150, label=f"Random {num_restarts}")
        num_restarts += 1
        if num_restarts > 30: # Safety cap
            break

    # Phase 3: Perturbation of the best solution
    # Try to escape local minima by perturbing the best found solution
    if best_pts is not None and time.time() < deadline - 10:
        # Try a few perturbations with decreasing noise
        noise_levels = [0.5, 0.2, 0.1, 0.05]
        for noise in noise_levels:
            if time.time() > deadline - 5:
                break
            for k in range(5):
                if time.time() > deadline - 5:
                    break
                seed = 1000 + int(noise * 1000) + k
                pts_pert = _perturb_points(best_pts, sigma=noise, seed=seed)
                try_optimize(pts_pert, max_iter=100, label=f"Perturb {noise} {k}")

    # Phase 4: Final polish
    if best_pts is not None and time.time() < deadline - 2:
        try_optimize(best_pts, max_iter=500, label="Polish")

    # Final safety check
    if best_pts is None:
        best_pts = _initial_fibonacci(N)
        
    return _project_to_sphere(best_pts)

# EVOLVE-BLOCK-END
