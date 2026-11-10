"""Rewritten program: use scipy L-BFGS-B in spherical coordinates for log-energy minimization."""
import numpy as np
import math
import time
from scipy.optimize import minimize

N = 150

# EVOLVE-BLOCK-START

def _spherical_to_cartesian(theta, phi):
    """Convert (theta, phi) arrays to Cartesian coordinates on unit sphere.
    theta: polar angle [0, pi], phi: azimuthal angle [0, 2*pi)
    """
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return np.column_stack([x, y, z])


def _cartesian_to_spherical(pts):
    """Convert Cartesian points on unit sphere to (theta, phi)."""
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    r = np.sqrt(x**2 + y**2)
    theta = np.arccos(np.clip(z, -1.0, 1.0))
    phi = np.arctan2(y, x)
    phi = np.where(phi < 0, phi + 2 * np.pi, phi)
    return theta, phi


def _energy_and_grad_spherical(theta, phi):
    """Compute log-energy and gradient w.r.t. (theta, phi) parameters.
    Energy = sum_{i<j} log(1/|xi-xj|) = -sum_{i<j} log|xi-xj|
    """
    pts = _spherical_to_cartesian(theta, phi)
    n = len(theta)
    
    # Pairwise distances
    sq_dists = 2.0 - 2.0 * (pts @ pts.T)
    np.fill_diagonal(sq_dists, 1.0)
    sq_dists = np.maximum(sq_dists, 1e-15)
    dists = np.sqrt(sq_dists)
    
    # Energy
    iu = np.triu_indices(n, k=1)
    log_dists = np.log(dists[iu])
    energy = -math.fsum(log_dists.tolist())
    
    # Gradient: dE/dtheta_i, dE/dphi_i
    # dE/dxi = -sum_{j!=i} (xi-xj)/|xi-xj|^2  (in Cartesian)
    # Then project to spherical coordinates.
    # 
    # For a point on the sphere, the tangent vectors are:
    # e_theta = (cos(theta)cos(phi), cos(theta)sin(phi), -sin(theta))
    # e_phi   = (-sin(phi), cos(phi), 0)
    # 
    # grad_theta_i = dE/dxi . e_theta_i
    # grad_phi_i   = dE/dxi . e_phi_i
    
    grad_cart = np.zeros_like(pts)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            diff = pts[i] - pts[j]
            d2 = sq_dists[i, j]
            grad_cart[i] += -diff / d2
    
    # Convert to spherical gradient
    ct = np.cos(theta)
    st = np.sin(theta)
    cp = np.cos(phi)
    sp = np.sin(phi)
    
    # e_theta components
    et_x = ct * cp
    et_y = ct * sp
    et_z = -st
    
    # e_phi components
    ep_x = -sp
    ep_y = cp
    ep_z = 0.0
    
    grad_theta = grad_cart[:, 0] * et_x + grad_cart[:, 1] * et_y + grad_cart[:, 2] * et_z
    grad_phi = grad_cart[:, 0] * ep_x + grad_cart[:, 1] * ep_y + grad_cart[:, 2] * ep_z
    
    grad = np.concatenate([grad_theta, grad_phi])
    return energy, grad


def _initial_fibonacci(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    theta, phi_sph = _cartesian_to_spherical(pts)
    return theta, phi_sph


def _initial_random(n, seed=42):
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    pts = pts / norms
    theta, phi = _cartesian_to_spherical(pts)
    return theta, phi


def _optimize_spherical(theta0, phi0, max_iter=3000, time_limit=None, start_time=None):
    """Optimize using scipy L-BFGS-B in spherical coordinates."""
    if start_time is None:
        start_time = time.time()
    
    x0 = np.concatenate([theta0, phi0])
    
    # Bounds: theta in [0, pi], phi in [0, 2*pi]
    bounds = [(0.0, np.pi)] * N + [(0.0, 2 * np.pi)] * N
    
    # Time-aware callback to stop early
    def callback(xk):
        if time_limit is not None:
            if time.time() - start_time > time_limit:
                return True
        return False
    
    res = minimize(
        _energy_and_grad_spherical,
        x0,
        method='L-BFGS-B',
        jac=True,
        bounds=bounds,
        options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50,
                 'callback': callback}
    )
    
    x_opt = res.x
    theta_opt = x_opt[:N]
    phi_opt = x_opt[N:]
    
    # Ensure valid range
    theta_opt = np.clip(theta_opt, 0.0, np.pi)
    phi_opt = np.where(phi_opt < 0, phi_opt + 2 * np.pi, phi_opt)
    phi_opt = phi_opt % (2 * np.pi)
    
    energy = res.fun
    if not np.isfinite(energy):
        # Fallback: compute directly
        energy, _ = _energy_and_grad_spherical(theta_opt, phi_opt)
    
    return theta_opt, phi_opt, energy


def _perturb_points(theta, phi, sigma=0.1):
    """Add small random perturbation to escape local minima."""
    theta_p = theta + np.random.randn(N) * sigma
    phi_p = phi + np.random.randn(N) * sigma
    theta_p = np.clip(theta_p, 0.0, np.pi)
    phi_p = phi_p % (2 * np.pi)
    return theta_p, phi_p


def construct():
    deadline = time.time() + 140  # Leave 10s margin before 150s limit
    
    best_energy = float('inf')
    best_theta = None
    best_phi = None
    
    # Start with Fibonacci spiral
    theta0, phi0 = _initial_fibonacci(N)
    t0 = time.time()
    theta_opt, phi_opt, e = _optimize_spherical(theta0, phi0, max_iter=5000,
                                                 time_limit=50, start_time=t0)
    if e < best_energy:
        best_energy = e
        best_theta = theta_opt
        best_phi = phi_opt
    
    # Try several random restarts
    np.random.seed(12345)
    for seed in range(20):
        if time.time() > deadline - 30:
            break
        
        theta_r, phi_r = _initial_random(N, seed=seed)
        t0 = time.time()
        theta_opt, phi_opt, e = _optimize_spherical(theta_r, phi_r, max_iter=3000,
                                                     time_limit=25, start_time=t0)
        if e < best_energy:
            best_energy = e
            best_theta = theta_opt
            best_phi = phi_opt
    
    # If we still have time, do perturbation-based restarts from best
    if best_theta is not None and time.time() < deadline - 25:
        for _ in range(5):
            if time.time() > deadline - 15:
                break
            theta_p, phi_p = _perturb_points(best_theta, best_phi, sigma=0.05)
            t0 = time.time()
            theta_opt, phi_opt, e = _optimize_spherical(theta_p, phi_p, max_iter=2000,
                                                         time_limit=20, start_time=t0)
            if e < best_energy:
                best_energy = e
                best_theta = theta_opt
                best_phi = phi_opt
    
    # Convert back to Cartesian
    pts = _spherical_to_cartesian(best_theta, best_phi)
    
    # Ensure unit norm
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    pts = pts / norms
    
    # Verify no points are too close
    sq_dists = 2.0 - 2.0 * (pts @ pts.T)
    np.fill_diagonal(sq_dists, 1.0)
    min_d2 = np.min(sq_dists)
    if min_d2 < 1e-18:
        # Fallback to Fibonacci if something went wrong
        pts = _spherical_to_cartesian(*_initial_fibonacci(N))
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / np.maximum(norms, 1e-15)
    
    return pts

# EVOLVE-BLOCK-END
