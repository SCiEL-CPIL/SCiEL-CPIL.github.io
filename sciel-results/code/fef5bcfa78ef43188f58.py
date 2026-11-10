"""
Strategy: Spherical coordinate optimization with high-volume multi-restart.
- Parameterize points by (theta, phi) to avoid projection issues.
- Use BFGS for robustness in early phases, L-BFGS-B for final polish.
- Exploit full 120s budget for many restarts (20+) to explore landscape.
"""

import numpy as np
from scipy.optimize import minimize
from math import log, fsum
import time

N = 150
DEADLINE_FRACTION = 0.90  # Safety margin for 120s limit

def _to_cartesian(sph):
    """Convert (theta, phi) to Cartesian unit vectors.
    theta: azimuth [0, 2*pi)
    phi: polar [0, pi]
    """
    x = np.sin(sph[:, 1]) * np.cos(sph[:, 0])
    y = np.sin(sph[:, 1]) * np.sin(sph[:, 0])
    z = np.cos(sph[:, 1])
    return np.column_stack([x, y, z])

def _to_spherical(cart):
    """Convert Cartesian unit vectors to (theta, phi)."""
    x, y, z = cart[:, 0], cart[:, 1], cart[:, 2]
    phi = np.arccos(np.clip(z, -1.0, 1.0))
    theta = np.arctan2(y, x)
    # Handle negative theta
    theta[theta < 0] += 2 * np.pi
    return np.column_stack([theta, phi])

def _energy_and_grad_spherical(sph):
    """
    Compute energy and gradient in spherical coordinates.
    E = sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij)
    d_ij = |x_i - x_j|
    
    Gradient w.r.t. (theta_i, phi_i):
    dE/dtheta_i = sum_{j!=i} (dE/dd_ij) * (dd_ij/dtheta_i)
    dE/dphi_i   = sum_{j!=i} (dE/dd_ij) * (dd_ij/dphi_i)
    
    dd_ij/dx_i = (x_i - x_j)_x / d_ij
    dx_i/dtheta = (-sin(phi)sin(theta), sin(phi)cos(theta), 0)
    dx_i/dphi   = (cos(phi)cos(theta), cos(phi)sin(theta), -sin(phi))
    """
    n = len(sph)
    cart = _to_cartesian(sph)
    
    # Pairwise differences
    diff = cart[:, np.newaxis, :] - cart[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Avoid division by zero
    mask = dists > 1e-12
    inv_d = np.zeros_like(dists)
    inv_d[mask] = 1.0 / dists[mask]
    
    # Energy
    iu = np.triu_indices(n, k=1)
    energy = -np.sum(np.log(dists[iu]))
    
    # Gradient accumulation
    # dE/dx_i = sum_{j!=i} (x_i - x_j) / d_ij^2  (from -log(d))
    # Actually d(-log(d))/dx = -1/d * (x_i-x_j)/d = -(x_i-x_j)/d^2
    # Wait, E = sum -log(d). dE/dx_i = -1/d * d(d)/dx_i = -1/d * (x_i-x_j)/d = -(x_i-x_j)/d^2.
    
    # Let's compute grad_cart first
    grad_cart = np.zeros_like(cart)
    # term[i,j] = -(cart[i] - cart[j]) / dists[i,j]^2
    # We need to sum over j for each i.
    # diff[i,j] = cart[i] - cart[j]
    # term = -diff * inv_d[:, :, np.newaxis] * inv_d[:, :, np.newaxis]
    # But only for i != j.
    
    # Vectorized approach:
    # grad_cart[i] = sum_{j!=i} -(cart[i]-cart[j])/d_ij^2
    # This is equivalent to:
    # grad_cart = -sum_j (cart[i]-cart[j])/d_ij^2
    
    # Let's compute the matrix of coefficients
    # coeff[i,j] = 1/d_ij^2 if i!=j else 0
    coeff = np.zeros_like(dists)
    coeff[mask] = 1.0 / (dists[mask] ** 2)
    np.fill_diagonal(coeff, 0.0)
    
    # grad_cart[i] = sum_j coeff[i,j] * (cart[i] - cart[j])
    # = cart[i] * sum_j coeff[i,j] - sum_j coeff[i,j] * cart[j]
    
    sum_coeff = np.sum(coeff, axis=1)
    sum_coeff_cart = coeff @ cart
    
    grad_cart = cart * sum_coeff[:, np.newaxis] - sum_coeff_cart
    
    # Now project gradient to spherical coordinates
    # dE/dtheta_i = grad_cart[i] . dx/dtheta_i
    # dE/dphi_i   = grad_cart[i] . dx/dphi_i
    
    theta = sph[:, 0]
    phi = sph[:, 1]
    
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    
    # dx/dtheta = (-sin_phi*sin_theta, sin_phi*cos_theta, 0)
    # dx/dphi   = (cos_phi*cos_theta, cos_phi*sin_theta, -sin_phi)
    
    grad_theta = -grad_cart[:, 0] * sin_phi * sin_theta + grad_cart[:, 1] * sin_phi * cos_theta
    grad_phi = grad_cart[:, 0] * cos_phi * cos_theta + grad_cart[:, 1] * cos_phi * sin_theta - grad_cart[:, 2] * sin_phi
    
    grad_sph = np.column_stack([grad_theta, grad_phi])
    
    return energy, grad_sph

def _random_spherical(n, rng):
    """Generate random points on sphere in spherical coordinates."""
    theta = rng.uniform(0, 2 * np.pi, n)
    # Uniform distribution on sphere: cos(phi) uniform in [-1, 1]
    cos_phi = rng.uniform(-1, 1, n)
    phi = np.arccos(cos_phi)
    return np.column_stack([theta, phi])

def _fibonacci_spherical(n):
    """Generate Fibonacci spiral in spherical coordinates."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.arccos(z)
    theta = np.pi * (1.0 + np.sqrt(5.0)) * i
    # Normalize theta to [0, 2pi)
    theta = theta % (2 * np.pi)
    return np.column_stack([theta, phi])

def construct():
    start_time = time.time()
    total_time = 120.0
    deadline = start_time + total_time * DEADLINE_FRACTION
    
    rng = np.random.default_rng(42)
    
    best_pts = None
    best_energy = 1e30
    
    # Phase 1: Multi-restart optimization
    # We will try as many restarts as possible within the time budget.
    # Each restart:
    # 1. Initialize (Fibonacci or Random)
    # 2. Optimize with BFGS (robust, no Hessian) or L-BFGS-B
    
    def objective(x):
        sph = x.reshape(N, 2)
        e, _ = _energy_and_grad_spherical(sph)
        return e
        
    def grad_func(x):
        sph = x.reshape(N, 2)
        _, g = _energy_and_grad_spherical(sph)
        return g.flatten()
        
    n_restarts = 0
    while time.time() < deadline - 10:  # Leave 10s for polish
        n_restarts += 1
        if n_restarts == 1:
            init_sph = _fibonacci_spherical(N)
        else:
            # Mix of random and perturbed Fibonacci
            if n_restarts % 2 == 0:
                init_sph = _random_spherical(N, rng)
            else:
                init_sph = _fibonacci_spherical(N)
                # Add noise
                noise = rng.standard_normal((N, 2)) * 0.1
                init_sph = init_sph + noise
                # Wrap theta
                init_sph[:, 0] = init_sph[:, 0] % (2 * np.pi)
                # Clip phi
                init_sph[:, 1] = np.clip(init_sph[:, 1], 1e-6, np.pi - 1e-6)
                
        # Optimize
        # Use L-BFGS-B for efficiency, it handles bounds well if we add them, 
        # but we don't have explicit bounds in the param space except periodicity.
        # L-BFGS-B is generally faster than BFGS for high dimensions.
        try:
            res = minimize(
                objective, init_sph.flatten(), jac=grad_func, method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-6, 'maxls': 50}
            )
            cand_sph = res.x.reshape(N, 2)
            # Clean up spherical coords
            cand_sph[:, 0] = cand_sph[:, 0] % (2 * np.pi)
            cand_sph[:, 1] = np.clip(cand_sph[:, 1], 1e-6, np.pi - 1e-6)
            
            cand_energy, _ = _energy_and_grad_spherical(cand_sph)
            
            if cand_energy < best_energy:
                best_energy = cand_energy
                best_pts = cand_sph
                # Update deadline if we are doing well? No, keep it fixed.
                # But we can break early if we find a very good solution?
                # No, let's use the time for more restarts to be safe.
                
        except Exception as e:
            continue
            
    if best_pts is None:
        # Fallback
        best_pts = _fibonacci_spherical(N)
        
    # Phase 2: Polish the best solution with tighter tolerances
    if time.time() < deadline - 2:
        try:
            res = minimize(
                objective, best_pts.flatten(), jac=grad_func, method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 100}
            )
            cand_sph = res.x.reshape(N, 2)
            cand_sph[:, 0] = cand_sph[:, 0] % (2 * np.pi)
            cand_sph[:, 1] = np.clip(cand_sph[:, 1], 1e-6, np.pi - 1e-6)
            
            cand_energy, _ = _energy_and_grad_spherical(cand_sph)
            
            if cand_energy < best_energy:
                best_energy = cand_energy
                best_pts = cand_sph
        except Exception:
            pass
            
    # Convert back to Cartesian for output
    cart = _to_cartesian(best_pts)
    # Normalize to ensure unit sphere (just in case of numerical drift)
    norms = np.linalg.norm(cart, axis=1, keepdims=True)
    norms = np.where(norms < 1e-15, 1.0, norms)
    cart = cart / norms
    
    return cart
