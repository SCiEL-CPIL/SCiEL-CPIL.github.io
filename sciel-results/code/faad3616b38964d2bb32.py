"""Spectral deformation + Riemannian polish for log-energy minimization on sphere."""
import numpy as np
import math
from scipy.optimize import minimize
from scipy.special import lpmv

N = 150

def _energy_and_grad(pts):
    """Compute logarithmic energy and gradient for points on unit sphere.
    Energy: sum_{i<j} log(1/|xi-xj|) = -sum_{i<j} log(|xi-xj|)
    Gradient wrt xi: sum_{j!=i} (xi - xj) / |xi-xj|^2
    """
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, 1.0)
    
    iu = np.triu_indices(n, k=1)
    energy = -np.sum(np.log(dist[iu]))
    
    inv_dist2 = 1.0 / (dist ** 2)
    np.fill_diagonal(inv_dist2, 0.0)
    grad = np.sum(diff * inv_dist2[:, :, np.newaxis], axis=1)
    
    return energy, grad


def _project_to_sphere(pts):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def _spherical_harmonic(l, m, theta, phi):
    """Compute spherical harmonic Y_l^m(theta, phi) using associated Legendre polynomials."""
    if l < abs(m):
        return 0.0
    P = lpmv(abs(m), l, np.cos(theta))
    norm = np.sqrt((2*l + 1) / (4*np.pi) * math.factorial(l - abs(m)) / math.factorial(l + abs(m)))
    if m < 0:
        norm *= (-1)**m
    return norm * P * np.exp(1j * m * phi)


def _generate_base_grid(N):
    """Generate a uniform grid on sphere using Fibonacci spiral."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return pts


def _spectral_deform(base_pts, coeffs, max_l):
    """Deform base points using spectral coefficients.
    coeffs: dict mapping (l, m) -> complex coefficient
    Returns deformed points on unit sphere.
    """
    n = len(base_pts)
    # Convert to spherical coordinates
    x, y, z = base_pts[:, 0], base_pts[:, 1], base_pts[:, 2]
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(np.clip(z / r, -1.0, 1.0))
    phi = np.arctan2(y, x)
    
    # Start with base position
    deformed = np.zeros_like(base_pts)
    
    # Add spectral deformation
    for (l, m), c in coeffs.items():
        if l > max_l:
            continue
        Y = _spherical_harmonic(l, m, theta, phi)
        # Use real part for deformation in radial direction
        radial_shift = np.real(c * Y)
        deformed += radial_shift[:, np.newaxis] * base_pts / r[:, np.newaxis]
    
    # Project to sphere
    return _project_to_sphere(deformed)


def _spectral_gradient(pts, base_pts, coeffs, max_l):
    """Compute gradient of energy with respect to spectral coefficients."""
    energy, grad = _energy_and_grad(pts)
    
    # Compute gradient with respect to each coefficient
    n = len(base_pts)
    x, y, z = base_pts[:, 0], base_pts[:, 1], base_pts[:, 2]
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(np.clip(z / r, -1.0, 1.0))
    phi = np.arctan2(y, x)
    
    coeff_grad = {}
    for (l, m), c in coeffs.items():
        if l > max_l:
            continue
        Y = _spherical_harmonic(l, m, theta, phi)
        # d(pts_i)/d(c) = Y_i * base_pts_i / r_i
        # dE/dc = sum_i grad_i . d(pts_i)/dc
        dpts_dc = np.real(Y)[:, np.newaxis] * base_pts / r[:, np.newaxis]
        coeff_grad[(l, m)] = np.sum(grad * dpts_dc)
    
    return energy, coeff_grad


def construct():
    # Generate base grid
    base_pts = _generate_base_grid(N)
    
    # Initialize spectral coefficients (start with small random values)
    np.random.seed(42)
    max_l = 4
    coeffs = {}
    for l in range(1, max_l + 1):
        for m in range(-l, l + 1):
            coeffs[(l, m)] = (np.random.randn() + 1j * np.random.randn()) * 0.01
    
    # Optimize in spectral coefficient space
    best_coeffs = coeffs.copy()
    best_energy, _ = _spectral_gradient(_spectral_deform(base_pts, coeffs, max_l), base_pts, coeffs, max_l)
    
    lr = 0.05
    max_iter = 2000
    patience = 300
    no_improve = 0
    
    for it in range(max_iter):
        pts = _spectral_deform(base_pts, coeffs, max_l)
        energy, coeff_grad = _spectral_gradient(pts, base_pts, coeffs, max_l)
        
        # Gradient descent in coefficient space
        new_coeffs = {}
        for key in coeffs:
            new_coeffs[key] = coeffs[key] - lr * coeff_grad[key]
        
        # Check if new configuration is valid
        new_pts = _spectral_deform(base_pts, new_coeffs, max_l)
        new_energy, _ = _energy_and_grad(new_pts)
        
        if new_energy < energy - 1e-10:
            coeffs = new_coeffs
            if new_energy < best_energy - 1e-12:
                best_energy = new_energy
                best_coeffs = coeffs.copy()
                no_improve = 0
            else:
                no_improve += 1
            if it % 100 == 0 and it > 0:
                lr *= 0.95
        else:
            # Backtrack
            lr *= 0.5
            no_improve += 1
        
        if no_improve > patience:
            break
    
    # Get deformed points
    pts = _spectral_deform(base_pts, best_coeffs, max_l)
    
    # Riemannian polish phase
    best_pts = pts.copy()
    best_energy, _ = _energy_and_grad(best_pts)
    
    lr = 0.01
    max_iter = 3000
    patience = 400
    no_improve = 0
    
    for it in range(max_iter):
        energy, grad = _energy_and_grad(pts)
        grad_proj = grad - np.sum(pts * grad, axis=1, keepdims=True) * pts
        
        step_lr = lr
        for _ in range(20):
            new_pts = pts + step_lr * grad_proj
            new_pts = _project_to_sphere(new_pts)
            new_energy, _ = _energy_and_grad(new_pts)
            if new_energy < energy - 1e-10:
                break
            step_lr *= 0.5
        else:
            step_lr = lr * 1e-3
            new_pts = pts + step_lr * grad_proj
            new_pts = _project_to_sphere(new_pts)
            new_energy, _ = _energy_and_grad(new_pts)
        
        if new_energy < energy - 1e-10:
            pts = new_pts
            if new_energy < best_energy - 1e-12:
                best_energy = new_energy
                best_pts = pts.copy()
                no_improve = 0
            else:
                no_improve += 1
            if it % 200 == 0 and it > 0:
                lr *= 0.9
        else:
            no_improve += 1
            lr *= 0.95
        
        if no_improve > patience:
            break
    
    pts = best_pts
    
    # Final fine polish
    lr = 0.001
    for it in range(1500):
        energy, grad = _energy_and_grad(pts)
        grad_proj = grad - np.sum(pts * grad, axis=1, keepdims=True) * pts
        
        step_lr = lr
        for _ in range(20):
            new_pts = pts + step_lr * grad_proj
            new_pts = _project_to_sphere(new_pts)
            new_energy, _ = _energy_and_grad(new_pts)
            if new_energy < energy - 1e-12:
                break
            step_lr *= 0.5
        else:
            break
        
        if new_energy < energy - 1e-12:
            pts = new_pts
            if it % 100 == 0:
                lr *= 0.9
        else:
            break
    
    pts = _project_to_sphere(pts)
    
    # Check for very close points
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, 1.0)
    min_dist = np.min(dist)
    
    if min_dist < 1e-9:
        for k in range(100):
            iu = np.argwhere(dist < 1e-9)
            if len(iu) == 0:
                break
            idx = iu[0]
            noise = np.random.randn(3) * 1e-6
            pts[idx[0]] += noise
            pts[idx[0]] /= np.linalg.norm(pts[idx[0]])
            pts[idx[1]] += noise
            pts[idx[1]] /= np.linalg.norm(pts[idx[1]])
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dist = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dist, 1.0)
    
    return pts
