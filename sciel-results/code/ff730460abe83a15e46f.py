"""
Spherical harmonic density-based optimization for minimal logarithmic energy
of 100 points on the unit sphere.

Approach:
1. Represent a probability density on the sphere using spherical harmonics up to
   degree L. Use a squared expansion to ensure positivity:
   rho(theta, phi) = (1 + sum_{l=1}^{L} sum_{m=-l}^{l} c_{lm} Y_{lm}(theta,phi))^2
2. Minimize the continuous energy E = ∫∫ log(1/|x-y|) rho(x) rho(y) dΩ_x dΩ_y
   using gradient descent on the coefficients, with normalization constraint.
3. Sample 100 points from the optimized density using importance sampling.
4. Refine with direct point-based optimization (L-BFGS-B) starting from the
   sampled points.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.special import lpmv
from math import fsum, log, sqrt, pi
import time

N = 100

def _energy(points):
    """Compute the logarithmic energy sum_{i<j} log(1/|x_i - x_j|)."""
    n = len(points)
    total = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(points[i] - points[j])
            if d < 1e-15:
                return 1e10
            total += log(1.0 / d)
    return total

def _energy_grad(points):
    """Compute gradient of the logarithmic energy with respect to point positions."""
    n = len(points)
    grad = np.zeros_like(points)
    for i in range(n):
        for j in range(i + 1, n):
            diff = points[i] - points[j]
            d = np.linalg.norm(diff)
            if d < 1e-15:
                continue
            grad[i] -= diff / (d * d)
            grad[j] += diff / (d * d)
    return grad

def _project_to_sphere(points):
    """Project points onto the unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return points / norms

def _spherical_harmonics(theta, phi, L):
    """
    Compute real spherical harmonics Y_lm(theta, phi) for l=0..L, m=-l..l.
    Returns array of shape (n_theta, n_phi, (L+1)^2) or similar.
    We'll use complex spherical harmonics for simplicity.
    """
    n = len(theta)
    coeffs = []
    for l in range(L + 1):
        for m in range(-l, l + 1):
            # Complex spherical harmonic Y_l^m(theta, phi)
            # Y_l^m = sqrt((2l+1)/(4pi) * (l-m)!/(l+m)!) * P_l^m(cos(theta)) * exp(i*m*phi)
            P_lm = lpmv(abs(m), l, np.cos(theta))
            # Condon-Shortley phase is included in lpmv
            if m < 0:
                P_lm = (-1)**abs(m) * lpmv(abs(m), l, np.cos(theta))
            norm_factor = sqrt((2*l + 1)/(4*pi) * 1.0)
            # For simplicity, use a simplified normalization
            Y = P_lm * np.exp(1j * m * phi) * norm_factor
            coeffs.append(Y)
    return np.array(coeffs).T  # shape (n, (L+1)^2)

def _build_quadrature_grid(n_theta, n_phi):
    """Build a quadrature grid on the sphere."""
    theta = np.linspace(0, pi, n_theta)
    phi = np.linspace(0, 2*pi, n_phi, endpoint=False)
    theta_grid, phi_grid = np.meshgrid(theta, phi, indexing='ij')
    theta_flat = theta_grid.flatten()
    phi_flat = phi_grid.flatten()
    # Weights: sin(theta) * dtheta * dphi
    dtheta = pi / (n_theta - 1)
    dphi = 2 * pi / n_phi
    weights = np.sin(theta_flat) * dtheta * dphi
    # Normalize weights
    weights /= weights.sum()
    return theta_flat, phi_flat, weights

def _optimize_density(L, n_theta, n_phi, maxiter=200, lr=0.01):
    """
    Optimize spherical harmonic coefficients to minimize continuous log-energy.
    """
    theta, phi, weights = _build_quadrature_grid(n_theta, n_phi)
    n_quad = len(theta)
    
    # Precompute spherical harmonics at quadrature points
    Y = _spherical_harmonics(theta, phi, L)  # shape (n_quad, (L+1)^2)
    # Y[0] corresponds to l=0, m=0 (constant)
    
    # Number of coefficients (excluding l=0,m=0 which is fixed)
    n_coeffs = (L + 1)**2 - 1  # exclude the constant term
    
    # Initialize coefficients to zero (uniform density)
    c = np.zeros(n_coeffs)
    
    def _density_from_coeffs(c):
        """Compute density at quadrature points from coefficients."""
        # rho = (1 + sum_{l>=1} c_{lm} Y_{lm})^2
        # Y[0] is l=0,m=0 = 1/sqrt(4*pi)
        # We want rho to be a probability density: ∫ rho dΩ = 1
        # Let f = 1 + sum_{l>=1} c_{lm} Y_{lm}
        # rho = f^2 / (4*pi)  so that ∫ rho dΩ = 1 if ∫ f^2 dΩ = 4*pi
        # Actually, let's just compute rho = f^2 and normalize
        f = 1.0 + np.sum(c[:, None] * Y[:, 1:], axis=1)
        rho = f**2
        # Ensure positivity (should be by construction)
        rho = np.maximum(rho, 1e-15)
        # Normalize
        rho_int = np.sum(rho * weights)
        rho = rho / rho_int
        return rho
    
    def _continuous_energy(c):
        """Compute continuous energy E = ∫∫ log(1/|x-y|) rho(x) rho(y) dΩ_x dΩ_y."""
        rho = _density_from_coeffs(c)
        # Compute pairwise distances
        # x_i = (sin(theta_i)cos(phi_i), sin(theta_i)sin(phi_i), cos(theta_i))
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        pts = np.column_stack([x, y, z])
        
        # Compute pairwise distances
        n = len(pts)
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    dists[i, j] = np.linalg.norm(pts[i] - pts[j])
        
        # Compute kernel K_ij = log(1/d_ij) for i != j, and 0 for i=j
        K = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j and dists[i, j] > 1e-15:
                    K[i, j] = log(1.0 / dists[i, j])
        
        # E = sum_i sum_j rho_i rho_j K_ij * w_i * w_j
        # Use vectorized computation
        E = np.sum(rho[:, None] * rho[None, :] * K * weights[:, None] * weights[None, :])
        return E
    
    def _energy_grad_density(c):
        """Compute gradient of continuous energy with respect to coefficients."""
        # Use finite differences for simplicity (slow but correct)
        grad = np.zeros(n_coeffs)
        eps = 1e-6
        for i in range(n_coeffs):
            c_plus = c.copy()
            c_plus[i] += eps
            c_minus = c.copy()
            c_minus[i] -= eps
            grad[i] = (_continuous_energy(c_plus) - _continuous_energy(c_minus)) / (2*eps)
        return grad
    
    # Gradient descent
    for it in range(maxiter):
        if time.time() - start_time > timeout:
            break
        e = _continuous_energy(c)
        if it % 20 == 0:
            print(f"  Density opt iter {it}, E={e:.6f}")
        
        # Compute gradient (use finite differences)
        g = _energy_grad_density(c)
        
        # Update with learning rate
        c = c - lr * g
        
        # Decay learning rate
        lr = max(0.001, lr * 0.99)
    
    return c

def _sample_points_from_density(c, L, n_points, seed=42):
    """Sample n_points from the density defined by coefficients c."""
    rng = np.random.RandomState(seed)
    
    # Use a fine grid for sampling
    n_theta = 200
    n_phi = 400
    theta, phi, weights = _build_quadrature_grid(n_theta, n_phi)
    
    Y = _spherical_harmonics(theta, phi, L)
    f = 1.0 + np.sum(c[:, None] * Y[:, 1:], axis=1)
    rho = f**2
    rho = np.maximum(rho, 1e-15)
    
    # Normalize to get probability weights
    prob = rho * weights
    prob = prob / prob.sum()
    
    # Sample indices
    indices = rng.choice(len(prob), size=n_points, replace=False, p=prob)
    
    # Convert to Cartesian
    x = np.sin(theta[indices]) * np.cos(phi[indices])
    y = np.sin(theta[indices]) * np.sin(phi[indices])
    z = np.cos(theta[indices])
    return np.column_stack([x, y, z])

def _fibonacci_spiral():
    """Generate Fibonacci spiral points on the unit sphere."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _optimize_points(init_points, maxiter=1500):
    """Optimize point positions to minimize log-energy using L-BFGS-B."""
    pts = init_points.copy().astype(np.float64)
    
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    theta = np.arccos(np.clip(z, -1.0, 1.0))
    phi = np.arctan2(y, x)
    
    def _spherical_to_cart(theta, phi):
        sin_t = np.sin(theta)
        return np.column_stack([
            sin_t * np.cos(phi),
            sin_t * np.sin(phi),
            np.cos(theta)
        ])
    
    def _objective(params):
        th = params[:N]
        ph = params[N:]
        p = _spherical_to_cart(th, ph)
        return _energy(p)
    
    def _grad(params):
        th = params[:N]
        ph = params[N:]
        p = _spherical_to_cart(th, ph)
        g_cart = _energy_grad(p)
        sin_t = np.sin(th)
        cos_t = np.cos(th)
        sin_p = np.sin(ph)
        cos_p = np.cos(ph)
        
        dtheta = np.column_stack([cos_t * cos_p, cos_t * sin_p, -sin_t])
        g_th = np.sum(g_cart * dtheta, axis=1)
        
        dphi = np.column_stack([-sin_p, cos_p, np.zeros(N)])
        g_ph = np.sum(g_cart * dphi, axis=1)
        
        return np.concatenate([g_th, g_ph])
    
    params0 = np.concatenate([theta, phi])
    
    result = minimize(_objective, params0, method='L-BFGS-B',
                      jac=_grad, options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-10})
    
    best_params = result.x
    best_points = _spherical_to_cart(best_params[:N], best_params[N:])
    best_energy = result.fun
    
    return best_points, best_energy

def construct():
    """
    Construct 100 points on the unit sphere minimizing logarithmic energy.
    Uses spherical harmonic density optimization followed by point refinement.
    """
    global start_time, timeout
    start_time = time.time()
    timeout = 100.0
    
    best_points = None
    best_energy = np.inf
    
    # Step 1: Optimize density using spherical harmonics
    L = 6  # degree of spherical harmonic expansion
    try:
        c_opt = _optimize_density(L, n_theta=100, n_phi=200, maxiter=50, lr=0.05)
        # Step 2: Sample points from optimized density
        pts_density = _sample_points_from_density(c_opt, L, N, seed=42)
        pts_density = _project_to_sphere(pts_density)
        e_density = _energy(pts_density)
        print(f"Density-sampled energy: {e_density:.6f}")
        
        # Step 3: Refine with direct point optimization
        pts_refined, e_refined = _optimize_points(pts_density, maxiter=800)
        pts_refined = _project_to_sphere(pts_refined)
        e_refined = _energy(pts_refined)
        print(f"Refined energy: {e_refined:.6f}")
        
        if e_refined < best_energy:
            best_energy = e_refined
            best_points = pts_refined.copy()
    except Exception as e:
        print(f"Density optimization failed: {e}")
    
    # Step 4: Also try from Fibonacci spiral as fallback/alternative
    if time.time() - start_time < timeout:
        try:
            pts_fib = _fibonacci_spiral()
            pts_fib_opt, e_fib = _optimize_points(pts_fib, maxiter=800)
            pts_fib_opt = _project_to_sphere(pts_fib_opt)
            e_fib = _energy(pts_fib_opt)
            print(f"Fibonacci-refined energy: {e_fib:.6f}")
            
            if e_fib < best_energy:
                best_energy = e_fib
                best_points = pts_fib_opt.copy()
        except Exception as e:
            print(f"Fibonacci optimization failed: {e}")
    
    if best_points is None:
        best_points = _fibonacci_spiral()
    
    # Final validation
    norms = np.linalg.norm(best_points, axis=1)
    if np.any(norms < 1e-9) or not np.all(np.isfinite(best_points)):
        best_points = _fibonacci_spiral()
    
    assert best_points.shape == (N, 3)
    return best_points
