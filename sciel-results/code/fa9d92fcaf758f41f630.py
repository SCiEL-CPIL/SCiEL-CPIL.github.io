"""Improved logarithmic energy minimization on the sphere using spectral methods."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
from math import fsum, log, sqrt
import time


def _pairwise_distances(points):
    """Compute pairwise chordal distances for points on unit sphere."""
    dots = points @ points.T
    d2 = 2.0 - 2.0 * dots
    np.clip(d2, 0.0, 4.0, out=d2)
    d = np.sqrt(d2)
    return d


def _log_energy(points):
    """Compute logarithmic energy sum_{i<j} log(1/|x_i - x_j|)."""
    d = _pairwise_distances(points)
    iu = np.triu_indices(N, k=1)
    dists = d[iu]
    dists = np.maximum(dists, 1e-15)
    vals = -np.log(dists)
    return fsum(vals.tolist())


def _log_energy_grad(points):
    """Compute gradient of log energy w.r.t. points (in R^3)."""
    d2 = 2.0 - 2.0 * (points @ points.T)
    np.clip(d2, 1e-30, 4.0, out=d2)
    M = 1.0 / d2
    np.fill_diagonal(M, 0.0)
    row_sums = M.sum(axis=1, keepdims=True)
    grad = row_sums * points - M @ points
    return grad


def _normalize(points):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return points / norms


def _project_gradient(points, grad):
    """Project gradient onto tangent space of unit sphere."""
    dots = (points * grad).sum(axis=1, keepdims=True)
    return grad - dots * points


def _spherical_harmonics(l_max, n_theta, n_phi):
    """Compute real spherical harmonics on a grid.
    Returns: array of shape (n_theta, n_phi, n_coeffs)
    where n_coeffs = (l_max+1)^2
    """
    from scipy.special import lpmv, gamma
    
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    theta_2d, phi_2d = np.meshgrid(theta, phi, indexing='ij')
    
    n_coeffs = (l_max + 1) ** 2
    coeffs = np.zeros((n_theta, n_phi, n_coeffs))
    
    idx = 0
    for l in range(l_max + 1):
        for m in range(-l, l + 1):
            # Real spherical harmonics
            if m == 0:
                # Y_l0 = sqrt((2l+1)/(4pi)) * P_l(cos(theta))
                P_l = lpmv(0, l, np.cos(theta_2d))
                norm = np.sqrt((2*l + 1) / (4 * np.pi))
                coeffs[:, :, idx] = norm * P_l
            elif m > 0:
                # Y_lm = sqrt(2) * sqrt((2l+1)/(4pi) * (l-m)!/(l+m)!) * P_l^m(cos(theta)) * cos(m*phi)
                P_lm = lpmv(m, l, np.cos(theta_2d))
                norm = np.sqrt(2) * np.sqrt((2*l + 1) / (4 * np.pi) * gamma(l - m + 1) / gamma(l + m + 1))
                coeffs[:, :, idx] = norm * P_lm * np.cos(m * phi_2d)
            else:
                # Y_l,-m = sqrt(2) * sqrt((2l+1)/(4pi) * (l-|m|)!/(l+|m|)!) * P_l^{|m|}(cos(theta)) * sin(|m|*phi)
                m_abs = -m
                P_lm = lpmv(m_abs, l, np.cos(theta_2d))
                norm = np.sqrt(2) * np.sqrt((2*l + 1) / (4 * np.pi) * gamma(l - m_abs + 1) / gamma(l + m_abs + 1))
                coeffs[:, :, idx] = norm * P_lm * np.sin(m_abs * phi_2d)
            idx += 1
    
    return coeffs, theta, phi


def _init_coefficients(l_max=8, n_theta=64, n_phi=128, seed=42):
    """Initialize spherical harmonic coefficients to approximate uniform distribution."""
    # For uniform distribution, only l=0, m=0 is nonzero
    n_coeffs = (l_max + 1) ** 2
    coeffs = np.zeros(n_coeffs)
    # Y_00 = 1/sqrt(4*pi), so coefficient should be sqrt(4*pi) for density = 1
    coeffs[0] = np.sqrt(4 * np.pi)
    
    # Add small random perturbations to break symmetry
    rng = np.random.default_rng(seed)
    coeffs[1:] += rng.standard_normal(n_coeffs - 1) * 0.01
    
    return coeffs


def _coeffs_to_density(coeffs, l_max, n_theta, n_phi):
    """Convert spherical harmonic coefficients to density on grid."""
    basis, theta, phi = _spherical_harmonics(l_max, n_theta, n_phi)
    # basis shape: (n_theta, n_phi, n_coeffs)
    density = np.tensordot(basis, coeffs, axes=([2], [0]))
    # Ensure non-negative density
    density = np.maximum(density, 0.0)
    # Normalize to integrate to 1
    # Integral over sphere: sum density * sin(theta) * dtheta * dphi
    dtheta = np.pi / (n_theta - 1)
    dphi = 2 * np.pi / n_phi
    integral = np.sum(density * np.sin(theta)[:, None]) * dtheta * dphi
    if integral > 1e-15:
        density /= integral
    return density, theta, phi


def _sample_points_from_density(density, theta, phi, n_points, seed=42):
    """Sample points from a density distribution on the sphere."""
    rng = np.random.default_rng(seed)
    
    n_theta, n_phi = density.shape
    # Flatten density
    flat_density = density.flatten()
    # Normalize to probability
    total = flat_density.sum()
    if total < 1e-15:
        # Fallback to uniform
        theta_idx = rng.integers(0, n_theta, size=n_points)
        phi_idx = rng.integers(0, n_phi, size=n_points)
    else:
        probs = flat_density / total
        # Sample from distribution
        indices = rng.choice(n_theta * n_phi, size=n_points, p=probs)
        theta_idx = indices // n_phi
        phi_idx = indices % n_phi
    
    # Get coordinates
    theta_vals = theta[theta_idx]
    phi_vals = phi[phi_idx]
    
    # Convert to Cartesian
    x = np.sin(theta_vals) * np.cos(phi_vals)
    y = np.sin(theta_vals) * np.sin(phi_vals)
    z = np.cos(theta_vals)
    
    points = np.column_stack([x, y, z])
    return points


def _spectral_gradient_descent(coeffs, l_max, n_theta, n_phi, lr, n_iters, momentum=0.9, seed=42):
    """Gradient descent on spherical harmonic coefficients."""
    rng = np.random.default_rng(seed)
    v = np.zeros_like(coeffs)
    best_energy = np.inf
    best_coeffs = coeffs.copy()
    best_points = None
    
    for it in range(n_iters):
        # Convert coefficients to density and sample points
        density, theta, phi = _coeffs_to_density(coeffs, l_max, n_theta, n_phi)
        points = _sample_points_from_density(density, theta, phi, N, seed=seed + it)
        points = _normalize(points)
        
        # Compute energy
        energy = _log_energy(points)
        
        # Track best
        if energy < best_energy:
            best_energy = energy
            best_coeffs = coeffs.copy()
            best_points = points.copy()
        
        # Compute gradient of energy w.r.t. coefficients
        # This requires computing how points depend on coefficients
        # Use finite differences for simplicity (expensive but works)
        grad = np.zeros_like(coeffs)
        eps = 1e-4
        for i in range(len(coeffs)):
            coeffs_plus = coeffs.copy()
            coeffs_plus[i] += eps
            density_plus, _, _ = _coeffs_to_density(coeffs_plus, l_max, n_theta, n_phi)
            points_plus = _sample_points_from_density(density_plus, theta, phi, N, seed=seed + it)
            points_plus = _normalize(points_plus)
            energy_plus = _log_energy(points_plus)
            grad[i] = (energy_plus - energy) / eps
        
        # Gradient descent with momentum
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-12:
            break
        
        v = momentum * v - lr * grad / (grad_norm + 1e-15)
        coeffs = coeffs + v
        
        # Decay learning rate
        lr_eff = lr * (1.0 - it / (2.0 * n_iters))
    
    return best_coeffs, best_energy, best_points


def _gradient_descent(points, lr, n_iters, momentum=0.9, lr_decay=1.0, min_dist=1e-9):
    """Gradient descent on the sphere with momentum."""
    points = points.copy()
    points = _normalize(points)
    v = np.zeros_like(points)
    best_energy = _log_energy(points)
    best_points = points.copy()
    
    for it in range(n_iters):
        energy = _log_energy(points)
        grad = _log_energy_grad(points)
        grad = _project_gradient(points, grad)
        
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-12:
            break
        
        v = momentum * v - lr * lr_decay * grad / (grad_norm + 1e-15)
        points = points + v
        points = _normalize(points)
        
        d = _pairwise_distances(points)
        if d.min() < min_dist:
            iu = np.triu_indices(N, k=1)
            close = d[iu] < min_dist * 10
            for idx in range(len(close)):
                if close[idx]:
                    i, j = iu[0][idx], iu[1][idx]
                    mid = (points[i] + points[j]) / 2.0
                    mid_norm = np.linalg.norm(mid)
                    if mid_norm > 1e-15:
                        mid = mid / mid_norm
                    perp = points[i] - points[j]
                    perp = perp - np.dot(perp, mid) * mid
                    perp_norm = np.linalg.norm(perp)
                    if perp_norm < 1e-15:
                        perp = np.array([1.0, 0.0, 0.0])
                        perp = perp - np.dot(perp, mid) * mid
                        perp_norm = np.linalg.norm(perp)
                    perp = perp / perp_norm
                    points[i] = _normalize(points[i] + 0.01 * perp)
                    points[j] = _normalize(points[j] - 0.01 * perp)
        
        if energy < best_energy:
            best_energy = energy
            best_points = points.copy()
        
        lr_decay = max(0.999, 1.0 - it / (2.0 * n_iters))
    
    return best_points, best_energy


def construct():
    # Start with Fibonacci spiral as initial guess
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    points = _normalize(points)
    
    # Phase 1: Cartesian gradient descent with larger steps
    points, e1 = _gradient_descent(points, lr=0.005, n_iters=500, momentum=0.9)
    
    # Phase 2: Spectral optimization to escape local minima
    l_max = 6
    n_theta = 48
    n_phi = 96
    coeffs = _init_coefficients(l_max, n_theta, n_phi, seed=42)
    coeffs, e_spectral, points_spectral = _spectral_gradient_descent(
        coeffs, l_max, n_theta, n_phi, lr=0.01, n_iters=100, momentum=0.9, seed=42
    )
    
    # Use the better of the two
    if e_spectral < e1:
        points = points_spectral
        e1 = e_spectral
    
    # Phase 3: Fine Cartesian gradient descent
    points, e2 = _gradient_descent(points, lr=0.001, n_iters=1000, momentum=0.9)
    
    # Phase 4: Very fine polish
    points, e3 = _gradient_descent(points, lr=0.0001, n_iters=2000, momentum=0.95)
    
    # Phase 5: Final polish
    points, e4 = _gradient_descent(points, lr=0.00005, n_iters=1000, momentum=0.99)
    
    # Ensure valid output
    points = _normalize(points)
    
    # Verify no points are too close
    d = _pairwise_distances(points)
    if d.min() < 1e-9:
        # Fallback to Fibonacci spiral
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
        points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        points = _normalize(points)
    
    return points
# EVOLVE-BLOCK-END
