"""Optimized logarithmic energy minimization for 100 points on the unit sphere."""
import numpy as np
from scipy.optimize import minimize
import time
import math

N = 100
DEADLINE = 145.0  # Safety margin before 150s kill


def _energy_and_grad(points):
    """Compute logarithmic energy and gradient for points on unit sphere.
    
    Energy = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    Using chordal distance |x_i - x_j| = sqrt(2 - 2*x_i·x_j)
    """
    n = points.shape[0]
    # Compute pairwise dot products
    dots = points @ points.T
    # Clip to valid range for numerical stability
    dots = np.clip(dots, -1.0, 1.0)
    # Chordal distances squared: |xi - xj|^2 = 2 - 2*dot
    d2 = 2.0 - 2.0 * dots
    np.fill_diagonal(d2, 1.0)  # avoid log(0)
    
    # log(1/d) = -log(d) = -0.5*log(d2)
    log_d2 = np.log(d2)
    # Energy = sum_{i<j} -0.5 * log(d2_ij)
    # Sum over upper triangle
    iu = np.triu_indices(n, k=1)
    energy = -0.5 * np.sum(log_d2[iu])
    
    # Gradient: d/dx_i sum_{j!=i} -0.5*log(2 - 2*x_i·x_j)
    # = sum_{j!=i} (x_i - x_j) / (2 - 2*x_i·x_j)
    # = sum_{j!=i} (x_i - x_j) / d2_ij
    # But we need to project onto tangent space (remove radial component)
    
    # Compute gradient
    grad = np.zeros_like(points)
    # For each point i, grad_i = sum_{j!=i} (x_i - x_j) / d2_ij
    # Vectorized: 
    # diff[i,j] = x_i - x_j, shape (n,n,3)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    # d2 shape (n,n)
    # grad[i] = sum_j (diff[i,j] / d2[i,j]) for j != i
    # Set diagonal to 0 to avoid self-interaction
    d2_safe = d2.copy()
    np.fill_diagonal(d2_safe, 1.0)
    inv_d2 = 1.0 / d2_safe
    inv_d2[np.eye(n, dtype=bool)] = 0.0  # zero out diagonal
    
    # grad[i] = sum_j diff[i,j] * inv_d2[i,j]
    # diff shape (n,n,3), inv_d2 shape (n,n)
    grad = np.einsum('ij,ijc->c', inv_d2, diff)
    
    # Project gradient onto tangent space: grad_i -= (grad_i · x_i) * x_i
    radial = np.sum(grad * points, axis=1, keepdims=True)
    grad = grad - radial * points
    
    return energy, grad


def _energy_only(points):
    """Just the energy value (no gradient)."""
    energy, _ = _energy_and_grad(points)
    return energy


def _normalize(points):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return points / norms


def _fibonacci_spiral(n, offset=0.5, scale=1.0):
    """Generate Fibonacci spiral points on unit sphere."""
    i = np.arange(n) + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i * scale
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _normalize(pts)


def _random_points(n, rng):
    """Generate random points on unit sphere."""
    pts = rng.standard_normal((n, 3))
    return _normalize(pts)


def _shifted_spiral(n, shift):
    """Generate a shifted spiral pattern."""
    i = np.arange(n) + 0.5 + shift
    z = 1.0 - 2.0 * np.clip(i / n, 0, 1)
    phi = np.pi * (1.0 + np.sqrt(5.0)) * (i + shift * n)
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _normalize(pts)


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initializations
    initializations = []
    
    # 1. Standard Fibonacci spiral
    initializations.append(_fibonacci_spiral(N))
    
    # 2. Perturbed Fibonacci spiral
    base = _fibonacci_spiral(N)
    perturbed = base + 0.1 * rng.standard_normal((N, 3))
    initializations.append(_normalize(perturbed))
    
    # 3. Shifted spirals
    for shift in [0.1, 0.3, 0.7]:
        initializations.append(_shifted_spiral(N, shift))
    
    # 4. Random points
    for _ in range(3):
        initializations.append(_random_points(N, rng))
    
    # 5. Anti-symmetric pairs (better distribution)
    half = N // 2
    pts_half = _random_points(half, rng)
    pts_full = np.vstack([pts_half, -pts_half])
    if N % 2 == 1:
        pts_full = np.vstack([pts_full, np.array([[0, 0, 1.0]])])
    initializations.append(_normalize(pts_full))
    
    # Optimize each initialization
    best_points = initializations[0]
    best_energy = _energy_only(best_points)
    
    for init_idx, init_pts in enumerate(initializations):
        if time.time() > deadline - 5.0:
            break
        
        # Phase 1: Coarse optimization with L-BFGS-B
        try:
            result = minimize(
                fun=lambda x: _energy_only(_normalize(x)),
                x0=init_pts.flatten(),
                jac=lambda x: _energy_and_grad(_normalize(x))[1].flatten(),
                method='L-BFGS-B',
                options={
                    'maxiter': 500,
                    'ftol': 1e-12,
                    'gtol': 1e-8,
                    'maxls': 50
                }
            )
            if result.fun < best_energy:
                best_energy = result.fun
                best_points = _normalize(result.x.reshape(N, 3))
        except Exception:
            continue
        
        if time.time() > deadline - 10.0:
            break
        
        # Phase 2: Fine optimization with tighter tolerances
        try:
            result2 = minimize(
                fun=lambda x: _energy_only(_normalize(x)),
                x0=best_points.flatten(),
                jac=lambda x: _energy_and_grad(_normalize(x))[1].flatten(),
                method='L-BFGS-B',
                options={
                    'maxiter': 1000,
                    'ftol': 1e-15,
                    'gtol': 1e-11,
                    'maxls': 50
                }
            )
            if result2.fun < best_energy:
                best_energy = result2.fun
                best_points = _normalize(result2.x.reshape(N, 3))
        except Exception:
            continue
    
    # Final high-precision polish pass
    if time.time() < deadline - 5.0:
        try:
            result_final = minimize(
                fun=lambda x: _energy_only(_normalize(x)),
                x0=best_points.flatten(),
                jac=lambda x: _energy_and_grad(_normalize(x))[1].flatten(),
                method='L-BFGS-B',
                options={
                    'maxiter': 2000,
                    'ftol': 1e-16,
                    'gtol': 1e-13,
                    'maxls': 50
                }
            )
            if result_final.fun < best_energy:
                best_energy = result_final.fun
                best_points = _normalize(result_final.x.reshape(N, 3))
        except Exception:
            pass
    
    # Ensure valid output
    best_points = _normalize(best_points)
    # Check for any points too close together
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(best_points[i] - best_points[j])
            if d < 1e-9:
                # Nudge apart
                best_points[j] = best_points[j] + 1e-8 * rng.standard_normal(3)
                best_points = _normalize(best_points)
    
    return best_points
