"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import math
import time
import numpy as np
from scipy.optimize import minimize

def _log_energy(pts):
    """Compute logarithmic energy sum_{i<j} log(1/d_ij)"""
    n = len(pts)
    total = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-9:
                return float('inf')
            total += math.log(1.0 / d)
    return total

def _log_energy_fast(pts):
    """Vectorized logarithmic energy computation"""
    n = len(pts)
    # Compute all pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    # Mask diagonal and lower triangle
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    upper_dists = dists[mask]
    if np.any(upper_dists < 1e-9):
        return float('inf')
    return float(np.sum(np.log(1.0 / upper_dists)))

def _log_energy_grad(pts):
    """Compute gradient of logarithmic energy w.r.t. point coordinates"""
    n = len(pts)
    grad = np.zeros_like(pts)
    for i in range(n):
        for j in range(i + 1, n):
            diff = pts[i] - pts[j]
            d = np.linalg.norm(diff)
            if d < 1e-9:
                return None
            # d/dx_i log(1/d) = -1/d * d/dx_i d = -1/d * (x_i - x_j)/d = -(x_i - x_j)/d^2
            # d/dx_j log(1/d) = -1/d * d/dx_j d = -1/d * (x_j - x_i)/d = (x_i - x_j)/d^2
            grad[i] += -diff / (d * d)
            grad[j] += diff / (d * d)
    return grad

def _normalize(pts):
    """Normalize points to unit sphere"""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms

def _project_to_sphere(pts):
    """Project points to unit sphere (re-normalize)"""
    return _normalize(pts)

def _exponential_map(pts, tangent):
    """Exponential map on sphere: move points along geodesics"""
    result = np.zeros_like(pts)
    for i in range(len(pts)):
        p = pts[i]
        v = tangent[i]
        # Remove radial component to get true tangent
        v_tan = v - np.dot(v, p) * p
        theta = np.linalg.norm(v_tan)
        if theta < 1e-10:
            result[i] = p
        else:
            v_hat = v_tan / theta
            result[i] = np.cos(theta) * p + np.sin(theta) * v_hat
    return _normalize(result)

def _riemannian_cg(pts, max_iter=500, tol=1e-8):
    """Riemannian conjugate gradient method for sphere optimization"""
    pts = _normalize(pts.copy())
    current_energy = _log_energy_fast(pts)
    
    # Initial gradient
    grad = _log_energy_grad(pts)
    if grad is None:
        return pts, current_energy
    
    # Project gradient to tangent space
    def project_grad(p, g):
        return g - np.sum(g * p, axis=1, keepdims=True) * p
    
    g_tan = project_grad(pts, grad)
    
    # Conjugate gradient parameters
    p_vec = -g_tan
    old_g_tan = g_tan.copy()
    
    for iteration in range(max_iter):
        # Line search
        alpha = 0.1
        for ls_iter in range(30):
            new_pts = _exponential_map(pts, alpha * p_vec)
            new_energy = _log_energy_fast(new_pts)
            if new_energy < current_energy:
                break
            alpha *= 0.5
        
        if alpha < 1e-10:
            break
        
        pts = new_pts
        current_energy = new_energy
        
        # Compute new gradient
        grad = _log_energy_grad(pts)
        if grad is None:
            break
        new_g_tan = project_grad(pts, grad)
        
        # Check convergence
        if np.linalg.norm(new_g_tan) < tol:
            break
        
        # Update conjugate direction
        beta = np.sum(new_g_tan * (new_g_tan - old_g_tan)) / (np.sum(old_g_tan * old_g_tan) + 1e-12)
        p_vec = -new_g_tan + beta * p_vec
        
        old_g_tan = new_g_tan
    
    return pts, current_energy

def _lbfgs_optimize(pts, maxiter=1000, ftol=1e-10, gtol=1e-10):
    """L-BFGS optimization with sphere constraint via penalty or projection"""
    pts = _normalize(pts.copy())
    
    def energy_and_grad(x):
        pts_3d = x.reshape(-1, 3)
        # Soft penalty for points not on sphere
        norms = np.linalg.norm(pts_3d, axis=1)
        penalty = 100.0 * np.sum((norms - 1.0)**2)
        energy = _log_energy_fast(_normalize(pts_3d))
        if energy == float('inf'):
            return energy + penalty, np.zeros_like(x)
        
        # Gradient
        grad_3d = _log_energy_grad(_normalize(pts_3d))
        if grad_3d is None:
            return energy + penalty, np.zeros_like(x)
        # Add penalty gradient
        penalty_grad = 200.0 * (norms - 1.0) * pts_3d / (norms + 1e-12)
        total_grad = grad_3d + penalty_grad
        return energy + penalty, total_grad.flatten()
    
    result = minimize(energy_and_grad, pts.flatten(), method='L-BFGS-B',
                      options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol})
    return _normalize(result.x.reshape(-1, 3)), result.fun

def _fibonacci_sphere(n):
    """Generate Fibonacci spiral points on sphere"""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _halton_sphere(n):
    """Generate Halton sequence points on sphere"""
    def halton(index, base):
        f = 1.0
        r = 0.0
        while index > 0:
            f /= base
            r += f * (index % base)
            index //= base
        return r
    
    points = np.zeros((n, 3))
    for i in range(n):
        u = halton(i + 1, 2)
        v = halton(i + 1, 3)
        z = 1.0 - 2.0 * v
        theta = 2.0 * np.pi * u
        r = np.sqrt(1.0 - z * z)
        points[i] = [r * np.cos(theta), r * np.sin(theta), z]
    return points

def _polish(pts, max_iter=200, tol=1e-10):
    """High-precision polish pass with tight tolerances"""
    pts = _normalize(pts.copy())
    
    # Use L-BFGS with very tight tolerances
    pts, energy = _lbfgs_optimize(pts, maxiter=max_iter, ftol=tol, gtol=tol)
    
    # Additional CG refinement
    pts, energy = _riemannian_cg(pts, max_iter=max_iter // 2, tol=tol / 10)
    
    return pts, energy

def construct():
    """Main function to construct optimal point configuration"""
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s limit
    
    # Generate initial configurations
    best_pts = _fibonacci_sphere(N)
    best_energy = _log_energy_fast(best_pts)
    
    # Try different initializations and optimize
    initializations = [
        _fibonacci_sphere(N),
        _halton_sphere(N),
    ]
    
    # Add some random perturbations for diversity
    for seed in range(3):
        rng = np.random.RandomState(seed)
        random_pts = rng.randn(N, 3)
        initializations.append(_normalize(random_pts))
    
    for init_pts in initializations:
        if time.time() > deadline:
            break
        
        # Riemannian CG optimization
        try:
            opt_pts, energy = _riemannian_cg(init_pts, max_iter=300, tol=1e-8)
            if energy < best_energy:
                best_energy = energy
                best_pts = opt_pts
        except Exception:
            pass
        
        # L-BFGS optimization
        if time.time() > deadline:
            break
        try:
            opt_pts, energy = _lbfgs_optimize(init_pts, maxiter=500, ftol=1e-9, gtol=1e-9)
            if energy < best_energy:
                best_energy = energy
                best_pts = opt_pts
        except Exception:
            pass
    
    # Final polish pass
    if time.time() < deadline - 20.0:
        try:
            best_pts, best_energy = _polish(best_pts, max_iter=100, tol=1e-10)
        except Exception:
            pass
    
    # Ensure valid output
    best_pts = _normalize(best_pts)
    
    # Final validation
    if not np.all(np.isfinite(best_pts)):
        return _fibonacci_sphere(N)
    
    return best_pts
# EVOLVE-BLOCK-END
