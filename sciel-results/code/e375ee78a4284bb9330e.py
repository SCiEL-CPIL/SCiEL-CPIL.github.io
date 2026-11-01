"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def _log_energy(pts):
    """Compute logarithmic energy: sum_{i<j} log(1/|x_i - x_j|)."""
    n = len(pts)
    total = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-15:
                return float('inf')
            total += math.log(1.0 / d)
    return total

def _log_energy_fast(pts):
    """Vectorized log energy computation."""
    n = len(pts)
    # Compute all pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    # Only upper triangle
    iu = np.triu_indices(n, k=1)
    d = dists[iu]
    d = np.maximum(d, 1e-15)
    return np.sum(np.log(1.0 / d))

def _project_to_sphere(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def _grad_log_energy(pts):
    """
    Gradient of the log energy on the sphere.
    E = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log|x_i - x_j|
    dE/dx_i = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    Projected to tangent space: grad_i = grad_i - (grad_i . x_i) * x_i
    """
    n = len(pts)
    grad = np.zeros_like(pts)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d = pts[i] - pts[j]
            dist = np.linalg.norm(d)
            if dist < 1e-15:
                continue
            grad[i] += d / (dist * dist)
    # Project to tangent space
    for i in range(n):
        grad[i] -= np.dot(grad[i], pts[i]) * pts[i]
    return grad

def _grad_log_energy_vec(pts):
    """Vectorized gradient computation."""
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (n,n,3)
    dists = np.sqrt(np.sum(diff * diff, axis=2))  # (n,n)
    np.fill_diagonal(dists, 1.0)  # avoid division by zero
    # grad[i] = sum_j (x_i - x_j) / |x_i - x_j|^2
    # = sum_j diff[i,j] / dists[i,j]^2
    inv_d2 = 1.0 / (dists * dists)  # (n,n)
    inv_d2[np.diag_indices(n)] = 0.0
    grad = np.sum(diff * inv_d2[:, :, np.newaxis], axis=1)  # (n,3)
    # Project to tangent space
    for i in range(n):
        grad[i] -= np.dot(grad[i], pts[i]) * pts[i]
    return grad

def _riemannian_cg(pts, max_iter=500, tol=1e-10, time_limit=None, start_time=None):
    """
    Riemannian conjugate gradient on the sphere.
    """
    n = len(pts)
    pts = _project_to_sphere(pts.copy())
    energy = _log_energy_fast(pts)
    
    if time_limit is not None and start_time is None:
        start_time = time.time()
    
    # Initial gradient
    grad = _grad_log_energy_vec(pts)
    # Project gradient to tangent space (already done in _grad_log_energy_vec)
    
    # Initial search direction
    d = -grad.copy()
    
    best_pts = pts.copy()
    best_energy = energy
    
    for it in range(max_iter):
        if time_limit is not None and (time.time() - start_time) > time_limit - 5:
            break
        
        # Check convergence
        gnorm = np.max(np.linalg.norm(grad, axis=1))
        if gnorm < tol:
            break
        
        # Line search with backtracking
        alpha = 0.01
        for ls_iter in range(30):
            # Move along geodesic (spherical interpolation)
            # x_new = cos(alpha) * x + sin(alpha) * (d / |d|)
            d_norm = np.linalg.norm(d, axis=1, keepdims=True)
            d_norm = np.maximum(d_norm, 1e-15)
            d_unit = d / d_norm
            
            # Spherical interpolation
            cos_a = np.cos(alpha)
            sin_a = np.sin(alpha)
            pts_new = cos_a * pts + sin_a * d_unit
            pts_new = _project_to_sphere(pts_new)
            
            e_new = _log_energy_fast(pts_new)
            if e_new < energy:
                break
            alpha *= 0.5
        
        if e_new >= energy:
            # Try smaller step
            alpha = 0.001
            for ls_iter in range(20):
                d_norm = np.linalg.norm(d, axis=1, keepdims=True)
                d_norm = np.maximum(d_norm, 1e-15)
                d_unit = d / d_norm
                cos_a = np.cos(alpha)
                sin_a = np.sin(alpha)
                pts_new = cos_a * pts + sin_a * d_unit
                pts_new = _project_to_sphere(pts_new)
                e_new = _log_energy_fast(pts_new)
                if e_new < energy:
                    break
                alpha *= 0.5
        
        if e_new >= energy:
            # No improvement, try gradient descent step
            alpha = 0.001
            for ls_iter in range(20):
                pts_new = pts - alpha * grad
                pts_new = _project_to_sphere(pts_new)
                e_new = _log_energy_fast(pts_new)
                if e_new < energy:
                    break
                alpha *= 0.5
        
        if e_new >= energy:
            break
        
        # Update
        old_grad = grad
        pts = pts_new
        energy = e_new
        
        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
        
        # Compute new gradient
        grad = _grad_log_energy_vec(pts)
        
        # Fletcher-Reeves beta with damping
        beta = np.max(np.sum(grad * grad, axis=1)) / np.max(np.sum(old_grad * old_grad, axis=1) + 1e-15)
        beta = min(beta, 1.0)
        
        d = -grad + beta * d
        # Ensure d is in tangent space
        for i in range(n):
            d[i] -= np.dot(d[i], pts[i]) * pts[i]
    
    return best_pts, best_energy

def _perturb_and_optimize(pts, rng, time_limit, start_time):
    """Perturb points and optimize."""
    # Add random perturbation
    perturbed = pts + rng.normal(0, 0.05, pts.shape)
    perturbed = _project_to_sphere(perturbed)
    
    remaining = time_limit - (time.time() - start_time)
    if remaining < 5:
        return pts, _log_energy_fast(pts)
    
    opt_pts, opt_energy = _riemannian_cg(perturbed, max_iter=200, tol=1e-9, time_limit=remaining, start_time=start_time)
    return opt_pts, opt_energy

def _golden_spiral(offset=0.5):
    """Generate golden angle spiral points with optional offset."""
    i = np.arange(N) + offset
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def construct():
    start_time = time.time()
    time_limit = 120.0  # Safety margin below 150s
    
    rng = np.random.default_rng(42)
    
    # Try multiple initial configurations
    initial_configs = [
        _golden_spiral(0.5),
        _golden_spiral(0.0),
        _golden_spiral(0.25),
        _golden_spiral(0.75),
        _golden_spiral(0.1),
        _golden_spiral(0.9),
    ]
    
    # Also try random configurations
    for _ in range(4):
        random_pts = rng.normal(0, 1, (N, 3))
        random_pts = _project_to_sphere(random_pts)
        initial_configs.append(random_pts)
    
    best_pts = None
    best_energy = float('inf')
    
    # Optimize each configuration
    for idx, init_pts in enumerate(initial_configs):
        remaining = time_limit - (time.time() - start_time)
        if remaining < 10:
            break
        
        # First pass: optimize the initial configuration
        opt_pts, opt_energy = _riemannian_cg(init_pts, max_iter=500, tol=1e-10, time_limit=remaining, start_time=start_time)
        
        if opt_energy < best_energy:
            best_energy = opt_energy
            best_pts = opt_pts.copy()
        
        # Second pass: perturb and re-optimize
        remaining = time_limit - (time.time() - start_time)
        if remaining < 10:
            break
        
        perturbed_pts, perturbed_energy = _perturb_and_optimize(opt_pts, rng, time_limit, start_time)
        if perturbed_energy < best_energy:
            best_energy = perturbed_energy
            best_pts = perturbed_pts.copy()
        
        # Third pass: more aggressive perturbation
        remaining = time_limit - (time.time() - start_time)
        if remaining < 10:
            break
        
        perturbed_pts2 = best_pts + rng.normal(0, 0.1, best_pts.shape)
        perturbed_pts2 = _project_to_sphere(perturbed_pts2)
        opt_pts2, opt_energy2 = _riemannian_cg(perturbed_pts2, max_iter=300, tol=1e-10, time_limit=remaining, start_time=start_time)
        if opt_energy2 < best_energy:
            best_energy = opt_energy2
            best_pts = opt_pts2.copy()
    
    # Final high-precision polish
    remaining = time_limit - (time.time() - start_time)
    if remaining > 5 and best_pts is not None:
        final_pts, final_energy = _riemannian_cg(best_pts, max_iter=1000, tol=1e-12, time_limit=remaining, start_time=start_time)
        if final_energy < best_energy:
            best_energy = final_energy
            best_pts = final_pts.copy()
    
    # Ensure valid output
    if best_pts is None:
        best_pts = _golden_spiral(0.5)
    
    # Verify no points are too close
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_pts[i] - best_pts[j])
            if d < 1e-9:
                # Nudge apart
                best_pts[j] += rng.normal(0, 1e-6, 3)
                best_pts = _project_to_sphere(best_pts)
    
    # Final normalization
    best_pts = _project_to_sphere(best_pts)
    
    # Verify finite values
    if not np.all(np.isfinite(best_pts)):
        best_pts = _golden_spiral(0.5)
    
    return best_pts
# EVOLVE-BLOCK-END
