"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 100

# EVOLVE-BLOCK-START

def _energy_and_grad_tangent(params):
    """
    Compute logarithmic energy and gradients in tangent-space coordinates.
    params: (N, 3) array of tangent vectors (small perturbations from reference sphere)
    We use a local chart: each point is x_i = normalize(r_i + t_i) where r_i is a base point
    and t_i is a small tangent vector. For simplicity, we parameterize directly in 3D
    and project gradients to the sphere.
    
    Actually, let's use a simpler approach: work in 3D Cartesian and use the fact that
    on the sphere, the gradient of a function f(x) is grad_f - (x . grad_f) x.
    """
    n = len(params)
    pts = params  # (N, 3) on or near unit sphere
    
    # Project to unit sphere
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    pts = pts / norms
    
    # Compute pairwise dot products
    c = pts @ pts.T
    
    # Mask diagonal
    idx = np.triu_indices(n, k=1)
    c_ij = c[idx]
    
    # Energy E = -sum log(sqrt(2 - 2c)) = -0.5 sum log(2 - 2c)
    val = 2.0 - 2.0 * c_ij
    val = np.maximum(val, 1e-15)
    
    energy = -0.5 * np.sum(np.log(val))
    
    # Gradient in 3D: dE/dx_i = sum_{j != i} dE/dc_ij * dc_ij/dx_i
    # dE/dc_ij = -0.5 * (1/(2-2c_ij)) * (-2) = 1/(2-2c_ij) = 1/(2(1-c_ij))
    # dc_ij/dx_i = x_j
    
    # So grad_3d[i] = sum_{j != i} x_j / (2(1 - c_ij))
    
    denom = 2.0 * (1.0 - c)
    # Set diagonal to 1 to avoid division by zero
    denom[np.diag_indices(n)] = 1.0
    
    # weight[i,j] = 1/denom[i,j] for i != j
    with np.errstate(divide='ignore', invalid='ignore'):
        weight = np.where(~np.eye(n, dtype=bool), 1.0 / denom, 0.0)
    
    # grad_3d[i] = sum_j weight[i,j] * x_j = pts.T @ weight ... wait
    # grad_3d[i] = sum_j weight[i,j] * pts[j]
    # = (weight @ pts)[i]
    grad_3d = weight @ pts
    
    # Project to tangent space: grad_tan[i] = grad_3d[i] - (pts[i] . grad_3d[i]) * pts[i]
    dot_products = np.sum(pts * grad_3d, axis=1, keepdims=True)
    grad_tangent = grad_3d - dot_products * pts
    
    return energy, grad_tangent

# (removed - replaced by _fibonacci_sphere_3d and _random_sphere_3d)

def _optimize_single_cartesian(pts0, deadline, maxiter=500):
    """Run L-BFGS-B optimization in Cartesian coordinates with projection."""
    if time.time() > deadline:
        return None, float('inf')
        
    def obj_and_grad(x):
        if time.time() > deadline:
            raise TimeoutError()
        # x is (N, 3) flattened
        pts = x.reshape(-1, 3)
        # Project to unit sphere
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-15)
        pts = pts / norms
        
        # Compute energy and gradient
        n = len(pts)
        c = pts @ pts.T
        idx = np.triu_indices(n, k=1)
        c_ij = c[idx]
        val = 2.0 - 2.0 * c_ij
        val = np.maximum(val, 1e-15)
        energy = -0.5 * np.sum(np.log(val))
        
        denom = 2.0 * (1.0 - c)
        denom[np.diag_indices(n)] = 1.0
        with np.errstate(divide='ignore', invalid='ignore'):
            weight = np.where(~np.eye(n, dtype=bool), 1.0 / denom, 0.0)
        grad_3d = weight @ pts
        dot_products = np.sum(pts * grad_3d, axis=1, keepdims=True)
        grad_tangent = grad_3d - dot_products * pts
        return energy, grad_tangent.flatten()
    
    try:
        x0 = pts0.flatten()
        res = minimize(
            obj_and_grad,
            x0,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
        )
        if res.success or res.fun < float('inf'):
            pts_opt = res.x.reshape(-1, 3)
            # Final projection
            norms = np.linalg.norm(pts_opt, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-15)
            pts_opt = pts_opt / norms
            return pts_opt, res.fun
        return None, float('inf')
    except Exception:
        return None, float('inf')

def _fibonacci_sphere_3d(n):
    """Generate Fibonacci spiral points in 3D Cartesian coordinates."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    r = np.sqrt(np.maximum(1.0 - z**2, 0.0))
    golden_angle = np.pi * (1.0 + np.sqrt(5.0))
    theta = golden_angle * i
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.column_stack([x, y, z])

def _random_sphere_3d(n, rng):
    """Generate random points on the unit sphere."""
    pts = rng.standard_normal((n, 3))
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def construct():
    """Construct 100 points on the unit sphere minimizing logarithmic energy."""
    deadline = time.time() + 140.0  # 10s safety margin
    start_time = time.time()
    
    best_pts = None
    best_energy = float('inf')
    
    # Start with Fibonacci
    fib_pts = _fibonacci_sphere_3d(N)
    
    # Optimize from Fibonacci
    pts_opt, energy_opt = _optimize_single_cartesian(fib_pts, deadline, maxiter=300)
    if pts_opt is not None and energy_opt < best_energy:
        best_energy = energy_opt
        best_pts = pts_opt.copy()
    
    # Try multiple random restarts
    rng = np.random.default_rng(42)
    n_restarts = 25
    
    for r in range(n_restarts):
        if time.time() > deadline - 15.0:
            break
        
        elapsed = time.time() - start_time
        # Use less time per restart as we go
        remaining = deadline - time.time()
        if remaining < 20:
            break
        
        # Mix of pure random and perturbed Fibonacci
        if r % 3 == 0:
            pts0 = _random_sphere_3d(N, rng)
        elif r % 3 == 1:
            # Perturbed Fibonacci
            noise = rng.standard_normal((N, 3)) * 0.3
            pts0 = fib_pts + noise
            norms = np.linalg.norm(pts0, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-15)
            pts0 = pts0 / norms
        else:
            # Cluster-based: start from a subset of optimized points with noise
            if best_pts is not None:
                idx = rng.choice(N, size=N, replace=False)
                pts0 = best_pts[idx] + rng.standard_normal((N, 3)) * 0.2
                norms = np.linalg.norm(pts0, axis=1, keepdims=True)
                norms = np.maximum(norms, 1e-15)
                pts0 = pts0 / norms
            else:
                pts0 = _random_sphere_3d(N, rng)
        
        # Limit iterations based on remaining time
        time_per_restart = min(30.0, (deadline - time.time()) / max(1, n_restarts - r))
        maxiter = int(max(50, time_per_restart * 10))
        
        pts_opt, energy_opt = _optimize_single_cartesian(pts0, deadline, maxiter=maxiter)
        if pts_opt is not None and energy_opt < best_energy:
            best_energy = energy_opt
            best_pts = pts_opt.copy()
    
    # Final high-precision polish if time permits
    if best_pts is not None and time.time() < deadline - 10.0:
        remaining = deadline - time.time()
        if remaining > 15:
            maxiter_polish = int(min(1000, remaining * 5))
            pts_opt, energy_opt = _optimize_single_cartesian(best_pts, deadline, maxiter=maxiter_polish)
            if pts_opt is not None and energy_opt < best_energy:
                best_energy = energy_opt
                best_pts = pts_opt.copy()
    
    # Second polish pass with even tighter tolerance
    if best_pts is not None and time.time() < deadline - 5.0:
        remaining = deadline - time.time()
        if remaining > 10:
            maxiter_polish2 = int(min(500, remaining * 3))
            pts_opt, energy_opt = _optimize_single_cartesian(best_pts, deadline, maxiter=maxiter_polish2)
            if pts_opt is not None and energy_opt < best_energy:
                best_energy = energy_opt
                best_pts = pts_opt.copy()

    if best_pts is None:
        # Fallback: Fibonacci
        best_pts = _fibonacci_sphere_3d(N)

    # Final validation and correction
    norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    best_pts = best_pts / norms
    
    # Check for close points
    diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    iu = np.triu_indices(N, k=1)
    min_dist = np.min(dist[iu])
    if min_dist < 1e-9:
        # Perturb the closest pair
        for _ in range(200):
            min_idx = np.argmin(dist[iu])
            i, j = iu[0][min_idx], iu[1][min_idx]
            # Move point j in a random direction
            rand_dir = rng.standard_normal(3)
            rand_dir /= np.linalg.norm(rand_dir)
            # Ensure we move away from point i
            if np.dot(rand_dir, best_pts[i] - best_pts[j]) > 0:
                rand_dir = -rand_dir
            best_pts[j] = best_pts[j] + 1e-6 * rand_dir
            best_pts[j] /= np.linalg.norm(best_pts[j])
            diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
            dist = np.linalg.norm(diff, axis=2)
            if np.min(dist[iu]) >= 1e-9:
                break

    # Final normalization
    norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    best_pts = best_pts / norms
    
    return best_pts

# EVOLVE-BLOCK-END
