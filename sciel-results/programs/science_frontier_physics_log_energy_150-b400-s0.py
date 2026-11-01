"""Direct approach for minimal logarithmic energy of 150 points on sphere."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 150


def _log_energy(pts):
    """Compute logarithmic energy sum_{i<j} log(1/|x_i - x_j|) using math.fsum."""
    n = len(pts)
    terms = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-12:
                return float('inf')
            terms.append(math.log(1.0 / d))
    return math.fsum(terms)


def _energy_and_grad(pts):
    """Vectorized energy and gradient (projected to tangent space)."""
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, 1.0)
    inv_d2 = 1.0 / (dists ** 2)
    np.fill_diagonal(inv_d2, 0.0)
    
    weighted_sum_pts = inv_d2 @ pts
    sum_inv_d2 = np.sum(inv_d2, axis=1)
    grad = -pts * sum_inv_d2[:, np.newaxis] + weighted_sum_pts
    
    # Project gradient to tangent space
    dot_prod = np.sum(grad * pts, axis=1)
    grad -= dot_prod[:, np.newaxis] * pts
    
    # Compute energy (upper triangle only)
    iu = np.triu_indices(n, k=1)
    d = dists[iu]
    d = np.maximum(d, 1e-15)
    energy = np.sum(np.log(1.0 / d))
    
    return energy, grad


def _log_energy_fast(pts):
    """Vectorized log energy computation for speed."""
    n = len(pts)
    diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diffs, axis=2)
    iu = np.triu_indices(n, k=1)
    d = dists[iu]
    d = np.maximum(d, 1e-15)
    return np.sum(np.log(1.0 / d))


def _fibonacci_sphere(n):
    """Generate Fibonacci spiral points on unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts


def _normalize(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def _lbfgs_optimize(pts, time_limit=30.0):
    """Optimize using L-BFGS-B with projected gradient."""
    start = time.time()
    
    # Flatten points for L-BFGS-B
    x0 = pts.flatten()
    
    def fun_and_grad(x):
        p = x.reshape(N, 3)
        p = _normalize(p)
        energy, grad = _energy_and_grad(p)
        # Project gradient back to flat space
        return energy, grad.flatten()
    
    # Use L-BFGS-B with bounds to keep points reasonable
    # No explicit bounds needed since we normalize after each step
    res = minimize(fun_and_grad, x0, method='L-BFGS-B', jac=True,
                   options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-12,
                            'maxfun': 5000, 'maxls': 50})
    
    # Check time
    if time.time() - start > time_limit:
        return pts, _log_energy_fast(pts)
    
    # Extract optimized points
    x_opt = res.x.reshape(N, 3)
    x_opt = _normalize(x_opt)
    energy = _log_energy_fast(x_opt)
    
    return x_opt, energy


def _polish(pts, time_limit=5.0):
    """High-precision polish with very small steps."""
    start = time.time()
    current = pts.copy()
    energy = _log_energy_fast(current)
    step_size = 1e-4
    
    for it in range(5000):
        if time.time() - start > time_limit - 0.1:
            break
        
        e, grad = _energy_and_grad(current)
        grad_norm = np.linalg.norm(grad)
        
        if grad_norm < 1e-16:
            break
        
        # Backtracking line search with very tight tolerance
        alpha = step_size
        accepted = False
        for _ in range(30):
            new_pts = current - alpha * grad
            new_pts = _normalize(new_pts)
            new_energy = _log_energy_fast(new_pts)
            
            if new_energy < e - 1e-16:
                accepted = True
                break
            alpha *= 0.5
        
        if accepted:
            current = new_pts
            energy = new_energy
            step_size = min(alpha * 1.02, 0.01)
        else:
            step_size *= 0.5
            if step_size < 1e-18:
                break
    
    return current, energy


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Initial Fibonacci spiral
    pts_fib = _fibonacci_sphere(N)
    
    best_pts = pts_fib.copy()
    best_energy = _log_energy_fast(pts_fib)
    
    # Generate diverse initializations
    restarts = [
        ("fibonacci", pts_fib.copy()),
    ]
    
    # Random initializations
    for seed in range(4):
        rng_local = np.random.default_rng(seed * 123 + 7)
        pts_rand = rng_local.standard_normal((N, 3))
        pts_rand = _normalize(pts_rand)
        restarts.append((f"random_{seed}", pts_rand))
    
    # Perturbed Fibonacci
    for seed in range(3):
        rng_local = np.random.default_rng(seed * 456 + 13)
        pts_pert = pts_fib + rng_local.standard_normal((N, 3)) * 0.02
        pts_pert = _normalize(pts_pert)
        restarts.append((f"perturbed_{seed}", pts_pert))
    
    # Optimize each restart
    for name, init_pts in restarts:
        if time.time() > deadline - 40:
            break
        
        # First pass: L-BFGS-B optimization
        time_left = deadline - time.time() - 10
        if time_left < 5:
            break
        
        opt_pts, opt_energy = _lbfgs_optimize(init_pts, time_limit=min(time_left, 25.0))
        
        # Second pass: Polish
        if time.time() < deadline - 5:
            polish_time = min(deadline - time.time() - 2, 5.0)
            if polish_time > 0.5:
                polished_pts, polished_energy = _polish(opt_pts, time_limit=polish_time)
                if polished_energy < opt_energy:
                    opt_pts = polished_pts
                    opt_energy = polished_energy
        
        # Update best
        if opt_energy < best_energy:
            best_energy = opt_energy
            best_pts = opt_pts.copy()
    
    # Final high-precision polish on best solution
    if time.time() < deadline - 3:
        final_time = min(deadline - time.time() - 1, 8.0)
        if final_time > 0.5:
            final_pts, final_energy = _polish(best_pts, time_limit=final_time)
            if final_energy < best_energy:
                best_energy = final_energy
                best_pts = final_pts
    
    # Final normalization
    best_pts = _normalize(best_pts)
    
    return best_pts
# EVOLVE-BLOCK-END
