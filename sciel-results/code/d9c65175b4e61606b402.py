"""Optimize logarithmic energy of 150 points on the unit sphere."""
import numpy as np
import math
import time
from scipy.optimize import minimize

N = 150
DEADLINE = 148.0  # seconds, with 2s safety margin before 150s limit


def _energy_and_grad(pts):
    """Compute logarithmic energy and gradient for points on unit sphere.
    
    Energy = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log|d_ij|
    where d_ij is chordal distance.
    
    Gradient: dE/dx_i = -sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    """
    # Compute pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (N, N, 3)
    dist_sq = np.einsum('ijk,ijk->ij', diff, diff)  # (N, N)
    np.fill_diagonal(dist_sq, 1.0)  # avoid division by zero
    dist = np.sqrt(dist_sq)
    
    # Energy: sum over i<j of -log(d_ij)
    iu = np.triu_indices(N, k=1)
    # Using np.sum for speed during optimization
    energy = -np.sum(np.log(dist[iu]))
    
    # grad[i] = -sum_{j!=i} (x_i - x_j) / dist_sq[i,j]
    inv_dist_sq = 1.0 / dist_sq  # (N, N)
    np.fill_diagonal(inv_dist_sq, 0.0)
    
    row_sums = np.sum(inv_dist_sq, axis=1)  # (N,)
    weighted_sum = inv_dist_sq @ pts  # (N, 3)
    
    grad = -pts * row_sums[:, np.newaxis] + weighted_sum  # (N, 3)
    
    return energy, grad


def _energy_fsum(pts):
    """Compute energy using math.fsum for high accuracy."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.einsum('ijk,ijk->ij', diff, diff)
    iu = np.triu_indices(N, k=1)
    dists = np.sqrt(dist_sq[iu])
    # math.fsum is slow, so we only use it for final verification if needed
    # But for the optimizer, np.sum is sufficient.
    # Let's stick to np.sum for the optimizer to save time.
    return -np.sum(np.log(dists))


def _project_to_sphere(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def _fibonacci_sphere(n):
    """Generate Fibonacci spiral points on unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(n, rng):
    """Generate random points on unit sphere."""
    pts = rng.standard_normal((n, 3))
    return _project_to_sphere(pts)


def _optimize_single(pts0, max_time, rng=None):
    """Optimize a single starting configuration using L-BFGS-B on the sphere."""
    start_time = time.time()
    x0 = pts0.flatten()
    
    def objective(x):
        pts = _project_to_sphere(x.reshape(N, 3))
        e, _ = _energy_and_grad(pts)
        return e
    
    def gradient(x):
        pts_raw = x.reshape(N, 3)
        norms = np.linalg.norm(pts_raw, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-15)
        pts = pts_raw / norms
        
        e, g = _energy_and_grad(pts)
        
        # Chain rule: dE/dx_raw = J_proj^T @ g
        g_dot_pts = np.sum(g * pts, axis=1)
        g_tan = g - g_dot_pts[:, np.newaxis] * pts
        grad_raw = g_tan / norms
        
        return grad_raw.flatten()
    
    best_x = x0.copy()
    best_e = objective(x0)
    
    # Use L-BFGS-B
    # We run multiple iterations if time permits
    while time.time() - start_time < max_time:
        result = minimize(
            objective, x0, jac=gradient,
            method='L-BFGS-B',
            options={'maxiter': 3000, 'ftol': 1e-16, 'gtol': 1e-12, 'maxls': 50}
        )
        
        x_new = result.x
        e_new = objective(x_new)
        
        if e_new < best_e:
            best_e = e_new
            best_x = x_new.copy()
        
        if result.success and result.status in [0, 1]:
            # Try one more polish with tighter tolerance
            if time.time() - start_time < max_time - 2:
                result2 = minimize(
                    objective, x_new, jac=gradient,
                    method='L-BFGS-B',
                    options={'maxiter': 5000, 'ftol': 1e-18, 'gtol': 1e-14, 'maxls': 50}
                )
                e2 = objective(result2.x)
                if e2 < best_e:
                    best_e = e2
                    best_x = result2.x.copy()
            break
        
        # If not converged, perturb and restart
        if time.time() - start_time < max_time * 0.8:
            if rng is None:
                rng = np.random.default_rng(42)
            noise = 0.05 * rng.standard_normal(len(x0))
            x0 = best_x + noise
        else:
            break
    
    return _project_to_sphere(best_x.reshape(N, 3)), best_e


def construct():
    start_time = time.time()
    rng = np.random.default_rng(42)
    
    # Generate diverse starting points
    starts = []
    
    # 1. Fibonacci spiral
    starts.append(_fibonacci_sphere(N))
    
    # 2. Perturbed Fibonacci
    for noise_scale in [0.03, 0.1, 0.2]:
        fib = _fibonacci_sphere(N)
        noise = noise_scale * rng.standard_normal((N, 3))
        perturbed = fib + noise
        starts.append(_project_to_sphere(perturbed))
        
    # 3. Random points
    for _ in range(5):
        starts.append(_random_sphere(N, rng))
        
    # 4. Icosahedron-based (approximate)
    # Use a simple grid or other structure if available, otherwise random
    # Let's add a few more random starts with different seeds
    for _ in range(5):
        starts.append(_random_sphere(N, rng))
    
    best_pts = None
    best_energy = float('inf')
    
    # Phase 1: Optimize initial starts
    # Allocate 50% of time to initial optimization
    phase1_deadline = start_time + (DEADLINE - start_time) * 0.5
    time_per_opt = (phase1_deadline - start_time) / len(starts)
    time_per_opt = max(2.0, min(10.0, time_per_opt))
    
    for i, s in enumerate(starts):
        if time.time() > phase1_deadline - 2:
            break
        
        remaining = phase1_deadline - time.time()
        if remaining < 2:
            break
            
        opt_time = min(time_per_opt, remaining - 1)
        if opt_time < 1:
            break
            
        try:
            pts_opt, e_opt = _optimize_single(s, opt_time, rng)
            if e_opt < best_energy:
                best_energy = e_opt
                best_pts = pts_opt.copy()
        except Exception:
            continue
            
    # Phase 2: Multi-start restarts from the best point
    # Use the remaining time to perturb and re-optimize
    if best_pts is not None:
        phase2_deadline = DEADLINE - 5  # Leave 5s for final polish
        current_best = best_pts.copy()
        current_best_e = best_energy
        
        # Loop until deadline
        while time.time() < phase2_deadline:
            # Generate perturbations
            perturbations = []
            for scale in [0.05, 0.1, 0.2, 0.5]:
                noise = scale * rng.standard_normal((N, 3))
                perturbed = current_best + noise
                perturbations.append(_project_to_sphere(perturbed))
            
            # Also add a new random start
            perturbations.append(_random_sphere(N, rng))
            
            # Optimize each perturbation with a small time budget
            # Time budget per perturbation: 5 seconds
            for p in perturbations:
                if time.time() > phase2_deadline - 1:
                    break
                try:
                    pts_opt, e_opt = _optimize_single(p, 5.0, rng)
                    if e_opt < current_best_e:
                        current_best_e = e_opt
                        current_best = pts_opt.copy()
                except Exception:
                    continue
            
            # If no improvement in a full cycle, we might be stuck
            # But we keep trying with different scales
            
    # Phase 3: Final high-precision polish
    if best_pts is not None and time.time() < DEADLINE - 2:
        remaining = DEADLINE - time.time() - 1
        if remaining > 2:
            try:
                pts_final, e_final = _optimize_single(current_best, remaining, rng)
                if e_final < best_energy:
                    best_energy = e_final
                    best_pts = pts_final.copy()
            except Exception:
                pass
    
    if best_pts is None:
        best_pts = _fibonacci_sphere(N)
    
    # Ensure valid output
    best_pts = _project_to_sphere(best_pts)
    assert best_pts.shape == (N, 3)
    assert np.all(np.isfinite(best_pts))
    assert np.all(np.linalg.norm(best_pts, axis=1) > 1e-15)
    
    # Check no two points are too close
    diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
    dists = np.sqrt(np.einsum('ijk,ijk->ij', diff, diff))
    iu = np.triu_indices(N, k=1)
    min_dists = dists[iu]
    for idx in np.where(min_dists < 1e-9)[0]:
        i, j = iu[0][idx], iu[1][idx]
        best_pts[j] += 1e-8 * rng.standard_normal(3)
        best_pts[j] = best_pts[j] / np.linalg.norm(best_pts[j])
        
    return best_pts
