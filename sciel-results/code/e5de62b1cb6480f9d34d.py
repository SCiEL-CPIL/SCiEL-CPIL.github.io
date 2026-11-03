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
    energy = -np.sum(np.log(dist[iu]))
    
    # grad[i] = -sum_{j!=i} (x_i - x_j) / dist_sq[i,j]
    inv_dist_sq = 1.0 / dist_sq  # (N, N)
    np.fill_diagonal(inv_dist_sq, 0.0)
    
    row_sums = np.sum(inv_dist_sq, axis=1)  # (N,)
    weighted_sum = inv_dist_sq @ pts  # (N, 3)
    
    grad = -pts * row_sums[:, np.newaxis] + weighted_sum  # (N, 3)
    
    return energy, grad


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


def _optimize_single(pts0, max_time):
    """Optimize a single starting configuration using L-BFGS-B on the sphere.
    
    Uses the chain rule to compute the correct gradient in the flat R^{3N} space
    when the objective projects x -> sphere(x) before evaluating energy.
    """
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
        # J_proj for point i: d(pts_i)/d(x_raw_i) = (I - pts_i pts_i^T) / norm_i
        # So: grad_raw[i] = (I - pts_i pts_i^T) @ g_i / norm_i
        # = (g_i - (g_i . pts_i) pts_i) / norm_i
        g_dot_pts = np.sum(g * pts, axis=1)  # (N,)
        g_tan = g - g_dot_pts[:, np.newaxis] * pts  # (N, 3)
        grad_raw = g_tan / norms  # (N, 3)
        
        return grad_raw.flatten()
    
    best_x = x0.copy()
    best_e = objective(x0)
    
    while time.time() - start_time < max_time:
        result = minimize(
            objective, x0, jac=gradient,
            method='L-BFGS-B',
            options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-12, 'maxls': 50}
        )
        
        x_new = result.x
        e_new = objective(x_new)
        
        if e_new < best_e:
            best_e = e_new
            best_x = x_new.copy()
        
        # Check if we've converged
        if result.success and result.status in [0, 1]:
            # Try one more polish from the converged point
            if time.time() - start_time < max_time - 2:
                result2 = minimize(
                    objective, x_new, jac=gradient,
                    method='L-BFGS-B',
                    options={'maxiter': 5000, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 50}
                )
                e2 = objective(result2.x)
                if e2 < best_e:
                    best_e = e2
                    best_x = result2.x.copy()
            break
        
        # Perturb to escape local minimum
        if time.time() - start_time < max_time * 0.7:
            noise = 0.05 * np.random.randn(len(x0))
            x0 = best_x + noise
        else:
            break
    
    return _project_to_sphere(best_x.reshape(N, 3)), best_e


def _simulated_annealing(pts, max_time, rng):
    """Simulated annealing with multi-point moves to escape local minima."""
    start_time = time.time()
    current = pts.copy()
    current_e, _ = _energy_and_grad(current)
    best = current.copy()
    best_e = current_e
    
    T = 0.3
    T_min = 1e-7
    cooling = 0.995
    
    while time.time() - start_time < max_time and T > T_min:
        # Pick 1-3 random points to perturb simultaneously
        n_move = rng.integers(1, 4)
        indices = rng.choice(N, size=n_move, replace=False)
        
        new_pts = current.copy()
        for idx in indices:
            rand_dir = rng.standard_normal(3)
            rand_dir -= np.dot(rand_dir, current[idx]) * current[idx]
            norm = np.linalg.norm(rand_dir)
            if norm > 1e-15:
                rand_dir /= norm
            step_size = T * 0.3 * np.sqrt(n_move)
            new_pt = current[idx].copy()
            new_pt += step_size * rand_dir
            new_pt = new_pt / np.linalg.norm(new_pt)
            new_pts[idx] = new_pt
        
        new_e, _ = _energy_and_grad(new_pts)
        delta_e = new_e - current_e
        
        if delta_e < 0 or rng.random() < math.exp(-delta_e / T):
            current = new_pts
            current_e = new_e
            if current_e < best_e:
                best_e = current_e
                best = current.copy()
        
        T *= cooling
    
    return best, best_e


def construct():
    start_time = time.time()
    rng = np.random.default_rng(42)
    
    # Generate diverse starting points
    starts = []
    
    # 1. Fibonacci spiral
    starts.append(_fibonacci_sphere(N))
    
    # 2. Perturbed Fibonacci (different noise levels)
    for noise_scale in [0.03, 0.1, 0.2]:
        fib = _fibonacci_sphere(N)
        noise = noise_scale * rng.standard_normal((N, 3))
        perturbed = fib + noise
        starts.append(_project_to_sphere(perturbed))
    
    # 3. Random points
    for _ in range(4):
        starts.append(_random_sphere(N, rng))
    
    # 4. Clustered starts (to explore different basins)
    for _ in range(2):
        centers = _random_sphere(10, rng)
        pts = []
        for c in centers:
            local = c + 0.1 * rng.standard_normal((15, 3))
            pts.append(_project_to_sphere(local))
        starts.append(np.vstack(pts))
    
    # Optimize each start
    best_pts = None
    best_energy = float('inf')
    
    # Phase 1: Optimize all starts with equal time allocation
    n_starts = len(starts)
    phase1_deadline = start_time + (DEADLINE - start_time) * 0.65
    time_per_opt = (phase1_deadline - start_time) / n_starts
    time_per_opt = max(4.0, min(15.0, time_per_opt))
    
    for i, s in enumerate(starts):
        if time.time() > phase1_deadline - 5:
            break
        
        remaining = phase1_deadline - time.time()
        if remaining < 4:
            break
        
        opt_time = min(time_per_opt, remaining - 2)
        if opt_time < 3:
            break
        
        try:
            pts_opt, e_opt = _optimize_single(s, opt_time)
            
            if e_opt < best_energy:
                best_energy = e_opt
                best_pts = pts_opt.copy()
        except Exception:
            continue
    
    # Phase 2: SA + polish on the best result
    if best_pts is not None and time.time() < DEADLINE - 15:
        sa_time = min(20.0, (DEADLINE - time.time()) - 20)
        if sa_time > 5:
            try:
                pts_sa, e_sa = _simulated_annealing(best_pts, sa_time, rng)
                if e_sa < best_energy:
                    best_energy = e_sa
                    best_pts = pts_sa.copy()
            except Exception:
                pass
    
    # Phase 3: Final high-precision polish
    if best_pts is not None and time.time() < DEADLINE - 5:
        remaining = DEADLINE - time.time() - 3
        if remaining > 5:
            try:
                pts_final, e_final = _optimize_single(best_pts, remaining)
                if e_final < best_energy:
                    best_energy = e_final
                    best_pts = pts_final.copy()
            except Exception:
                pass
    
    if best_pts is None:
        # Fallback to Fibonacci
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
