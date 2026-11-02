"""Stronger algorithm for 100-point logarithmic energy on the unit sphere."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 100
TIME_LIMIT = 148.0  # safety margin before 150s kill


def _energy_grad_fast(points):
    """Compute energy and gradient efficiently using dist_sq directly."""
    n = len(points)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    inv_dist_sq = 1.0 / dist_sq
    np.fill_diagonal(inv_dist_sq, 0.0)
    grad = -np.sum(inv_dist_sq[:, :, np.newaxis] * diff, axis=1)
    
    # Use only upper triangle for energy (each pair counted once)
    iu = np.triu_indices(n, k=1)
    d = np.sqrt(dist_sq[iu])
    d = np.maximum(d, 1e-15)
    energy = math.fsum(np.log(1.0 / d).tolist())
    return energy, grad


def _riemannian_cg(points, time_budget, maxiter=5000, ftol=1e-12, gtol=1e-10):
    """
    Riemannian conjugate gradient method on the sphere.
    This respects the constraint directly without projection overhead.
    """
    start_time = time.time()
    n = len(points)
    x = points.copy()
    
    e, g = _energy_grad_fast(x)
    g_tangent = g - (g * x).sum(axis=1, keepdims=True) * x
    
    # Check if already at optimum
    g_norm = np.linalg.norm(g_tangent)
    if g_norm < gtol:
        return x
    
    # Conjugate gradient state
    p = -g_tangent.copy()
    beta = 0.0
    
    for it in range(maxiter):
        elapsed = time.time() - start_time
        if elapsed > time_budget:
            break
        
        # Line search (backtracking)
        alpha = 0.01
        e_old = e
        g_old = g_tangent.copy()
        
        for ls_iter in range(30):
            # Move in tangent direction and project back to sphere
            x_new = x + alpha * p
            x_new = x_new / np.linalg.norm(x_new, axis=1, keepdims=True)
            
            e_new, g_new = _energy_grad_fast(x_new)
            g_new_tangent = g_new - (g_new * x_new).sum(axis=1, keepdims=True) * x_new
            
            # Armijo condition
            grad_dot = np.sum(g_old * p)
            if e_new <= e_old + 1e-4 * alpha * grad_dot:
                break
            alpha *= 0.5
        
        # Check convergence
        e_new, g_new = _energy_grad_fast(x_new)
        g_new_tangent = g_new - (g_new * x_new).sum(axis=1, keepdims=True) * x_new
        g_norm_new = np.linalg.norm(g_new_tangent)
        
        if g_norm_new < gtol:
            x = x_new
            break
        
        if abs(e_new - e) < ftol:
            x = x_new
            break
        
        # Update conjugate direction
        g_diff = g_new_tangent - g_old
        beta = np.sum(g_diff * g_diff) / max(np.sum(g_old * g_old), 1e-15)
        beta = min(beta, 1.0)  # Cap beta for stability
        
        p = -g_new_tangent + beta * p
        
        x = x_new
        e = e_new
        g_tangent = g_new_tangent
    
    return x


def _energy(points):
    """Compute logarithmic energy sum_{i<j} log(1/|x_i - x_j|)."""
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    d = np.sqrt(dist_sq[mask])
    d = np.maximum(d, 1e-15)
    return math.fsum(np.log(1.0 / d).tolist())


def _project_to_sphere(points):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return points / norms


def _sphere_fun_grad(x):
    """Objective and gradient in ambient R^(3N) space with projection."""
    p = _project_to_sphere(x.reshape(-1, 3))
    e, g = _energy_grad_fast(p)
    g_tangent = g - (g * p).sum(axis=1, keepdims=True) * p
    return e, g_tangent.flatten()


def _optimize_single(points, time_budget, maxiter=5000, ftol=1e-14, gtol=1e-12):
    """Run L-BFGS-B optimization from a single starting point."""
    start_time = time.time()
    x0 = points.flatten().copy()
    
    try:
        res = minimize(_sphere_fun_grad, x0, method='L-BFGS-B', jac=True,
                       options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol, 'maxls': 50})
        
        if hasattr(res, 'x'):
            new_points = _project_to_sphere(res.x.reshape(-1, 3))
            return new_points
    except Exception:
        pass
    
    return points


def _optimize_hybrid(points, time_budget, maxiter=5000):
    """
    Hybrid optimization: L-BFGS-B followed by Riemannian CG for better convergence.
    """
    start_time = time.time()
    
    # Phase 1: L-BFGS-B to get close to optimum
    elapsed = time.time() - start_time
    if elapsed < time_budget * 0.6:
        points = _optimize_single(points, time_budget * 0.6, maxiter=maxiter, ftol=1e-13, gtol=1e-11)
    
    # Phase 2: Riemannian CG for fine-tuning
    elapsed = time.time() - start_time
    remaining = time_budget - elapsed
    if remaining > 1.0:
        points = _riemannian_cg(points, remaining, maxiter=3000, ftol=1e-14, gtol=1e-12)
    
    return points


def _fibonacci_spiral(n):
    """Generate initial points using Fibonacci spiral on sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _project_to_sphere(points)


def _random_points(n, seed=None):
    """Generate random points on sphere."""
    if seed is not None:
        np.random.seed(seed)
    points = np.random.randn(n, 3)
    return _project_to_sphere(points)


def _bands_initial(n, num_bands=5, seed=None):
    """Generate points in latitudinal bands for diverse initial configurations."""
    if seed is not None:
        np.random.seed(seed)
    points = []
    for b in range(num_bands):
        z_center = -1.0 + 2.0 * (b + 0.5) / num_bands
        z_spread = 0.8 / num_bands
        n_band = n // num_bands + (1 if b < n % num_bands else 0)
        for _ in range(n_band):
            z = z_center + np.random.uniform(-z_spread, z_spread)
            z = np.clip(z, -1.0, 1.0)
            r = np.sqrt(1.0 - z * z)
            phi = np.random.uniform(0, 2 * np.pi)
            points.append([r * np.cos(phi), r * np.sin(phi), z])
    points = np.array(points)
    return _project_to_sphere(points)


def _rings_initial(n, num_rings=3, seed=None):
    """Generate points on concentric rings for diverse initial configurations."""
    if seed is not None:
        np.random.seed(seed)
    points = []
    for r_idx in range(num_rings):
        z = -1.0 + 2.0 * (r_idx + 0.5) / num_rings
        r = np.sqrt(1.0 - z * z)
        n_ring = n // num_rings + (1 if r_idx < n % num_rings else 0)
        for i in range(n_ring):
            phi = 2.0 * np.pi * i / n_ring
            points.append([r * np.cos(phi), r * np.sin(phi), z])
    points = np.array(points)
    return _project_to_sphere(points)


def _perturb_and_reoptimize(points, time_budget, seed=None):
    """
    Apply small random perturbations to points and re-optimize.
    More efficient than full SA since each perturbation gets a proper local optimization.
    """
    start_time = time.time()
    if seed is not None:
        np.random.seed(seed)
    
    best_points = points.copy()
    best_energy = _energy(best_points)
    
    # Try several perturbation scales
    perturbation_scales = [0.01, 0.02, 0.05, 0.1, 0.03, 0.08, 0.15]
    
    for scale in perturbation_scales:
        elapsed = time.time() - start_time
        if elapsed > time_budget:
            break
        
        remaining = time_budget - elapsed
        if remaining < 1.0:
            break
        
        # Perturb a subset of points
        n_perturb = np.random.choice([3, 5, 8, 10, 15])
        indices = np.random.choice(N, size=n_perturb, replace=False)
        
        new_points = best_points.copy()
        for idx in indices:
            # Small random rotation
            angle = np.random.uniform(0, 2 * np.pi) * scale
            axis = np.random.randn(3)
            axis /= np.linalg.norm(axis)
            
            c = math.cos(angle)
            s = math.sin(angle)
            k = axis
            k_outer_k = np.outer(k, k)
            K = np.array([[0, -k[2], k[1]],
                          [k[2], 0, -k[0]],
                          [-k[1], k[0], 0]])
            R = c * np.eye(3) + s * K + (1 - c) * k_outer_k
            
            new_points[idx] = R @ new_points[idx]
        
        new_points = _project_to_sphere(new_points)
        
        # Re-optimize from perturbed position using hybrid method
        opt_budget = min(remaining * 0.35, 6.0)
        if opt_budget < 0.5:
            continue
            
        optimized = _optimize_hybrid(new_points, opt_budget, maxiter=2500)
        e = _energy(optimized)
        
        if e < best_energy:
            best_energy = e
            best_points = optimized
    
    return best_points


def _high_precision_polish(points, time_budget):
    """
    Final high-precision polish: use very tight tolerances and more iterations.
    Uses a staged approach with progressively tighter convergence criteria.
    """
    start_time = time.time()
    x0 = points.flatten().copy()
    
    # Stage 1: L-BFGS-B to get close
    elapsed = time.time() - start_time
    remaining = time_budget - elapsed
    if remaining > 1.0:
        try:
            res = minimize(_sphere_fun_grad, x0, method='L-BFGS-B', jac=True,
                           options={'maxiter': 8000, 'ftol': 1e-13, 'gtol': 1e-11, 'maxls': 60})
            if hasattr(res, 'x'):
                x0 = res.x
        except Exception:
            pass
    
    # Stage 2: Riemannian CG for fine-tuning
    elapsed = time.time() - start_time
    remaining = time_budget - elapsed
    if remaining > 2.0:
        points_mid = _project_to_sphere(x0.reshape(-1, 3))
        points_mid = _riemannian_cg(points_mid, remaining * 0.5, maxiter=5000, ftol=1e-14, gtol=1e-12)
        x0 = points_mid.flatten()
    
    # Stage 3: Final L-BFGS-B with tightest tolerances
    elapsed = time.time() - start_time
    remaining = time_budget - elapsed
    if remaining > 1.0:
        try:
            res = minimize(_sphere_fun_grad, x0, method='L-BFGS-B', jac=True,
                           options={'maxiter': 10000, 'ftol': 1e-15, 'gtol': 1e-13, 'maxls': 100})
            if hasattr(res, 'x'):
                x0 = res.x
        except Exception:
            pass
    
    # Stage 4: Final Riemannian CG polish
    elapsed = time.time() - start_time
    remaining = time_budget - elapsed
    if remaining > 1.5:
        points_final = _project_to_sphere(x0.reshape(-1, 3))
        points_final = _riemannian_cg(points_final, remaining, maxiter=3000, ftol=1e-15, gtol=1e-14)
        x0 = points_final.flatten()
    
    new_points = _project_to_sphere(x0.reshape(-1, 3))
    return new_points


def _kick_and_optimize(points, kick_size, time_budget):
    """
    Kick the best points by a random rotation, then re-optimize.
    This helps escape shallow local minima.
    """
    start_time = time.time()
    
    # Apply a random rotation
    axis = np.random.randn(3)
    axis /= np.linalg.norm(axis)
    angle = np.random.uniform(0, 2 * np.pi) * kick_size
    
    c = math.cos(angle)
    s = math.sin(angle)
    k = axis
    k_outer_k = np.outer(k, k)
    K = np.array([[0, -k[2], k[1]],
                  [k[2], 0, -k[0]],
                  [-k[1], k[0], 0]])
    R = c * np.eye(3) + s * K + (1 - c) * k_outer_k
    
    kicked = (R @ points.T).T
    kicked = _project_to_sphere(kicked)
    
    # Optimize from the kicked position using hybrid method
    remaining = time_budget - (time.time() - start_time)
    if remaining < 0.5:
        return kicked
    
    optimized = _optimize_hybrid(kicked, remaining, maxiter=2500)
    return optimized


def construct():
    start_time = time.time()
    
    # Start with Fibonacci spiral
    points = _fibonacci_spiral(N)
    best_points = points.copy()
    best_energy = _energy(best_points)
    
    # Phase 1: L-BFGS-B from Fibonacci spiral
    elapsed = time.time() - start_time
    if elapsed < TIME_LIMIT * 0.15:
        optimized = _optimize_single(points, TIME_LIMIT * 0.12, maxiter=5000, ftol=1e-14, gtol=1e-12)
        e = _energy(optimized)
        if e < best_energy:
            best_energy = e
            best_points = optimized
    
    # Phase 2: Multiple diverse random restarts with hybrid optimization
    np.random.seed(123)
    restart_count = 0
    while True:
        elapsed = time.time() - start_time
        if elapsed > TIME_LIMIT * 0.55:
            break
        restart_count += 1
        
        # Generate diverse random starting points
        # Prioritize perturbations of the current best (more likely to find improvements)
        if restart_count % 8 in [1, 2, 3]:
            # Perturb the current best with varying scales
            scale = 0.1 + 0.1 * (restart_count % 3)
            perturb = np.random.randn(N, 3) * scale
            random_points = _project_to_sphere(best_points + perturb)
        elif restart_count % 8 == 4:
            # Rotated Fibonacci
            base = _fibonacci_spiral(N)
            axis = np.random.randn(3)
            axis /= np.linalg.norm(axis)
            angle = np.random.uniform(0, 2 * np.pi)
            c = math.cos(angle)
            s = math.sin(angle)
            k = axis
            k_outer_k = np.outer(k, k)
            K = np.array([[0, -k[2], k[1]],
                          [k[2], 0, -k[0]],
                          [-k[1], k[0], 0]])
            R = c * np.eye(3) + s * K + (1 - c) * k_outer_k
            random_points = (R @ base.T).T
        elif restart_count % 8 == 5:
            random_points = _random_points(N)
        elif restart_count % 8 == 6:
            random_points = _bands_initial(N, num_bands=5, seed=100 + restart_count)
        elif restart_count % 8 == 7:
            random_points = _rings_initial(N, num_rings=3, seed=200 + restart_count)
        else:
            # Larger perturbation of best
            perturb = np.random.randn(N, 3) * 0.4
            random_points = _project_to_sphere(best_points + perturb)
        
        budget_per_restart = min(TIME_LIMIT * 0.022, (TIME_LIMIT * 0.55 - elapsed) * 0.85)
        if budget_per_restart < 0.8:
            break
        
        optimized = _optimize_hybrid(random_points, budget_per_restart, maxiter=3000)
        
        e = _energy(optimized)
        if e < best_energy:
            best_energy = e
            best_points = optimized
    
    # Phase 3: Kick-and-optimize from best points (multiple kicks)
    elapsed = time.time() - start_time
    if elapsed < TIME_LIMIT * 0.80:
        remaining = TIME_LIMIT * 0.80 - elapsed
        if remaining > 2.0:
            np.random.seed(456)
            kick_sizes = [0.03, 0.05, 0.08, 0.12, 0.06, 0.15, 0.20, 0.10]
            for kick_size in kick_sizes:
                elapsed = time.time() - start_time
                if elapsed > TIME_LIMIT * 0.80:
                    break
                
                kick_budget = min(remaining * 0.12, (TIME_LIMIT * 0.80 - elapsed) * 0.9)
                if kick_budget < 0.5:
                    break
                
                kicked = _kick_and_optimize(best_points, kick_size, kick_budget)
                e = _energy(kicked)
                if e < best_energy:
                    best_energy = e
                    best_points = kicked
    
    # Phase 4: Perturb-and-reoptimize refinement (replaces expensive SA)
    elapsed = time.time() - start_time
    if elapsed < TIME_LIMIT * 0.92:
        remaining = TIME_LIMIT * 0.92 - elapsed
        if remaining > 2.0:
            np.random.seed(789)
            sa_points = _perturb_and_reoptimize(best_points, remaining * 0.85, seed=789)
            e = _energy(sa_points)
            if e < best_energy:
                best_energy = e
                best_points = sa_points
    
    # Phase 5: High-precision polish
    elapsed = time.time() - start_time
    if elapsed < TIME_LIMIT * 0.97:
        remaining = TIME_LIMIT * 0.97 - elapsed
        if remaining > 2.0:
            polished = _high_precision_polish(best_points, remaining * 0.9)
            e = _energy(polished)
            if e < best_energy:
                best_energy = e
                best_points = polished
            
            # Additional micro-polish if time remains
            elapsed = time.time() - start_time
            remaining = TIME_LIMIT * 0.995 - elapsed
            if remaining > 2.0:
                micro_polished = _high_precision_polish(best_points, remaining * 0.85)
                e2 = _energy(micro_polished)
                if e2 < best_energy:
                    best_energy = e2
                    best_points = micro_polished
    
    # Final projection and validation
    best_points = _project_to_sphere(best_points)
    
    # Check for points too close together
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_points[i] - best_points[j])
            if d < 1e-9:
                perturb = np.random.randn(3) * 1e-5
                best_points[i] = best_points[i] + perturb
                best_points[i] = best_points[i] / np.linalg.norm(best_points[i])
    
    # Final energy check
    final_energy = _energy(best_points)
    if not np.isfinite(final_energy):
        best_points = _fibonacci_spiral(N)
    
    # One last ultra-tight polish using Riemannian CG if time allows
    elapsed = time.time() - start_time
    remaining = TIME_LIMIT - elapsed
    if remaining > 1.0:
        try:
            final_points = _riemannian_cg(best_points, remaining * 0.8, maxiter=5000, ftol=1e-15, gtol=1e-14)
            e_final = _energy(final_points)
            if e_final < best_energy and np.isfinite(e_final):
                best_points = final_points
                best_energy = e_final
        except Exception:
            pass
    
    # Final validation
    best_points = _project_to_sphere(best_points)
    final_energy = _energy(best_points)
    if not np.isfinite(final_energy):
        best_points = _fibonacci_spiral(N)
    
    return best_points
