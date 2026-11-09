"""Search-based solver: multi-restart with diverse initializations and high-precision polish."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize


def _log_energy(pts):
    """Compute logarithmic energy: sum_{i<j} log(1/|x_i - x_j|)."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    iu = np.triu_indices(N, k=1)
    dist = np.sqrt(dist_sq[iu])
    return -math.fsum(np.log(dist).tolist())


def _energy_and_gradient(pts):
    """Compute energy and gradient of log energy w.r.t. points.
    
    E = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    
    dE/dx_i = sum_{j != i} (x_j - x_i) / |x_i - x_j|^2
    """
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)
    
    iu = np.triu_indices(N, k=1)
    energy = -math.fsum(np.log(dist[iu]).tolist())
    
    inv_dist_sq = 1.0 / dist_sq
    np.fill_diagonal(inv_dist_sq, 0.0)
    grad = -np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
    return energy, grad


def _project_to_sphere(pts):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms


def _fibonacci_points(n):
    """Generate Fibonacci spiral points on unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_points(n, rng):
    """Generate random points on unit sphere."""
    pts = rng.standard_normal((n, 3))
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms


def _lgbfgs_optimize(pts, deadline, maxiter=500):
    """Optimize using L-BFGS-B with spherical constraint via projection."""
    pts = pts.copy()
    pts = _project_to_sphere(pts)
    
    def objective_and_grad(flat_pts):
        p = flat_pts.reshape(N, 3)
        p = _project_to_sphere(p)
        e, g = _energy_and_gradient(p)
        # Project gradient to tangent space
        g_tangent = g - np.sum(g * p, axis=1, keepdims=True) * p
        return e, g_tangent.flatten()
    
    x0 = pts.flatten()
    try:
        result = minimize(
            objective_and_grad,
            x0,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50}
        )
        if time.time() > deadline:
            return pts, _log_energy(pts)
        opt_pts = _project_to_sphere(result.x.reshape(N, 3))
        opt_energy = _log_energy(opt_pts)
        return opt_pts, opt_energy
    except Exception:
        return pts, _log_energy(pts)


def _gradient_descent(pts, max_iter=5000, lr_init=0.01, deadline=None):
    """Gradient descent with backtracking line search and sphere projection."""
    pts = pts.copy()
    pts = _project_to_sphere(pts)
    current_energy = _log_energy(pts)
    
    best_pts = pts.copy()
    best_energy = current_energy
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        e, grad = _energy_and_gradient(pts)
        
        # Project gradient to tangent space for sphere constraint
        grad_tangent = grad - np.sum(grad * pts, axis=1, keepdims=True) * pts
        
        # Backtracking line search
        lr = lr_init
        improved = False
        for _ in range(30):
            new_pts = pts - lr * grad_tangent
            new_pts = _project_to_sphere(new_pts)
            
            # Check for very close points
            diff = new_pts[:, np.newaxis, :] - new_pts[np.newaxis, :, :]
            dist_sq = np.sum(diff * diff, axis=2)
            np.fill_diagonal(dist_sq, 1.0)
            min_dist_sq = np.min(dist_sq)
            
            if min_dist_sq < 1e-16:
                lr *= 0.5
                continue
            
            new_energy = _log_energy(new_pts)
            
            if new_energy < current_energy - 1e-12:
                pts = new_pts
                current_energy = new_energy
                improved = True
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_pts = pts.copy()
                break
            else:
                lr *= 0.5
        
        if not improved:
            # Try with smaller initial lr
            lr = lr_init * 0.1
            for _ in range(20):
                new_pts = pts - lr * grad_tangent
                new_pts = _project_to_sphere(new_pts)
                new_energy = _log_energy(new_pts)
                if new_energy < current_energy - 1e-12:
                    pts = new_pts
                    current_energy = new_energy
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_pts = pts.copy()
                    break
                lr *= 0.5
            else:
                # No improvement, break
                break
    
    return best_pts, best_energy


def _simulated_annealing(pts, max_iter=3000, t_init=0.01, t_min=1e-6, deadline=None, rng=None):
    """Simulated annealing for escaping local minima."""
    if rng is None:
        rng = np.random.default_rng()
    pts = pts.copy()
    pts = _project_to_sphere(pts)
    current_energy = _log_energy(pts)
    best_pts = pts.copy()
    best_energy = current_energy
    
    T = t_init
    cooling = (t_min / t_init) ** (1.0 / max_iter)
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        noise_scale = 0.005 * (1.0 - 0.5 * it / max_iter)
        new_pts = pts + noise_scale * rng.standard_normal((N, 3))
        new_pts = _project_to_sphere(new_pts)
        
        new_energy = _log_energy(new_pts)
        delta = new_energy - current_energy
        
        if delta < 0 or rng.random() < math.exp(-delta / max(T, 1e-10)):
            pts = new_pts
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_pts = pts.copy()
        
        T *= cooling
    
    return best_pts, best_energy


def _high_precision_polish(pts, deadline):
    """Final high-precision polish with very tight tolerances and multiple passes."""
    pts = _project_to_sphere(pts.copy())
    
    def objective_and_grad(flat_pts):
        p = flat_pts.reshape(N, 3)
        p = _project_to_sphere(p)
        e, g = _energy_and_gradient(p)
        g_tangent = g - np.sum(g * p, axis=1, keepdims=True) * p
        return e, g_tangent.flatten()
    
    x0 = pts.flatten()
    current_energy = _log_energy(pts)
    
    # Multiple polish passes with decreasing tolerances
    for pass_num, (maxiter, ftol, gtol) in enumerate([
        (500, 1e-14, 1e-12),
        (500, 1e-15, 1e-13),
        (500, 1e-16, 1e-14)
    ]):
        if time.time() > deadline:
            break
        
        try:
            result = minimize(
                objective_and_grad,
                x0,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol, 'maxls': 100}
            )
            if time.time() > deadline:
                break
            opt_pts = _project_to_sphere(result.x.reshape(N, 3))
            opt_energy = _log_energy(opt_pts)
            
            if opt_energy < current_energy:
                pts = opt_pts
                current_energy = opt_energy
                x0 = opt_pts.flatten()
        except Exception:
            break
    
    return pts, current_energy


def construct():
    deadline = time.time() + 145.0
    rng = np.random.default_rng(42)
    
    best_pts = None
    best_energy = float('inf')
    
    fib_pts = _fibonacci_points(N)
    
    # Multi-restart with diverse initializations
    initializations = [fib_pts]
    
    # Add random points with different seeds
    for seed in [1, 7, 13, 21, 33]:
        r = np.random.default_rng(seed)
        initializations.append(_random_points(N, r))
    
    # Add perturbed Fibonacci points
    for scale in [0.03, 0.07, 0.12, 0.2, 0.3]:
        perturbed = fib_pts + scale * rng.standard_normal((N, 3))
        initializations.append(_project_to_sphere(perturbed))
    
    # Add ring-based initializations
    for n_rings in [3, 5, 7]:
        ring_pts = []
        # Distribute N points among n_rings as evenly as possible
        base_count = N // n_rings
        remainder = N % n_rings
        for ring_idx in range(n_rings):
            z = 1.0 - 2.0 * (ring_idx + 0.5) / n_rings
            r = np.sqrt(1.0 - z * z)
            count = base_count + (1 if ring_idx < remainder else 0)
            for k in range(count):
                theta = 2.0 * np.pi * k / count + np.pi / count
                ring_pts.append([r * np.cos(theta), r * np.sin(theta), z])
        # Ensure exactly N points (safety check, should be N by construction)
        if len(ring_pts) != N:
             # Fallback if logic fails, though it shouldn't
             ring_pts = ring_pts[:N]
             while len(ring_pts) < N:
                 ring_pts.append(ring_pts[0])
        ring_pts = np.array(ring_pts)
        initializations.append(_project_to_sphere(ring_pts))
    
    for init_pts in initializations:
        if time.time() > deadline - 25.0:
            break
        
        # Phase 1: L-BFGS-B optimization (fast convergence)
        lgbfgs_pts, lgbfgs_energy = _lgbfgs_optimize(init_pts, deadline - 5.0, maxiter=300)
        
        if lgbfgs_energy < best_energy:
            best_energy = lgbfgs_energy
            best_pts = lgbfgs_pts.copy()
        
        # Phase 2: Gradient descent for further refinement
        if time.time() < deadline - 15.0:
            gd_pts, gd_energy = _gradient_descent(lgbfgs_pts, max_iter=3000, lr_init=0.01, deadline=deadline - 10.0)
            
            if gd_energy < best_energy:
                best_energy = gd_energy
                best_pts = gd_pts.copy()
        
        # Phase 3: Simulated annealing to escape local minima
        if time.time() < deadline - 15.0:
            sa_pts, sa_energy = _simulated_annealing(lgbfgs_pts, max_iter=1000, t_init=0.003, t_min=1e-7, deadline=deadline - 10.0, rng=rng)
            
            if sa_energy < best_energy:
                best_energy = sa_energy
                best_pts = sa_pts.copy()
            
            # Phase 4: L-BFGS-B from SA result
            if time.time() < deadline - 10.0:
                lgbfgs2_pts, lgbfgs2_energy = _lgbfgs_optimize(sa_pts, deadline - 5.0, maxiter=200)
                
                if lgbfgs2_energy < best_energy:
                    best_energy = lgbfgs2_energy
                    best_pts = lgbfgs2_pts.copy()
    
    if best_pts is None:
        best_pts = fib_pts
    
    # Final high-precision polish
    if time.time() < deadline - 5.0:
        polished_pts, polished_energy = _high_precision_polish(best_pts, deadline)
        if polished_energy < best_energy:
            best_energy = polished_energy
            best_pts = polished_pts
    
    best_pts = _project_to_sphere(best_pts)
    
    # Validate: check for very close points
    dists = np.linalg.norm(best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :], axis=2)
    np.fill_diagonal(dists, 1.0)
    if np.min(dists) < 1e-9:
        best_pts = fib_pts
    
    return best_pts
# EVOLVE-BLOCK-END
