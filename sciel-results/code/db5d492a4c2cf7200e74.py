"""Optimized logarithmic energy minimization for 100 points on the unit sphere."""
import time
import math
import numpy as np

N = 100

# EVOLVE-BLOCK-START
def _fibonacci_spiral():
    """Generate Fibonacci spiral points on the unit sphere."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere_points(n, seed=None):
    """Generate random points on the unit sphere."""
    rng = np.random.default_rng(seed)
    pts = rng.normal(size=(n, 3))
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / np.maximum(norms, 1e-15)


def _project_to_sphere(pts):
    """Project points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def _log_energy(pts):
    """
    Compute logarithmic energy: sum_{i<j} log(1/|x_i - x_j|)
    Using chordal distances on unit sphere: |x_i - x_j| = sqrt(2 - 2*x_i.x_j)
    So log(1/|x_i - x_j|) = -0.5 * log(2 - 2*x_i.x_j)
    """
    # Compute all pairwise dot products
    # pts is (N, 3), so pts @ pts.T is (N, N)
    dot = pts @ pts.T
    # Clip to avoid log of negative or zero
    # 2 - 2*dot should be in (0, 4] for distinct points on sphere
    dist_sq = 2.0 - 2.0 * dot
    np.fill_diagonal(dist_sq, 1.0)  # avoid self-interaction issues
    # Only upper triangle
    iu = np.triu_indices(N, 1)
    vals = dist_sq[iu]
    vals = np.maximum(vals, 1e-30)
    return -0.5 * math.fsum(np.log(vals).tolist())


def _log_energy_and_grad(pts):
    """
    Compute energy and gradient of log(1/|x_i - x_j|) on unit sphere.
    
    Energy: E = sum_{i<j} log(1/|x_i - x_j|) = -0.5 * sum_{i<j} log(2 - 2*x_i.x_j)
    
    For a point x_k, the gradient contribution from pair (i,j) containing k:
    If k=i: d/dx_i log(1/|x_i - x_j|) = -(x_i - x_j)/|x_i - x_j|^2
    If k=j: d/dx_j log(1/|x_i - x_j|) = -(x_j - x_i)/|x_i - x_j|^2
    
    But we need the gradient on the sphere, so we project: grad_sphere = grad - (grad.x_k)*x_k
    
    Actually, let's use the formula directly.
    E = -0.5 * sum_{i<j} log(2 - 2*x_i.x_j)
    dE/dx_k = -0.5 * sum_{j!=k} d/dx_k log(2 - 2*x_k.x_j)
            = -0.5 * sum_{j!=k} (-2*x_j)/(2 - 2*x_k.x_j)
            = sum_{j!=k} x_j / (2 - 2*x_k.x_j)
            = sum_{j!=k} x_j / (2*(1 - x_k.x_j))
    
    Then project to tangent: grad_tangent = grad - (grad . x_k) * x_k
    """
    dot = pts @ pts.T
    # dist_sq[i,j] = 2 - 2*dot[i,j]
    dist_sq = 2.0 - 2.0 * dot
    np.fill_diagonal(dist_sq, 1.0)
    
    # Energy
    iu = np.triu_indices(N, 1)
    vals = dist_sq[iu]
    vals = np.maximum(vals, 1e-30)
    energy = -0.5 * math.fsum(np.log(vals).tolist())
    
    # Gradient: grad_k = sum_{j != k} x_j / dist_sq[k,j]
    # dist_sq is (N, N), symmetric
    # For each k, grad_k = sum_j x_j / dist_sq[k,j] for j != k
    # We can compute this as: (1/dist_sq) @ pts, but with diagonal excluded
    inv_dist_sq = 1.0 / dist_sq
    np.fill_diagonal(inv_dist_sq, 0.0)
    grad = inv_dist_sq @ pts  # (N, 3)
    
    # Project gradient to tangent space
    # grad_proj = grad - (grad . pts) * pts
    grad_dot = np.sum(grad * pts, axis=1, keepdims=True)
    grad = grad - grad_dot * pts
    
    return energy, grad


def _optimize_lbfgs(pts, max_iter=2000, ftol=1e-16, gtol=1e-12, time_budget=None):
    """
    L-BFGS optimization on the sphere using retraction (project back to sphere).
    """
    from scipy.optimize import minimize
    
    start = time.time()
    
    def objective_and_grad(x):
        if time_budget is not None and time.time() - start > time_budget:
            # Return current value and raise to stop
            pts = _project_to_sphere(x.reshape(N, 3))
            e, _ = _log_energy_and_grad(pts)
            raise RuntimeError("Time budget exceeded")
        pts = _project_to_sphere(x.reshape(N, 3))
        e, g = _log_energy_and_grad(pts)
        return e, g.flatten()
    
    try:
        result = minimize(objective_and_grad, pts.flatten(), method='L-BFGS-B',
                          jac=True, options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol,
                                             'maxls': 50, 'maxfun': 5000})
        pts_opt = _project_to_sphere(result.x.reshape(N, 3))
        return pts_opt, result.fun, result
    except RuntimeError:
        # Time exceeded, return best available
        pts_opt = _project_to_sphere(pts)
        e = _log_energy(pts_opt)
        return pts_opt, e, None
    except Exception:
        pts_opt = _project_to_sphere(pts)
        e = _log_energy(pts_opt)
        return pts_opt, e, None


def _refine(pts, time_budget=5.0):
    """High-precision refinement using adaptive step size and backtracking line search."""
    start = time.time()
    pts = pts.copy()
    best_e = _log_energy(pts)
    
    for step in range(500):
        if time.time() - start > time_budget:
            break
        
        e, g = _log_energy_and_grad(pts)
        
        # Check gradient norm
        gnorm = np.linalg.norm(g)
        if gnorm < 1e-15:
            break
        
        # Adaptive line search with backtracking
        lr = 0.1
        best_e_local = e
        best_pts_local = pts.copy()
        
        for _ in range(20):
            new_pts = pts - lr * g
            new_pts = _project_to_sphere(new_pts)
            new_e, _ = _log_energy_and_grad(new_pts)
            
            if new_e < best_e_local:
                best_e_local = new_e
                best_pts_local = new_pts
            else:
                lr *= 0.5
                if lr < 1e-10:
                    break
        
        if np.allclose(best_pts_local, pts, atol=1e-16):
            break
        
        pts = best_pts_local
        best_e = best_e_local
    
    return pts


def _multi_start_optimize(time_budget_per_start=10.0, num_starts=3):
    """Try multiple initializations and return the best result."""
    start_time = time.time()
    
    # Define different initializations
    initializations = []
    
    # 1. Fibonacci spiral
    initializations.append(_fibonacci_spiral())
    
    # 2. Random points with different seeds
    for seed in [42, 123]:
        initializations.append(_random_sphere_points(N, seed=seed))
    
    best_pts = None
    best_e = float('inf')
    
    for idx, init_pts in enumerate(initializations):
        if time.time() - start_time > time_budget_per_start * (idx + 1):
            break
        
        init_pts = _project_to_sphere(init_pts)
        
        try:
            # L-BFGS optimization
            pts_opt, e_opt, _ = _optimize_lbfgs(init_pts, max_iter=2000, ftol=1e-15, gtol=1e-13,
                                                 time_budget=min(time_budget_per_start, 
                                                                 time_budget_per_start * (idx + 1) - (time.time() - start_time)))
            
            if e_opt < best_e:
                best_e = e_opt
                best_pts = pts_opt.copy()
        except Exception:
            pass
    
    return best_pts, best_e


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Phase 1: Multi-start optimization
    best_pts, best_e = _multi_start_optimize(time_budget_per_start=15.0, num_starts=3)
    
    if best_pts is None:
        # Fallback to Fibonacci spiral
        best_pts = _fibonacci_spiral()
        best_e = _log_energy(best_pts)
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > deadline - 20.0:
        return _project_to_sphere(best_pts)
    
    # Phase 2: High-precision refinement
    time_for_refine = min(15.0, deadline - time.time() - 10.0)
    if time_for_refine > 2.0:
        try:
            refined = _refine(best_pts, time_budget=time_for_refine)
            e_ref = _log_energy(refined)
            if e_ref < best_e:
                best_pts = refined
                best_e = e_ref
        except Exception:
            pass
    
    # Check time
    if time.time() > deadline - 10.0:
        return _project_to_sphere(best_pts)
    
    # Phase 3: Second round of L-BFGS from refined point
    time_for_lbfgs2 = min(10.0, deadline - time.time() - 5.0)
    if time_for_lbfgs2 > 2.0:
        try:
            pts_opt2, e_opt2, _ = _optimize_lbfgs(best_pts, max_iter=1500, ftol=1e-16, gtol=1e-14,
                                                   time_budget=time_for_lbfgs2)
            if e_opt2 < best_e:
                best_pts = pts_opt2
                best_e = e_opt2
        except Exception:
            pass
    
    # Check time
    if time.time() > deadline - 5.0:
        return _project_to_sphere(best_pts)
    
    # Phase 4: Final refinement
    time_for_final = min(5.0, deadline - time.time() - 2.0)
    if time_for_final > 1.0:
        try:
            refined2 = _refine(best_pts, time_budget=time_for_final)
            e_ref2 = _log_energy(refined2)
            if e_ref2 < best_e:
                best_pts = refined2
                best_e = e_ref2
        except Exception:
            pass
    
    # Ensure validity
    best_pts = _project_to_sphere(best_pts)
    
    # Verify no points are too close
    dot = best_pts @ best_pts.T
    dist_sq = 2.0 - 2.0 * dot
    np.fill_diagonal(dist_sq, 1.0)
    iu = np.triu_indices(N, 1)
    min_dist_sq = np.min(dist_sq[iu])
    if min_dist_sq < 1e-18:  # distance < 1e-9
        # Perturb slightly
        best_pts += np.random.normal(0, 1e-6, (N, 3))
        best_pts = _project_to_sphere(best_pts)
    
    return best_pts
# EVOLVE-BLOCK-END
