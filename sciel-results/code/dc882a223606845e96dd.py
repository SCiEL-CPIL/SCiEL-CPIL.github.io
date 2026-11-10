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


def _optimize_lbfgs(pts, max_iter=2000, ftol=1e-16, gtol=1e-12):
    """
    L-BFGS optimization on the sphere using retraction (project back to sphere).
    """
    from scipy.optimize import minimize
    
    def objective_and_grad(x):
        pts = _project_to_sphere(x.reshape(N, 3))
        e, g = _log_energy_and_grad(pts)
        return e, g.flatten()
    
    result = minimize(objective_and_grad, pts.flatten(), method='L-BFGS-B',
                      jac=True, options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol,
                                         'maxls': 50, 'maxfun': 5000})
    pts_opt = _project_to_sphere(result.x.reshape(N, 3))
    return pts_opt, result.fun, result


def _refine(pts, time_budget=5.0):
    """High-precision refinement using Newton-like steps with tight tolerances."""
    start = time.time()
    pts = pts.copy()
    
    for step in range(200):
        if time.time() - start > time_budget:
            break
        e, g = _log_energy_and_grad(pts)
        
        # Check gradient norm
        gnorm = np.linalg.norm(g)
        if gnorm < 1e-14:
            break
        
        # Line search: try small steps
        best_e = e
        best_pts = pts.copy()
        for lr in [0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]:
            new_pts = pts - lr * g
            new_pts = _project_to_sphere(new_pts)
            new_e, _ = _log_energy_and_grad(new_pts)
            if new_e < best_e:
                best_e = new_e
                best_pts = new_pts
        
        if np.allclose(best_pts, pts, atol=1e-15):
            break
        pts = best_pts
    
    return pts


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Start with Fibonacci spiral
    pts = _fibonacci_spiral()
    pts = _project_to_sphere(pts)
    
    best_pts = pts.copy()
    best_e = _log_energy(pts)
    
    # Main L-BFGS optimization
    try:
        pts_opt, e_opt, result = _optimize_lbfgs(pts, max_iter=3000, ftol=1e-15, gtol=1e-13)
        if e_opt < best_e:
            best_pts = pts_opt
            best_e = e_opt
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline - 15.0:
        return best_pts
    
    # High-precision refinement
    try:
        refined = _refine(best_pts, time_budget=min(10.0, deadline - time.time() - 5.0))
        e_ref = _log_energy(refined)
        if e_ref < best_e:
            best_pts = refined
            best_e = e_ref
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline - 5.0:
        return best_pts
    
    # Try a second round of L-BFGS from the refined point
    try:
        pts_opt2, e_opt2, result2 = _optimize_lbfgs(best_pts, max_iter=1000, ftol=1e-16, gtol=1e-14)
        if e_opt2 < best_e:
            best_pts = pts_opt2
            best_e = e_opt2
    except Exception:
        pass
    
    # Final refinement
    if time.time() < deadline - 5.0:
        try:
            refined2 = _refine(best_pts, time_budget=min(5.0, deadline - time.time() - 2.0))
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
