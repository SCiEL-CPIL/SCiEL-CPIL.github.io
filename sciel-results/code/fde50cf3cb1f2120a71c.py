"""Optimized logarithmic energy minimizer for 100 points on the unit sphere."""
import numpy as np
import time
import math

N = 100
MAX_TIME = 115.0  # Safety margin before 120s limit


def _log_energy_and_grad(pts):
    """Compute logarithmic energy and its gradient on the sphere.
    
    Energy: sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    Gradient (Euclidean): for point i, grad_i = -sum_{j!=i} (x_i - x_j)/|x_i - x_j|^2
    Riemannian gradient: project to tangent space: grad_t = grad - (grad·x_i)*x_i
    """
    # Compute pairwise distances
    # pts is (N, 3)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (N, N, 3)
    dists = np.sqrt(np.sum(diff * diff, axis=2))  # (N, N)
    
    # Avoid division by zero on diagonal
    np.fill_diagonal(dists, 1.0)
    
    # Energy: sum over i<j of log(1/d_ij) = -sum over i<j of log(d_ij)
    # Use upper triangle
    iu = np.triu_indices(N, k=1)
    energy = -np.sum(np.log(dists[iu]))
    
    # Euclidean gradient: grad_i = -sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    # diff[i,j] = x_i - x_j
    # grad_i = -sum_j diff[i,j] / dists[i,j]^2
    inv_dist_sq = 1.0 / (dists * dists)
    np.fill_diagonal(inv_dist_sq, 0.0)
    eucl_grad = -np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)  # (N, 3)
    
    # Riemannian gradient (project to tangent space)
    # grad_t_i = eucl_grad_i - (eucl_grad_i · x_i) * x_i
    dot = np.sum(eucl_grad * pts, axis=1)  # (N,)
    riem_grad = eucl_grad - dot[:, np.newaxis] * pts  # (N, 3)
    
    return energy, riem_grad


def _retract(pts, grad_t, step):
    """Retraction: move along geodesic (exponential map) on the sphere.
    
    For small steps, use the approximation: x_new = x + step * grad_t, then normalize.
    For better accuracy, use the exponential map.
    """
    # Simple retraction: add gradient step, then normalize
    pts_new = pts + step * grad_t
    norms = np.sqrt(np.sum(pts_new * pts_new, axis=1, keepdims=True))
    norms = np.maximum(norms, 1e-15)
    pts_new = pts_new / norms
    return pts_new


def _fibonacci_spiral(n, offset=0.0, rotation=0.0):
    """Generate points on the sphere using Fibonacci spiral with offset and rotation."""
    i = np.arange(n) + 0.5 + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i + rotation
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return pts


def _random_sphere_points(n, rng):
    """Generate random points on the unit sphere."""
    pts = rng.standard_normal((n, 3))
    norms = np.sqrt(np.sum(pts * pts, axis=1, keepdims=True))
    return pts / norms


def _conjugate_gradient_riemannian(pts, max_iter=2000, time_limit=None, start_time=None):
    """Riemannian conjugate gradient method for minimizing logarithmic energy."""
    if start_time is None:
        start_time = time.time()
    
    energy, grad_t = _log_energy_and_grad(pts)
    best_energy = energy
    best_pts = pts.copy()
    
    # Initial search direction
    d = -grad_t.copy()
    
    # Riemannian transport (parallel transport) for updating direction
    def transport(d_prev, grad_prev, grad_new):
        """Parallel transport d_prev from x to x_new using the formula for spheres."""
        # For spheres, parallel transport of vector v from x to y:
        # v_trans = v + (v·(y-x))/|y-x|^2 * (y-x)  -- this is for the tangent plane
        # Actually, simpler: use the formula v_trans = v - (v·x)*x + (v·y)*y + correction
        # Let's use a simpler approach: since we're on the sphere, 
        # v_trans = v + (v·(y-x)) * (y-x) / |y-x|^2
        # But this requires knowing x and y. Let's use a different approach.
        
        # Actually, for the CG method, we can use a simpler update:
        # Just project d onto the tangent space at the new point
        # and use a modified beta.
        pass
    
    for it in range(max_iter):
        if time_limit is not None and time.time() - start_time > time_limit:
            break
        
        # Line search (backtracking)
        step = 0.01
        min_step = 1e-12
        max_step = 1.0
        best_step = None
        best_step_energy = None
        
        for ls in range(50):
            pts_trial = _retract(pts, d, step)
            e_trial, _ = _log_energy_and_grad(pts_trial)
            
            if e_trial < energy:
                best_step = step
                best_step_energy = e_trial
                break
            step *= 0.5
            if step < min_step:
                break
        
        if best_step is None:
            # Try a smaller step or break
            step = min_step
            pts_trial = _retract(pts, d, step)
            e_trial, _ = _log_energy_and_grad(pts_trial)
            if e_trial >= energy:
                break
            best_step = step
            best_step_energy = e_trial
        
        # Update
        pts = pts_trial
        energy = best_step_energy
        
        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
        
        # Compute new gradient
        _, grad_new = _log_energy_and_grad(pts)
        
        # Fletcher-Reeves beta with damping
        grad_norm_sq = np.sum(grad_new * grad_new)
        if grad_norm_sq < 1e-20:
            break
        
        # Compute old grad norm squared
        # We need to store it
        if it == 0:
            old_grad_norm_sq = grad_norm_sq
        else:
            old_grad_norm_sq = np.sum(grad_prev * grad_prev)
        
        beta = np.sum(grad_new * grad_new) / max(old_grad_norm_sq, 1e-20)
        # Damped FR: beta = min(beta, 0.5) to avoid instability
        beta = min(beta, 0.5)
        
        # Update direction: d_new = -grad_new + beta * transport(d_old)
        # For simplicity, use parallel transport approximation
        # On the sphere, we can approximate transport by projecting d onto new tangent space
        d_old_tangent = d - np.sum(d * pts, axis=1, keepdims=True) * pts
        d = -grad_new + beta * d_old_tangent
        
        # Ensure d is in tangent space
        d = d - np.sum(d * pts, axis=1, keepdims=True) * pts
        
        grad_prev = grad_new.copy()
    
    return best_pts, best_energy


def _local_search_with_perturbation(pts, rng, time_limit, start_time):
    """Try to escape local minima using random perturbations."""
    best_pts = pts.copy()
    best_energy, _ = _log_energy_and_grad(pts)
    
    while time.time() - start_time < time_limit:
        # Create a perturbed version
        noise = rng.standard_normal((N, 3)) * 0.1
        pts_pert = pts + noise
        norms = np.sqrt(np.sum(pts_pert * pts_pert, axis=1, keepdims=True))
        pts_pert = pts_pert / norms
        
        # Run CG on the perturbed points
        pts_opt, e_opt = _conjugate_gradient_riemannian(
            pts_pert, max_iter=500, 
            time_limit=min(10.0, time_limit - (time.time() - start_time)),
            start_time=start_time
        )
        
        if e_opt < best_energy:
            best_energy = e_opt
            best_pts = pts_opt.copy()
            pts = pts_opt.copy()
        else:
            # Keep the original if perturbation didn't help
            pass
    
    return best_pts, best_energy


def construct():
    start_time = time.time()
    deadline = start_time + MAX_TIME
    
    rng = np.random.default_rng(42)
    
    # Try multiple initial configurations
    initializations = []
    
    # Fibonacci spiral
    initializations.append(_fibonacci_spiral(N))
    
    # Fibonacci spiral with different offsets
    for offset in [0.1, 0.3, 0.7]:
        initializations.append(_fibonacci_spiral(N, offset=offset))
    
    # Fibonacci spiral with rotation
    for rot in [0.5, 1.0, 2.0]:
        initializations.append(_fibonacci_spiral(N, rotation=rot))
    
    # Random initializations
    for _ in range(5):
        initializations.append(_random_sphere_points(N, rng))
    
    best_pts = None
    best_energy = np.inf
    
    for init_pts in initializations:
        if time.time() > deadline - 5.0:
            break
        
        # Run CG optimization
        pts_opt, e_opt = _conjugate_gradient_riemannian(
            init_pts, max_iter=2000,
            time_limit=min(15.0, deadline - time.time() - 5.0),
            start_time=start_time
        )
        
        if e_opt < best_energy:
            best_energy = e_opt
            best_pts = pts_opt.copy()
    
    # If we have time, try local search with perturbation
    if best_pts is not None and time.time() < deadline - 10.0:
        best_pts, best_energy = _local_search_with_perturbation(
            best_pts, rng, deadline - time.time() - 5.0, start_time
        )
    
    # Final polish: run more CG iterations with tighter tolerance
    if best_pts is not None and time.time() < deadline - 3.0:
        best_pts, best_energy = _conjugate_gradient_riemannian(
            best_pts, max_iter=5000,
            time_limit=deadline - time.time() - 2.0,
            start_time=start_time
        )
    
    # Ensure points are on the unit sphere
    norms = np.sqrt(np.sum(best_pts * best_pts, axis=1, keepdims=True))
    best_pts = best_pts / np.maximum(norms, 1e-15)
    
    return best_pts
