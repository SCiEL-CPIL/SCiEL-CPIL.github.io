"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import distance_matrix
import math
import time

N = 100
_START_TIME = time.time()
_TIME_LIMIT = 115.0  # Leave 5s buffer

# EVOLVE-BLOCK-START
def _log_energy_and_grad(pts):
    """
    Compute logarithmic energy E = sum_{i<j} log(1/|xi - xj|) = -sum_{i<j} log(|xi - xj|)
    and its gradient with respect to pts (N x 3).
    """
    pts = np.asarray(pts, dtype=np.float64)
    n = pts.shape[0]
    
    # Compute pairwise distances efficiently
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.einsum('ijk,ijk->ij', diff, diff)
    dist_sq = np.maximum(dist_sq, 1e-30)
    dist = np.sqrt(dist_sq)
    dist_safe = np.maximum(dist, 1e-15)
    
    # Energy: sum over i<j of -log(dist[i,j])
    iu = np.triu_indices(n, k=1)
    d_upper = dist_safe[iu]
    energy = -math.fsum(np.log(d_upper).tolist())
    
    # Gradient: grad[i] = -sum_{j != i} (pts[i] - pts[j]) / ||pts[i] - pts[j]||^2
    W = 1.0 / dist_sq
    np.fill_diagonal(W, 0.0)
    
    row_sum = W.sum(axis=1)
    grad = -pts * row_sum[:, np.newaxis] + W @ pts
    
    return energy, grad

def _project_to_sphere(pts):
    """Project points onto the unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def _time_remaining():
    """Return seconds remaining before timeout."""
    return _TIME_LIMIT - (time.time() - _START_TIME)

def _riemannian_conjugate_gradient(pts, max_iter=500, tol=1e-8, time_budget=None):
    """
    Perform Riemannian Conjugate Gradient on the sphere with improved line search.
    """
    pts = _project_to_sphere(pts.copy())
    current_energy, current_grad = _log_energy_and_grad(pts)
    
    # Project gradient to tangent space
    grad_dot = np.einsum('ij,ij->i', current_grad, pts)
    grad_tangent = current_grad - grad_dot[:, np.newaxis] * pts
    
    # Initial direction
    d = -grad_tangent
    
    for k in range(max_iter):
        # Check time
        if time_budget is not None and _time_remaining() < time_budget:
            break
        
        # Adaptive line search (Backtracking)
        alpha = 0.1
        improved = False
        best_alpha = 0.0
        best_energy = current_energy
        best_pts_new = pts.copy()
        best_grad_new = current_grad.copy()
        
        for backtrack in range(50):
            # Step in tangent space
            pts_new = pts + alpha * d
            # Retraction (project to sphere)
            pts_new = _project_to_sphere(pts_new)
            
            new_energy, new_grad = _log_energy_and_grad(pts_new)
            
            if new_energy < current_energy:
                best_alpha = alpha
                best_energy = new_energy
                best_pts_new = pts_new
                best_grad_new = new_grad
                improved = True
            else:
                alpha *= 0.7  # More aggressive backtracking
            
            if alpha < 1e-12:
                break
        
        if not improved:
            break
            
        pts = best_pts_new
        current_energy = best_energy
        current_grad = best_grad_new
        
        # Check convergence
        grad_norm = np.linalg.norm(grad_tangent)
        if grad_norm < tol:
            break
        
        # Compute new tangent gradient
        grad_dot_new = np.einsum('ij,ij->i', current_grad, pts)
        grad_tangent_new = current_grad - grad_dot_new[:, np.newaxis] * pts
        
        # Fletcher-Reeves beta with damping
        num = np.sum(grad_tangent_new ** 2)
        den = np.sum(grad_tangent ** 2)
        if den < 1e-15:
            beta = 0.0
        else:
            beta = num / den
            beta = min(beta, 0.8)  # Damp beta to avoid divergence
        
        # Transport d to new tangent space
        d_dot_new = np.einsum('ij,ij->i', d, pts)
        d_tangent = d - d_dot_new[:, np.newaxis] * pts
        
        # New direction
        d = -grad_tangent_new + beta * d_tangent
        
        # Periodic steepest descent reset to avoid cycling
        if k % 20 == 19:
            d = -grad_tangent_new
        
        # Update grad_tangent for next iteration's beta calculation
        grad_tangent = grad_tangent_new
        
    return pts, current_energy

def _fibonacci_spiral(n=N, offset=0.5):
    """Generate Fibonacci spiral points on the unit sphere."""
    i = np.arange(n) + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _icosahedral_fibonacci(n=N):
    """
    Generate initial points by placing icosahedral vertices and filling
    remaining positions with Fibonacci-spiral-like distribution on faces.
    This exploits structural symmetry of low-energy spherical codes.
    """
    t = (1.0 + np.sqrt(5.0)) / 2.0
    ico = np.array([
        [-1,  t,  0], [ 1,  t,  0], [-1, -t,  0], [ 1, -t,  0],
        [ 0, -1,  t], [ 0,  1,  t], [ 0, -1, -t], [ 0,  1, -t],
        [ t,  0, -1], [ t,  0,  1], [-t,  0, -1], [-t,  0,  1]
    ], dtype=np.float64)
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
    
    # Use icosahedron as seed, then add Fibonacci points with small perturbation
    fib = _fibonacci_spiral(n)
    # Blend: use icosahedral structure for first 12, rest from Fibonacci
    if n >= 12:
        pts = np.vstack([ico, fib[12:]])
    else:
        pts = fib
    
    # Small perturbation to break exact symmetry
    noise = np.random.normal(0, 0.01, size=(n, 3))
    pts = pts + noise
    return _project_to_sphere(pts)

def _golden_ratio_spiral(n=N, start_offset=0.0):
    """Generate points using golden ratio spacing with different starting offset."""
    i = np.arange(n) + start_offset
    z = 1.0 - 2.0 * (i + 0.5) / n
    golden = 2.39996322972865332  # 2*pi / phi^2
    phi_angle = golden * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0))
    return np.column_stack([r * np.cos(phi_angle), r * np.sin(phi_angle), z])

def _icosahedral_based_init(n=N):
    """Generate initial points based on icosahedral symmetry with perturbations."""
    # Start with icosahedron vertices
    t = (1.0 + np.sqrt(5.0)) / 2.0
    ico = np.array([
        [-1,  t,  0], [ 1,  t,  0], [-1, -t,  0], [ 1, -t,  0],
        [ 0, -1,  t], [ 0,  1,  t], [ 0, -1, -t], [ 0,  1, -t],
        [ t,  0, -1], [ t,  0,  1], [-t,  0, -1], [-t,  0,  1]
    ], dtype=np.float64)
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
    
    # Add perturbed Fibonacci points
    fib = _fibonacci_spiral(n)
    noise = np.random.normal(0, 0.05, size=(n, 3))
    pts = fib + noise
    return _project_to_sphere(pts)

def _refine_with_lbfgs(pts, max_iter=3000):
    """Refine points using L-BFGS-B in 3D with sphere constraint via penalty."""
    def penalty_energy(x):
        pts = x.reshape(-1, 3)
        # Compute distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.einsum('ijk,ijk->ij', diff, diff)
        dist_sq = np.maximum(dist_sq, 1e-30)
        dist = np.sqrt(dist_sq)
        dist_safe = np.maximum(dist, 1e-15)
        
        iu = np.triu_indices(N, k=1)
        energy = -np.sum(np.log(dist_safe[iu]))
        
        # Penalty for being off the sphere
        norms = np.linalg.norm(pts, axis=1)
        penalty = 1e6 * np.sum((norms - 1.0)**2)
        
        return energy + penalty
    
    def penalty_grad(x):
        pts = x.reshape(-1, 3)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.einsum('ijk,ijk->ij', diff, diff)
        dist_sq_safe = np.maximum(dist_sq, 1e-30)
        
        # Gradient of energy
        W = 1.0 / dist_sq_safe
        np.fill_diagonal(W, 0.0)
        
        row_sum = W.sum(axis=1)
        grad_energy = -pts * row_sum[:, np.newaxis] + W @ pts
        
        # Gradient of penalty
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        grad_penalty = 2e6 * (norms - 1.0) * pts / norms
        
        return (grad_energy + grad_penalty).flatten()
    
    x0 = pts.flatten()
    res = minimize(penalty_energy, x0, jac=penalty_grad, method='L-BFGS-B', 
                   options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8,
                            'maxls': 50})
    
    best_pts = _project_to_sphere(res.x.reshape(-1, 3))
    return best_pts

def _perturb_and_refine(pts, noise_scale=0.01, max_iter=2000):
    """Perturb points and refine using RCG."""
    noise = np.random.normal(0, noise_scale, size=(N, 3))
    perturbed = _project_to_sphere(pts + noise)
    res_pts, res_energy = _riemannian_conjugate_gradient(perturbed, max_iter=max_iter, tol=1e-10)
    return res_pts, res_energy

def construct():
    # Multiple initial configurations
    inits = []
    
    # 1. Fibonacci spiral (standard)
    inits.append(_fibonacci_spiral(N, offset=0.5))
    
    # 2. Fibonacci spiral with different offset
    inits.append(_fibonacci_spiral(N, offset=0.0))
    
    # 3. Golden ratio spiral
    inits.append(_golden_ratio_spiral(N, start_offset=0.0))
    
    # 4. Golden ratio spiral with offset
    inits.append(_golden_ratio_spiral(N, start_offset=0.5))
    
    # 5. Perturbed Fibonacci
    noise = np.random.normal(0, 0.1, size=(N, 3))
    inits.append(_project_to_sphere(_fibonacci_spiral(N) + noise))
    
    # 6. Icosahedral-based
    inits.append(_icosahedral_based_init(N))
    
    # 7. Icosahedral Fibonacci hybrid
    inits.append(_icosahedral_fibonacci(N))
    
    # 8. Random points
    inits.append(_project_to_sphere(np.random.normal(size=(N, 3))))
    
    # 9. Rotated Fibonacci
    theta = np.random.uniform(0, 2*np.pi)
    rot = np.array([[np.cos(theta), -np.sin(theta), 0],
                    [np.sin(theta), np.cos(theta), 0],
                    [0, 0, 1]])
    inits.append(_project_to_sphere(_fibonacci_spiral(N) @ rot.T))
    
    # 10. Another random perturbation
    noise2 = np.random.normal(0, 0.2, size=(N, 3))
    inits.append(_project_to_sphere(_fibonacci_spiral(N) + noise2))
    
    # 11. Scaled perturbation
    noise3 = np.random.normal(0, 0.05, size=(N, 3))
    inits.append(_project_to_sphere(_fibonacci_spiral(N) + noise3))
    
    # 12. Random rotation of icosahedral Fibonacci
    theta2 = np.random.uniform(0, 2*np.pi)
    theta3 = np.random.uniform(0, 2*np.pi)
    rot2 = np.array([[np.cos(theta2), -np.sin(theta2)*np.cos(theta3), -np.sin(theta2)*np.sin(theta3)],
                     [np.sin(theta2), np.cos(theta2)*np.cos(theta3), np.cos(theta2)*np.sin(theta3)],
                     [-0, np.sin(theta3), -np.cos(theta3)]])
    inits.append(_project_to_sphere(_icosahedral_fibonacci(N) @ rot2.T))
    
    best_energy = np.inf
    best_pts = inits[0].copy()
    
    # Run Riemannian CG from each initial configuration with time awareness
    for idx, x0 in enumerate(inits):
        if _time_remaining() < 30.0:
            break
        try:
            # Scale iterations based on remaining time
            remaining_frac = _time_remaining() / _TIME_LIMIT
            max_iter = int(2000 * max(0.3, remaining_frac))
            res_pts, res_energy = _riemannian_conjugate_gradient(x0, max_iter=max_iter, tol=1e-9)
            if res_energy < best_energy:
                best_energy = res_energy
                best_pts = res_pts.copy()
        except Exception:
            pass
    
    # Additional refinement passes from best solution with increasing perturbation
    for pass_num in range(7):
        if _time_remaining() < 15.0:
            break
        try:
            # Small perturbation to escape local minima
            if pass_num > 0:
                noise_scale = 0.003 * (pass_num + 1)
                max_iter = int(3000 * max(0.3, _time_remaining() / _TIME_LIMIT))
                res_pts, res_energy = _perturb_and_refine(best_pts, noise_scale=noise_scale, max_iter=max_iter)
            else:
                max_iter = int(3000 * max(0.3, _time_remaining() / _TIME_LIMIT))
                res_pts, res_energy = _riemannian_conjugate_gradient(best_pts.copy(), max_iter=max_iter, tol=1e-10)
            
            if res_energy < best_energy:
                best_energy = res_energy
                best_pts = res_pts.copy()
        except Exception:
            pass
    
    # L-BFGS-B refinement
    if _time_remaining() > 10.0:
        try:
            max_iter = int(3000 * max(0.3, _time_remaining() / _TIME_LIMIT))
            refined = _refine_with_lbfgs(best_pts, max_iter=max_iter)
            refined_energy, _ = _log_energy_and_grad(refined)
            if refined_energy < best_energy:
                best_energy = refined_energy
                best_pts = refined
        except Exception:
            pass
    
    # One more RCG pass after L-BFGS
    if _time_remaining() > 8.0:
        try:
            max_iter = int(2000 * max(0.3, _time_remaining() / _TIME_LIMIT))
            res_pts, res_energy = _riemannian_conjugate_gradient(best_pts, max_iter=max_iter, tol=1e-10)
            if res_energy < best_energy:
                best_energy = res_energy
                best_pts = res_pts
        except Exception:
            pass
    
    # Final fine-tuning with very small perturbations
    for final_pass in range(4):
        if _time_remaining() < 5.0:
            break
        try:
            noise_scale = 0.001 * (final_pass + 1)
            max_iter = int(2000 * max(0.2, _time_remaining() / _TIME_LIMIT))
            res_pts, res_energy = _perturb_and_refine(best_pts, noise_scale=noise_scale, max_iter=max_iter)
            if res_energy < best_energy:
                best_energy = res_energy
                best_pts = res_pts.copy()
        except Exception:
            pass
    
    # Final projection to ensure unit sphere
    best_pts = _project_to_sphere(best_pts)
    
    # Verify no points are too close
    diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
    dist_sq = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(dist_sq, np.inf)
    min_dist = np.sqrt(np.min(dist_sq))
    
    if min_dist < 1e-9:
        # If points are too close, fall back to Fibonacci spiral
        best_pts = _fibonacci_spiral(N)
        
    # Ensure finite values
    if not np.all(np.isfinite(best_pts)):
        best_pts = _fibonacci_spiral(N)
        
    return best_pts
# EVOLVE-BLOCK-END
