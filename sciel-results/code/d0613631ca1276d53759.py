"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import distance_matrix
import math

N = 100

# EVOLVE-BLOCK-START
def _log_energy_and_grad(pts):
    """
    Compute logarithmic energy E = sum_{i<j} log(1/|xi - xj|) = -sum_{i<j} log(|xi - xj|)
    and its gradient with respect to pts (N x 3).
    """
    pts = np.asarray(pts, dtype=np.float64)
    n = pts.shape[0]
    
    # Compute pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    
    # Avoid log(0) by clipping
    dist_safe = np.maximum(dist, 1e-15)
    
    # Energy: sum over i<j of -log(dist[i,j])
    iu = np.triu_indices(n, k=1)
    d_upper = dist_safe[iu]
    energy = -np.sum(np.log(d_upper))
    
    # Gradient: grad[i] = -sum_{j != i} (pts[i] - pts[j]) / ||pts[i] - pts[j]||^2
    dist_sq = dist_safe ** 2
    
    W = np.zeros_like(dist_sq)
    np.fill_diagonal(W, 0.0)
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

def _riemannian_conjugate_gradient(pts, max_iter=500, tol=1e-8):
    """
    Perform Riemannian Conjugate Gradient on the sphere.
    """
    pts = _project_to_sphere(pts.copy())
    current_energy, current_grad = _log_energy_and_grad(pts)
    
    # Project gradient to tangent space
    grad_dot = np.einsum('ij,ij->i', current_grad, pts)
    grad_tangent = current_grad - grad_dot[:, np.newaxis] * pts
    
    # Initial direction
    d = -grad_tangent
    
    for k in range(max_iter):
        # Line search (Backtracking)
        alpha = 0.1
        improved = False
        for backtrack in range(30):
            # Step in tangent space
            pts_new = pts + alpha * d
            # Retraction (project to sphere)
            pts_new = _project_to_sphere(pts_new)
            
            new_energy, new_grad = _log_energy_and_grad(pts_new)
            
            if new_energy < current_energy:
                pts = pts_new
                current_energy = new_energy
                current_grad = new_grad
                improved = True
                break
            else:
                alpha *= 0.5
        
        if not improved:
            break
            
        # Check convergence
        if np.linalg.norm(grad_tangent) < tol:
            break
            
        # Compute new tangent gradient
        grad_dot_new = np.einsum('ij,ij->i', current_grad, pts)
        grad_tangent_new = current_grad - grad_dot_new[:, np.newaxis] * pts
        
        # Fletcher-Reeves beta
        num = np.sum(grad_tangent_new ** 2)
        den = np.sum(grad_tangent ** 2)
        if den < 1e-15:
            beta = 0.0
        else:
            beta = num / den
            
        # Transport d to new tangent space
        # d is tangent to pts_old.
        # Project d onto tangent space of pts_new
        d_dot_new = np.einsum('ij,ij->i', d, pts)
        d_tangent = d - d_dot_new[:, np.newaxis] * pts
        
        # New direction
        d = -grad_tangent_new + beta * d_tangent
        
        # Update grad_tangent for next iteration's beta calculation
        grad_tangent = grad_tangent_new
        
    return pts, current_energy

def construct():
    # Start from a Fibonacci spiral (good initial guess)
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts_init = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_energy = np.inf
    best_pts = pts_init.copy()
    
    n_restarts = 20
    
    for restart in range(n_restarts):
        if restart == 0:
            x0 = pts_init.copy()
        elif restart == 1:
            # Random rotation of Fibonacci
            theta = np.random.uniform(0, 2*np.pi)
            rot = np.array([[np.cos(theta), -np.sin(theta), 0],
                            [np.sin(theta), np.cos(theta), 0],
                            [0, 0, 1]])
            x0 = pts_init @ rot.T
        else:
            # Random perturbation or random init
            if restart % 2 == 0:
                # Perturbed Fibonacci
                noise = np.random.normal(0, 0.1, size=(N, 3))
                x0 = pts_init + noise
            else:
                # Random points on sphere
                x0 = np.random.normal(size=(N, 3))
                
        x0 = _project_to_sphere(x0)
        
        try:
            res_pts, res_energy = _riemannian_conjugate_gradient(x0, max_iter=1000, tol=1e-8)
            if res_energy < best_energy:
                best_energy = res_energy
                best_pts = res_pts
        except Exception as e:
            pass

    # Final projection to ensure unit sphere
    best_pts = _project_to_sphere(best_pts)
    
    # Verify no points are too close
    dist = distance_matrix(best_pts, best_pts)
    np.fill_diagonal(dist, np.inf)
    min_dist = np.min(dist)
    
    if min_dist < 1e-9:
        # If points are too close, fall back to Fibonacci spiral
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        best_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        
    # Ensure finite values
    if not np.all(np.isfinite(best_pts)):
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        best_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        
    return best_pts
# EVOLVE-BLOCK-END
