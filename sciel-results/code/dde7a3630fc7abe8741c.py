"""Rewritten program: Multi-start L-BFGS-B for Logarithmic Energy Minimization."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math

def _log_energy(pts):
    """Compute sum_{i<j} log(1/||x_i - x_j||) = -sum_{i<j} log(||x_i - x_j||)."""
    # pts shape (N, 3), assumed on unit sphere
    dists = cdist(pts, pts)
    # Take upper triangle
    iu = np.triu_indices(N, k=1)
    d = dists[iu]
    # Avoid log(0)
    d = np.maximum(d, 1e-15)
    return -np.sum(np.log(d))

def _project_to_sphere(pts):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def _energy_and_grad_vec(pts):
    """
    Compute energy and Riemannian gradient (projected to tangent space) vectorized.
    pts: (N, 3) array on unit sphere.
    """
    # Compute pairwise distances
    # dists[i, j] = ||x_i - x_j||
    dists = cdist(pts, pts)
    
    # Energy
    iu = np.triu_indices(N, k=1)
    d_upper = dists[iu]
    d_upper_safe = np.maximum(d_upper, 1e-15)
    energy = -np.sum(np.log(d_upper_safe))
    
    # Gradient calculation
    # grad_i = - sum_{j != i} (x_i - x_j) / ||x_i - x_j||^2
    # Let D be matrix of squared distances. D[i,j] = ||x_i - x_j||^2
    # diff[i,j] = x_i - x_j
    # We want sum over j of diff[i,j] / D[i,j]
    
    # diff shape (N, N, 3)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    # D shape (N, N)
    D = np.sum(diff**2, axis=2)
    
    # Avoid division by zero on diagonal
    D_safe = np.where(D > 1e-15, D, 1e-15)
    
    # weights[i,j] = 1 / D_safe[i,j]
    weights = 1.0 / D_safe
    
    # Set diagonal to 0
    np.fill_diagonal(weights, 0.0)
    
    # grad[i] = sum_j weights[i,j] * (x_i - x_j)
    # grad = weights @ diff ? No, weights is (N,N), diff is (N,N,3).
    # grad[i] = sum_j weights[i,j] * diff[i,j]
    # This is a matrix multiplication of weights (N,N) with diff (N,N,3) along axis 1?
    # Actually, np.einsum('ij,ijk->ik', weights, diff)
    grad_cart = np.einsum('ij,ijk->ik', weights, diff)
    
    # Note: The derivative of -log(d) w.r.t x_i is -(x_i - x_j)/d^2.
    # So grad_cart is correct for the Euclidean gradient.
    
    # Project to tangent space
    # grad_tan_i = grad_i - (grad_i . x_i) x_i
    grad_dot = np.sum(grad_cart * pts, axis=1, keepdims=True)
    grad_tan = grad_cart - grad_dot * pts
    
    return energy, grad_tan

def construct():
    """Construct 100 points on the unit sphere minimizing logarithmic energy."""
    
    def objective_and_grad(x_flat):
        pts = x_flat.reshape(N, 3)
        pts = _project_to_sphere(pts)
        energy, grad = _energy_and_grad_vec(pts)
        # For L-BFGS in Cartesian space, we need to map the Riemannian gradient back?
        # Actually, if we minimize f(x) where x is Cartesian, and we project x to sphere,
        # the gradient of the projected function w.r.t x is not simply the tangent gradient.
        # However, a common trick is to use the tangent gradient as the gradient for the Cartesian optimizer,
        # relying on the optimizer's line search to stay close to the manifold or just re-projecting.
        # A more robust method is to use the fact that for small steps, the change in energy
        # is well approximated by the inner product of the step and the tangent gradient.
        # Let's stick to the standard approach: minimize in Cartesian, project after each eval.
        # The gradient returned should be the gradient of the energy w.r.t the Cartesian coordinates.
        # If we define F(x) = E(proj(x)), then grad F(x) = J_proj^T grad E.
        # This is complex. 
        # Alternative: Use scipy's minimize with method='L-BFGS-B' but pass the tangent gradient 
        # and hope the optimizer works? No, that's mathematically unsound for Cartesian steps.
        
        # Better approach for this specific task: 
        # Use a custom wrapper that performs line search on the sphere.
        # Or, simpler: Use the Cartesian gradient derived from the chain rule.
        # grad_x F(x) = sum_k (d proj(x)_k / dx) * grad E_k
        # This is messy.
        
        # Let's use a different strategy: 
        # Minimize in spherical coordinates? No, singularities.
        
        # Let's use the "Projection Gradient" method manually? 
        # No, let's use the fact that L-BFGS-B is robust. 
        # We will define the objective as E(proj(x)) and use finite differences for gradient? 
        # No, too slow.
        
        # Let's use the analytic gradient for the Cartesian variables.
        # Let y = proj(x). y = x / ||x||.
        # E(y).
        # dE/dx = (dE/dy) @ (dy/dx).
        # dy_i/dx_j = (delta_ij - y_i y_j) / ||x||.
        # So dE/dx_j = (1/||x_j||) * sum_i (dE/dy_i)_j * (delta_ij - y_i y_j)? No.
        
        # Actually, there is a simpler way. 
        # If we stay on the sphere, the Riemannian gradient is sufficient for a Riemannian optimizer.
        # Since we are using a Cartesian optimizer, we can use the following approximation:
        # The gradient of the energy w.r.t Cartesian coordinates, when the point is on the sphere,
        # is the Riemannian gradient extended by zero in the normal direction? 
        # No. The gradient of the constraint is x. 
        # Lagrangian: L = E(x) + lambda (||x||^2 - 1).
        # grad L = grad E + 2 lambda x = 0.
        # So grad E = -2 lambda x. This means grad E is normal to the sphere? 
        # No, grad E is the Euclidean gradient. The condition for a minimum on the manifold
        # is that the tangential component of grad E is zero.
        # So, if we use a Cartesian optimizer, and we provide the tangential gradient as the gradient,
        # the optimizer will try to move in that direction. Since the direction is tangent,
        # it will move off the sphere. The projection brings it back.
        # This effectively performs a Riemannian gradient step with a retraction (projection).
        # This is a valid Riemannian optimization scheme (Gradient Descent with Retraction).
        # L-BFGS-B uses line search. If we provide the tangent gradient, the line search
        # will evaluate points off the sphere, project them, and compute energy.
        # This should work.
        
        return energy, grad_tan.flatten()

    # Multi-start strategy
    best_pts = None
    best_energy = np.inf
    
    # Start 1: Fibonacci
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    x0_fib = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    starts = [x0_fib]
    
    # Add random starts
    rng = np.random.default_rng(42)
    for _ in range(9):
        x0_rand = rng.normal(size=(N, 3))
        x0_rand = _project_to_sphere(x0_rand)
        starts.append(x0_rand)
        
    for x0 in starts:
        try:
            result = minimize(
                lambda x: _log_energy(_project_to_sphere(x.reshape(N, 3))),
                x0.flatten(),
                jac=lambda x: _energy_and_grad_vec(_project_to_sphere(x.reshape(N, 3)))[1].flatten(),
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-6, 'disp': False}
            )
            if result.fun < best_energy:
                best_energy = result.fun
                best_pts = _project_to_sphere(result.x.reshape(N, 3))
        except Exception:
            continue

    # Polish the best result
    if best_pts is not None:
        try:
            result = minimize(
                lambda x: _log_energy(_project_to_sphere(x.reshape(N, 3))),
                best_pts.flatten(),
                jac=lambda x: _energy_and_grad_vec(_project_to_sphere(x.reshape(N, 3)))[1].flatten(),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False}
            )
            if result.fun < best_energy:
                best_energy = result.fun
                best_pts = _project_to_sphere(result.x.reshape(N, 3))
        except Exception:
            pass

    # Final check
    if best_pts is None:
        # Fallback to Fibonacci
        best_pts = x0_fib
        best_energy = _log_energy(best_pts)
        
    # Ensure validity
    dists = cdist(best_pts, best_pts)
    iu = np.triu_indices(N, k=1)
    if np.min(dists[iu]) < 1e-9:
        best_pts = best_pts + rng.normal(0, 1e-6, best_pts.shape)
        best_pts = _project_to_sphere(best_pts)
        
    return best_pts
# EVOLVE-BLOCK-END
