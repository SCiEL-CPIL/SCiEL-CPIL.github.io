"""Rewritten program: Riemannian optimization on the sphere via quaternions."""
import numpy as np
from scipy.optimize import minimize
import math

N = 150


def construct():
    """
    Place N points on the unit sphere to minimize logarithmic energy
    sum_{i<j} log(1/|x_i - x_j|).
    
    Uses a Riemannian conjugate gradient approach on the sphere manifold.
    We parameterize each point as a unit vector and use geodesic updates.
    """
    # Initialize with Fibonacci spiral (good starting point)
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    x0 = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Ensure unit norm
    norms = np.linalg.norm(x0, axis=1, keepdims=True)
    x0 = x0 / norms
    
    def energy_and_grad(x):
        """
        Compute logarithmic energy and its gradient on the sphere.
        x: (N, 3) array of points on unit sphere
        Returns: (energy, grad) where grad is (N, 3)
        """
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
        n = x.shape[0]
        
        # Pairwise distances
        # d_ij = |x_i - x_j|
        # diff[i,j] = x_i - x_j
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        
        # Avoid division by zero for i=j
        np.fill_diagonal(dist, 1.0)
        
        # Energy: sum_{i<j} log(1/dist_ij) = -sum_{i<j} log(dist_ij)
        # Use upper triangle
        iu = np.triu_indices(n, k=1)
        dists = dist[iu]
        
        # Filter out very small distances to avoid log(0)
        valid = dists > 1e-15
        if not np.all(valid):
            dists = np.where(valid, dists, 1e-15)
        
        energy = -np.sum(np.log(dists))
        
        # Gradient computation
        # d/dx_i [ -log(|x_i - x_j|) ] = (x_i - x_j) / |x_i - x_j|^2
        # Total gradient for point i: sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
        # But we need to project onto the tangent space: grad_tangent = grad - (grad . x_i) * x_i
        
        grad = np.zeros_like(x)
        
        # For each pair (i,j), the contribution to grad[i] is (x_i - x_j)/d^2
        # and to grad[j] is (x_j - x_i)/d^2
        # Since we only sum over i<j, we add to both
        
        for idx in range(len(iu[0])):
            i_idx, j_idx = iu[0][idx], iu[1][idx]
            d = dists[idx]
            if d < 1e-15:
                continue
            diff_ij = x[i_idx] - x[j_idx]
            coeff = 1.0 / (d * d)
            grad[i_idx] += coeff * diff_ij
            grad[j_idx] -= coeff * diff_ij
        
        # Project gradient onto tangent space of sphere at each point
        # grad_proj = grad - (grad . x) * x
        dot_products = np.sum(grad * x, axis=1, keepdims=True)
        grad_tangent = grad - dot_products * x
        
        return energy, grad_tangent
    
    def riemannian_cg(x_init, max_iter=200, tol=1e-10):
        """
        Riemannian conjugate gradient method on the sphere.
        """
        x = x_init.copy()
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
        
        f, g = energy_and_grad(x)
        p = -g  # initial search direction (steepest descent)
        
        # Project p onto tangent space
        dot_p = np.sum(p * x, axis=1, keepdims=True)
        p = p - dot_p * x
        
        alpha = 1.0  # initial step size
        best_f = f
        best_x = x.copy()
        
        for it in range(max_iter):
            # Geodesic line search
            # Move along geodesic: x_new = x * cos(alpha*|p|) + (p/|p|) * sin(alpha*|p|)
            p_norm = np.linalg.norm(p, axis=1)
            p_unit = p / np.maximum(p_norm, 1e-15)[:, np.newaxis]
            
            # Try different step sizes
            alphas = [alpha * 0.5, alpha * 0.75, alpha, alpha * 1.25, alpha * 1.5]
            best_alpha = alpha
            best_trial_f = np.inf
            best_trial_x = None
            
            for a in alphas:
                if a <= 0:
                    continue
                # Geodesic update
                theta = a * p_norm
                cos_t = np.cos(theta)
                sin_t = np.sin(theta)
                x_trial = x * cos_t + p_unit * sin_t
                # Normalize
                x_trial = x_trial / np.linalg.norm(x_trial, axis=1, keepdims=True)
                
                f_trial, g_trial = energy_and_grad(x_trial)
                
                if f_trial < best_trial_f:
                    best_trial_f = f_trial
                    best_trial_x = x_trial.copy()
                    best_trial_g = g_trial.copy()
                    best_alpha = a
            
            if best_trial_x is None:
                break
            
            if best_trial_f < best_f:
                best_f = best_trial_f
                best_x = best_trial_x.copy()
            
            # Check convergence
            if abs(f - best_trial_f) < tol and np.linalg.norm(best_trial_g) < 1e-8:
                x = best_trial_x
                break
            
            # Update
            x = best_trial_x
            g_new = best_trial_g
            f_new = best_trial_f
            
            # Polak-Ribiere beta
            grad_diff = np.sum(g_new * (-g), axis=1)
            beta = np.maximum(0, grad_diff / (np.sum(g * g, axis=1) + 1e-15))
            beta = np.clip(beta, 0, 1)
            
            # New search direction
            p_new = -g_new + beta[:, np.newaxis] * p
            
            # Project onto tangent space
            dot_p_new = np.sum(p_new * x, axis=1, keepdims=True)
            p_new = p_new - dot_p_new * x
            
            # If the new direction doesn't decrease, restart
            if np.sum(p_new * g_new) > 0:
                p_new = -g_new
                dot_p_new = np.sum(p_new * x, axis=1, keepdims=True)
                p_new = p_new - dot_p_new * x
            
            p = p_new
            g = g_new
            f = f_new
            
            # Adaptive step size
            alpha = best_alpha
            if best_trial_f > f:
                alpha = alpha * 0.5
            
            if it % 50 == 0 and it > 0:
                # Occasionally restart from best_x if stuck
                if f > best_f + 1e-6:
                    x = best_x.copy()
                    f, g = energy_and_grad(x)
                    p = -g
                    dot_p = np.sum(p * x, axis=1, keepdims=True)
                    p = p - dot_p * x
                    alpha = 0.1
        
        return best_x
    
    # Run optimization
    x_opt = riemannian_cg(x0, max_iter=100, tol=1e-10)
    
    # Final normalization
    x_opt = x_opt / np.linalg.norm(x_opt, axis=1, keepdims=True)
    
    return x_opt
