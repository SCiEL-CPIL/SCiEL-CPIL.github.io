import numpy as np
import time
import math
from scipy.optimize import minimize

N = 100

def construct():
    """
    Minimizes the logarithmic energy of N points on the unit sphere.
    Combines L-BFGS-B with penalty for coarse search and Riemannian 
    Gradient Descent for precise manifold-constrained refinement.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # 1. Initialization: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    X_init = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Helper functions for Energy and Gradient
    
    def compute_energy(pts):
        """Compute logarithmic energy: sum_{i<j} log(1/|xi-xj|)"""
        n = pts.shape[0]
        # Use dot product for efficiency
        dots = pts @ pts.T
        d2 = 2.0 - 2.0 * dots
        np.fill_diagonal(d2, 1.0)
        d2 = np.maximum(d2, 1e-15)
        dist = np.sqrt(d2)
        iu = np.triu_indices(n, k=1)
        log_d = np.log(dist[iu])
        return -math.fsum(log_d)

    def compute_euclidean_gradient(pts):
        """
        Compute the Euclidean gradient of the energy.
        grad[i] = - sum_{j!=i} (pts[i] - pts[j]) / ||pts[i] - pts[j]||^2
        """
        n = pts.shape[0]
        dots = pts @ pts.T
        d2 = 2.0 - 2.0 * dots
        np.fill_diagonal(d2, 1.0)
        d2 = np.maximum(d2, 1e-15)
        
        inv_d2 = 1.0 / d2
        np.fill_diagonal(inv_d2, 0.0)
        
        # grad_i = - sum_j (x_i - x_j) / d2_ij
        # = - sum_j x_i / d2_ij + sum_j x_j / d2_ij
        # = - x_i * sum_j (1/d2_ij) + sum_j (x_j / d2_ij)
        
        sum_inv_d2 = inv_d2.sum(axis=1)
        # (pts.T @ inv_d2).T gives sum_j x_j * inv_d2[j, i] ? 
        # inv_d2 is symmetric. 
        # We want sum_j (x_j * inv_d2[i, j]) for each i.
        # This is (inv_d2 @ pts).T ? No.
        # Let M = inv_d2. We want vector v_i = sum_j M[i,j] x_j.
        # This is (M @ pts).T? No, M is (N,N), pts is (N,3).
        # (M @ pts) is (N,3). Row i is sum_j M[i,j] pts[j].
        # So sum_xj_inv_d2 = inv_d2 @ pts.
        
        sum_xj_inv_d2 = inv_d2 @ pts
        
        grad = -(pts * sum_inv_d2[:, np.newaxis] - sum_xj_inv_d2)
        
        return grad

    def compute_tangent_gradient(pts):
        """
        Project the Euclidean gradient onto the tangent space of the sphere.
        grad_tan = grad - (grad . x) x
        """
        grad_eu = compute_euclidean_gradient(pts)
        # Project out radial component
        grad_dot_xi = np.sum(grad_eu * pts, axis=1, keepdims=True)
        grad_tan = grad_eu - grad_dot_xi * pts
        return grad_tan

    def project_to_sphere(pts):
        """Normalize points to unit sphere."""
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-15)
        return pts / norms

    # --- Stage 1: Coarse Optimization with L-BFGS-B and Penalty ---
    # We use a penalty to keep points on the sphere during the L-BFGS-B phase.
    # This is faster than pure Riemannian methods for large initial steps.
    
    lambda_pen = 100.0
    
    def objective_penalty(x_flat):
        X = x_flat.reshape(N, 3)
        E_log = compute_energy(X)
        norms_sq = np.sum(X * X, axis=1)
        E_pen = lambda_pen * np.sum((norms_sq - 1.0) ** 2)
        return E_log + E_pen

    def gradient_penalty(x_flat):
        X = x_flat.reshape(N, 3)
        grad_log = compute_euclidean_gradient(X)
        
        # Gradient of penalty: 4 * lambda * (||x||^2 - 1) * x
        norms_sq = np.sum(X * X, axis=1)
        grad_pen = 4.0 * lambda_pen * (norms_sq - 1.0)[:, np.newaxis] * X
        
        return (grad_log + grad_pen).flatten()

    x0 = X_init.flatten()
    best_x = x0.copy()
    best_val = objective_penalty(x0)
    
    try:
        # L-BFGS-B is good for this size
        res = minimize(
            objective_penalty, 
            x0, 
            jac=gradient_penalty,
            method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-6, 'disp': False}
        )
        
        if res.fun < best_val:
            best_x = res.x
            best_val = res.fun
            
    except Exception:
        pass

    # Project to sphere to start Stage 2
    X_best = best_x.reshape(N, 3)
    X_best = project_to_sphere(X_best)
    
    # --- Stage 2: Riemannian Gradient Descent ---
    # This phase strictly maintains points on the sphere and uses the correct
    # manifold geometry for convergence.
    
    def riemannian_descent(pts, max_iter=2000, tol=1e-10, deadline=None):
        pts = project_to_sphere(pts.copy())
        f = compute_energy(pts)
        
        # Adaptive step size
        step = 1.0
        
        for it in range(max_iter):
            if deadline is not None and time.time() > deadline:
                break
                
            grad_tan = compute_tangent_gradient(pts)
            grad_norm = np.linalg.norm(grad_tan)
            
            if grad_norm < tol:
                break
            
            # Backtracking line search
            accepted = False
            current_step = step
            for ls_iter in range(30):
                # Riemannian step: move in tangent direction, then project
                # Using exponential map would be more accurate, but projection
                # is a valid retraction and often sufficient.
                new_pts = pts - current_step * grad_tan
                new_pts = project_to_sphere(new_pts)
                
                new_f = compute_energy(new_pts)
                
                if np.isfinite(new_f) and new_f < f:
                    # Armijo condition: sufficient decrease
                    if new_f < f - 1e-4 * current_step * grad_norm * grad_norm:
                        accepted = True
                        break
                    # If not Armijo, accept if any decrease to ensure progress
                    if new_f < f - 1e-12:
                        accepted = True
                        break
                
                current_step *= 0.7
            
            if accepted:
                pts = new_pts
                f = new_f
                # Increase step size slightly if we didn't backtrack much
                if ls_iter == 0:
                    step = min(step * 1.05, 10.0)
                else:
                    step = current_step / 0.7 # Recover from backtrack
            else:
                break
                
        return pts, f

    if time.time() < deadline - 10.0:
        X_best, e_best = riemannian_descent(X_best, max_iter=1000, tol=1e-10, deadline=deadline - 10.0)
    
    # --- Stage 3: High-Precision Polish ---
    # Use L-BFGS-B again but with a very tight tolerance and a higher penalty
    # to ensure we are at a very precise local minimum.
    
    if time.time() < deadline - 5.0:
        lambda_polish = 1000.0
        
        def objective_polish(x_flat):
            X = x_flat.reshape(N, 3)
            E_log = compute_energy(X)
            norms_sq = np.sum(X * X, axis=1)
            E_pen = lambda_polish * np.sum((norms_sq - 1.0) ** 2)
            return E_log + E_pen

        def gradient_polish(x_flat):
            X = x_flat.reshape(N, 3)
            grad_log = compute_euclidean_gradient(X)
            norms_sq = np.sum(X * X, axis=1)
            grad_pen = 4.0 * lambda_polish * (norms_sq - 1.0)[:, np.newaxis] * X
            return (grad_log + grad_pen).flatten()

        x_polish = X_best.flatten()
        try:
            res2 = minimize(
                objective_polish,
                x_polish,
                jac=gradient_polish,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'disp': False}
            )
            if res2.fun < compute_energy(X_best):
                X_best = res2.x.reshape(N, 3)
        except Exception:
            pass

    # Final projection
    X_final = project_to_sphere(X_best)
    
    return X_final
