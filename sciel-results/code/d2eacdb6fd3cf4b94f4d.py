"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import math

def construct():
    # Energy function and gradient on the sphere
    def energy_and_grad(x):
        """Compute energy and gradient, with x on sphere."""
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
        
        # Pairwise differences
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        np.fill_diagonal(dist_sq, 1.0)
        
        # Energy: sum_{i<j} log(1/dist)
        triu_idx = np.triu_indices(N, 1)
        energy = -0.5 * np.sum(np.log(dist_sq[triu_idx]))
        
        # Gradient: dE/dx_i = sum_{j!=i} (x_i - x_j) / dist_sq_ij
        inv_dist_sq = 1.0 / dist_sq
        np.fill_diagonal(inv_dist_sq, 0.0)
        
        grad = np.einsum('ijk,ij->ik', diff, inv_dist_sq)
        return energy, grad
    
    def project_to_sphere(x):
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-15)
        return x / norms

    def md_relax(x0, steps=200, initial_temp=0.05, cooling=0.99):
        """Molecular Dynamics relaxation with Velocity Verlet."""
        x = x0.copy()
        v = np.zeros_like(x)
        temp = initial_temp
        
        for step in range(steps):
            # Calculate forces (negative gradient)
            _, grad = energy_and_grad(x)
            # Project gradient to tangent space
            g_dot_x = np.sum(grad * x, axis=1, keepdims=True)
            grad_tangent = grad - g_dot_x * x
            force = -grad_tangent
            
            # Velocity Verlet
            v += force * 0.5
            x = x + v
            x = project_to_sphere(x)
            
            _, grad_new = energy_and_grad(x)
            g_dot_x_new = np.sum(grad_new * x, axis=1, keepdims=True)
            grad_tangent_new = grad_new - g_dot_x_new * x
            force_new = -grad_tangent_new
            
            v = (v + force_new) * 0.5
            # Cooling
            v *= cooling
            temp *= cooling
            
            # Optional: random kicks if temp is high enough to escape minima
            if step < steps // 2:
                v += np.random.normal(0, temp * 0.1, x.shape)
                x = project_to_sphere(x)
        
        return x

    best_energy = np.inf
    best_x = None
    
    # Initialization 1: Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    fib = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Initialization 2: Random
    rng = np.random.default_rng(42)
    rand_pts = rng.normal(size=(N, 3))
    rand_pts = rand_pts / np.linalg.norm(rand_pts, axis=1, keepdims=True)
    
    inits = [fib, rand_pts]
    
    for x0 in inits:
        # Stage 1: MD Relaxation to get out of obvious local minima
        x_md = md_relax(x0, steps=300, initial_temp=0.02, cooling=0.98)
        
        # Stage 2: L-BFGS Polishing
        def objective(x_flat):
            x = x_flat.reshape(N, 3)
            x = project_to_sphere(x)
            e, _ = energy_and_grad(x)
            return e
        
        def gradient(x_flat):
            x = x_flat.reshape(N, 3)
            x = project_to_sphere(x)
            e, g = energy_and_grad(x)
            g_dot_x = np.sum(g * x, axis=1, keepdims=True)
            g_tangent = g - g_dot_x * x
            return g_tangent.flatten()
        
        try:
            res = minimize(
                objective,
                x_md.flatten(),
                jac=gradient,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            x_opt = project_to_sphere(res.x.reshape(N, 3))
            e_final, _ = energy_and_grad(x_opt)
            
            if e_final < best_energy:
                best_energy = e_final
                best_x = x_opt
        except Exception:
            pass

    # Final refinement from the best solution found
    if best_x is not None:
        def objective_final(x_flat):
            x = x_flat.reshape(N, 3)
            x = project_to_sphere(x)
            e, _ = energy_and_grad(x)
            return e
        
        def gradient_final(x_flat):
            x = x_flat.reshape(N, 3)
            x = project_to_sphere(x)
            e, g = energy_and_grad(x)
            g_dot_x = np.sum(g * x, axis=1, keepdims=True)
            g_tangent = g - g_dot_x * x
            return g_tangent.flatten()
        
        try:
            res_final = minimize(
                objective_final,
                best_x.flatten(),
                jac=gradient_final,
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            best_x = project_to_sphere(res_final.x.reshape(N, 3))
        except Exception:
            pass
    
    # Ensure output is valid
    best_x = project_to_sphere(best_x)
    
    # Verify no points are too close
    diff = best_x[:, np.newaxis, :] - best_x[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    min_dist = np.sqrt(np.min(dist_sq))
    
    if min_dist < 1e-9:
        rng = np.random.default_rng(123)
        best_x = best_x + rng.normal(0, 1e-6, best_x.shape)
        best_x = project_to_sphere(best_x)
        
    return best_x
# EVOLVE-BLOCK-END
