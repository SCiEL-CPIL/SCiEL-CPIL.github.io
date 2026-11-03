"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def construct():
    # Energy function and gradient on the sphere (using retraction approach)
    # We optimize in R^3 but project back to sphere after each gradient step
    # This is more efficient than penalty methods
    
    def energy_and_grad(x):
        """Compute energy and gradient, with x on sphere (approximately)."""
        # Ensure x is on sphere
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
        
        # Pairwise differences
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        np.fill_diagonal(dist_sq, 1.0)  # Avoid zero division
        
        # Energy: sum_{i<j} log(1/dist) = -sum_{i<j} log(dist)
        # = -0.5 * sum_{i<j} log(dist_sq)
        triu_idx = np.triu_indices(N, 1)
        energy = -0.5 * np.sum(np.log(dist_sq[triu_idx]))
        
        # Gradient: dE/dx_i = sum_{j!=i} (x_i - x_j) / dist_sq_ij
        # Note: this is the gradient in R^3 space
        inv_dist_sq = 1.0 / dist_sq
        np.fill_diagonal(inv_dist_sq, 0.0)
        
        # grad[i] = sum_j diff[i,j] * inv_dist_sq[i,j]
        grad = np.einsum('ijk,ij->ik', diff, inv_dist_sq)
        
        return energy, grad
    
    def project_to_sphere(x):
        """Project points back to unit sphere."""
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-15)
        return x / norms
    
    # Generate multiple initial configurations and optimize each
    best_energy = np.inf
    best_x = None
    
    # Different initialization strategies
    init_configs = []
    
    # 1. Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    fib = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    init_configs.append(fib)
    
    # 2. Random points on sphere
    rng = np.random.default_rng(42)
    for seed in [1, 2, 3]:
        rng_local = np.random.default_rng(seed)
        pts = rng_local.normal(size=(N, 3))
        pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        init_configs.append(pts)
    
    # 3. Fibonacci with jitter
    for jitter_scale in [0.05, 0.1]:
        rng_local = np.random.default_rng(100 + int(jitter_scale * 100))
        jittered = fib + rng_local.normal(0, jitter_scale, fib.shape)
        jittered = jittered / np.linalg.norm(jittered, axis=1, keepdims=True)
        init_configs.append(jittered)
    
    # Optimize each configuration
    for init_idx, x0 in enumerate(init_configs):
        x0 = project_to_sphere(x0)
        
        # Use L-BFGS-B with a wrapper that projects back to sphere
        # We'll use a custom approach: optimize in R^3 but constrain via projection
        
        def objective_with_projection(x_flat):
            x = x_flat.reshape(N, 3)
            x = project_to_sphere(x)
            e, _ = energy_and_grad(x)
            return e
        
        def gradient_with_projection(x_flat):
            x = x_flat.reshape(N, 3)
            x = project_to_sphere(x)
            e, g = energy_and_grad(x)
            # Project gradient to tangent space of sphere
            # Tangent space: vectors perpendicular to x
            # g_tangent = g - (g · x) * x
            g_dot_x = np.sum(g * x, axis=1, keepdims=True)
            g_tangent = g - g_dot_x * x
            return g_tangent.flatten()
        
        try:
            res = minimize(
                objective_with_projection,
                x0.flatten(),
                jac=gradient_with_projection,
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            
            x_opt = project_to_sphere(res.x.reshape(N, 3))
            e_final, _ = energy_and_grad(x_opt)
            
            if e_final < best_energy:
                best_energy = e_final
                best_x = x_opt
        except Exception:
            continue
    
    # Final refinement: run a few more iterations from the best solution
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
                options={'maxiter': 5000, 'ftol': 1e-16, 'gtol': 1e-12}
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
        # Add small perturbation if points are too close
        rng = np.random.default_rng(123)
        best_x = best_x + rng.normal(0, 1e-6, best_x.shape)
        best_x = project_to_sphere(best_x)
    
    return best_x
# EVOLVE-BLOCK-END
