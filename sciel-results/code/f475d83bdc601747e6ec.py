"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math
from scipy.optimize import minimize

N = 150

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    best_energy = float('inf')
    best_points = None
    
    def energy_and_grad(x_flat):
        """
        Computes the logarithmic energy and Euclidean gradient for L-BFGS-B.
        We work in R^3N and project the gradient to the tangent space.
        """
        x = x_flat.reshape(N, 3)
        # Normalize to handle any drift
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        safe_norms = np.where(norms < 1e-12, 1.0, norms)
        x = x / safe_norms
        
        # Pairwise squared distances using dot products
        dot = x @ x.T
        dist_sq = 2.0 - 2.0 * dot
        np.fill_diagonal(dist_sq, 1.0)
        dist = np.sqrt(dist_sq)
        
        # Gradient of energy w.r.t. x_i
        # E = -sum_{i<j} log(d_ij)
        # dE/dx_i = -sum_{j!=i} (1/d_ij) * (x_i - x_j)/d_ij = -sum_{j!=i} (x_i - x_j)/d_ij^2
        inv_dist_sq = 1.0 / dist_sq
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        F = diff * inv_dist_sq[:, :, np.newaxis]
        grad_E = -np.sum(F, axis=1)
        
        # Project to tangent space
        dot_prod = np.sum(grad_E * x, axis=1, keepdims=True)
        grad_tangent = grad_E - dot_prod * x
        
        # Energy
        mask = ~np.eye(N, dtype=bool)
        log_d = np.log(dist)
        log_d[~mask] = 0.0
        energy = -0.5 * np.sum(log_d)
        
        return energy, grad_tangent.flatten()

    def energy_only(x_flat):
        """Energy only for line search."""
        e, _ = energy_and_grad(x_flat)
        return e

    def project_to_sphere(x_flat):
        """Project points back to unit sphere."""
        x = x_flat.reshape(N, 3)
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        safe_norms = np.where(norms < 1e-12, 1.0, norms)
        x = x / safe_norms
        return x.flatten()

    def riemannian_gd_polish(x_init, max_iters=500, lr_init=0.1):
        """
        Riemannian gradient descent with backtracking line search.
        Uses exponential map (retraction) on the sphere.
        """
        x = x_init.copy()
        lr = lr_init
        prev_energy = energy_only(x.flatten())
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            _, grad_flat = energy_and_grad(x.flatten())
            grad = grad_flat.reshape(N, 3)
            
            # Project gradient to tangent space (should already be tangent, but ensure)
            dot_prod = np.sum(grad * x, axis=1, keepdims=True)
            grad_tangent = grad - dot_prod * x
            
            # Check convergence
            grad_norm = np.linalg.norm(grad_tangent)
            if grad_norm < 1e-14:
                break
            
            # Backtracking line search
            alpha = lr
            x_new = None
            for _ in range(30):
                # Retraction: x_new = normalize(x - alpha * grad_tangent)
                step = x - alpha * grad_tangent
                norms = np.linalg.norm(step, axis=1, keepdims=True)
                safe_norms = np.where(norms < 1e-12, 1.0, norms)
                x_candidate = step / safe_norms
                
                e_candidate = energy_only(x_candidate.flatten())
                
                if e_candidate < prev_energy - 1e-15:
                    x_new = x_candidate
                    break
                alpha *= 0.5
            
            if x_new is None:
                # No improvement found, reduce learning rate
                lr *= 0.5
                if lr < 1e-15:
                    break
                continue
            
            x = x_new
            prev_energy = energy_only(x.flatten())
        
        return x

    def optimize_from(x_init):
        """Run L-BFGS-B optimization from initial point."""
        x_flat = x_init.flatten()
        
        def fun(x_flat):
            return energy_only(x_flat)
        
        def jac(x_flat):
            _, g = energy_and_grad(x_flat)
            return g
        
        # Use L-BFGS-B with tight tolerances
        result = minimize(
            fun, x_flat, jac=jac, method='L-BFGS-B',
            options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-10, 'maxfun': 10000}
        )
        
        x_opt = project_to_sphere(result.x)
        E_opt = energy_only(x_opt)
        
        return x_opt.reshape(N, 3), E_opt

    def polish(x_init):
        """Final high-precision polish with very tight tolerances."""
        x_flat = x_init.flatten()
        
        def fun(x_flat):
            return energy_only(x_flat)
        
        def jac(x_flat):
            _, g = energy_and_grad(x_flat)
            return g
        
        result = minimize(
            fun, x_flat, jac=jac, method='L-BFGS-B',
            options={'maxiter': 10000, 'ftol': 1e-18, 'gtol': 1e-12, 'maxfun': 20000}
        )
        
        x_opt = project_to_sphere(result.x)
        E_opt = energy_only(x_opt)
        
        return x_opt.reshape(N, 3), E_opt

    # Generate diverse initializations
    initializations = []
    
    # 1. Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    x_fib = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    initializations.append(('fib', x_fib))
    
    # 2. Perturbed Fibonacci (small noise)
    np.random.seed(42)
    noise = np.random.normal(0, 0.03, (N, 3))
    x_pert1 = x_fib + noise
    x_pert1 = x_pert1 / np.linalg.norm(x_pert1, axis=1, keepdims=True)
    initializations.append(('pert1', x_pert1))
    
    # 3. Perturbed Fibonacci (medium noise)
    np.random.seed(999)
    noise = np.random.normal(0, 0.08, (N, 3))
    x_pert2 = x_fib + noise
    x_pert2 = x_pert2 / np.linalg.norm(x_pert2, axis=1, keepdims=True)
    initializations.append(('pert2', x_pert2))
    
    # 4. Random points
    np.random.seed(123)
    x_rand = np.random.normal(0, 1, (N, 3))
    x_rand = x_rand / np.linalg.norm(x_rand, axis=1, keepdims=True)
    initializations.append(('rand', x_rand))
    
    # 5. Another random with different seed
    np.random.seed(777)
    x_rand2 = np.random.normal(0, 1, (N, 3))
    x_rand2 = x_rand2 / np.linalg.norm(x_rand2, axis=1, keepdims=True)
    initializations.append(('rand2', x_rand2))
    
    # 6. Fibonacci with different golden ratio approximation
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i + 0.5  # phase shift
    r = np.sqrt(1.0 - z * z)
    x_fib2 = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    initializations.append(('fib2', x_fib2))
    
    # 7. More diverse random initializations
    for seed in [2024, 3141, 2718, 1618]:
        np.random.seed(seed)
        x_r = np.random.normal(0, 1, (N, 3))
        x_r = x_r / np.linalg.norm(x_r, axis=1, keepdims=True)
        initializations.append((f'rand_{seed}', x_r))
    
    # 8. More perturbed Fibonacci variants
    for noise_scale, seed in [(0.01, 555), (0.05, 666), (0.12, 777)]:
        np.random.seed(seed)
        noise = np.random.normal(0, noise_scale, (N, 3))
        x_p = x_fib + noise
        x_p = x_p / np.linalg.norm(x_p, axis=1, keepdims=True)
        initializations.append((f'pert_{seed}', x_p))
    
    # Optimize each initialization
    for name, x_init in initializations:
        if time.time() > deadline - 20:
            break
        
        try:
            x_opt, E_opt = optimize_from(x_init)
            if E_opt < best_energy:
                best_energy = E_opt
                best_points = x_opt.copy()
        except Exception:
            continue
    
    # Riemannian GD polish on the best result
    if best_points is not None and time.time() < deadline - 10:
        try:
            x_gd = riemannian_gd_polish(best_points)
            E_gd = energy_only(x_gd.flatten())
            if E_gd < best_energy:
                best_energy = E_gd
                best_points = x_gd.copy()
        except Exception:
            pass
    
    # Final high-precision polish on the best result
    if best_points is not None and time.time() < deadline - 5:
        try:
            x_opt, E_opt = polish(best_points)
            if E_opt < best_energy:
                best_energy = E_opt
                best_points = x_opt.copy()
        except Exception:
            pass
    
    # One more Riemannian GD pass after L-BFGS polish
    if best_points is not None and time.time() < deadline - 2:
        try:
            x_gd2 = riemannian_gd_polish(best_points)
            E_gd2 = energy_only(x_gd2.flatten())
            if E_gd2 < best_energy:
                best_energy = E_gd2
                best_points = x_gd2.copy()
        except Exception:
            pass
    
    # Ensure valid output
    if best_points is None:
        # Fallback to Fibonacci
        best_points = x_fib
    
    # Final normalization
    best_points = best_points / np.linalg.norm(best_points, axis=1, keepdims=True)
    
    return best_points

# EVOLVE-BLOCK-END
