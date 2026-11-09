# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import math

def construct():
    def _run_optimization(x0):
        # 2. Define Objective Function and Gradient
        lambda_pen = 10000.0
        
        def energy_and_grad(x_flat):
            x = x_flat.reshape(N, 3)
            
            # Compute pairwise distances
            diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
            dist2 = np.sum(diff ** 2, axis=2)
            
            # Mask diagonal
            mask = ~np.eye(N, dtype=bool)
            dist2_safe = np.where(mask, dist2, 1.0)
            
            # Log energy
            # E = - sum_{i<j} log(||x_i - x_j||)
            # E = -0.5 * sum_{i != j} log(sqrt(dist2)) = -0.25 * sum_{i != j} log(dist2)
            log_dist2 = np.log(dist2_safe)
            e_log = -0.25 * np.sum(log_dist2[mask])
            
            # Penalty
            norms2 = np.sum(x ** 2, axis=1)
            pen = lambda_pen * np.sum((norms2 - 1.0) ** 2)
            
            total_energy = e_log + pen
            
            # Gradient
            # dE/dx_k = - sum_{j != k} (x_k - x_j) / dist2_kj
            grad_log = diff / dist2_safe[:, :, np.newaxis]
            grad_log = np.sum(grad_log, axis=1)
            
            # Correct factor is -1.0, not -0.5
            grad_e = -1.0 * grad_log
            
            # Grad of penalty
            grad_pen = 4.0 * lambda_pen * (norms2 - 1.0)[:, np.newaxis] * x
            
            total_grad = grad_e + grad_pen
            
            return total_energy, total_grad.flatten()

        # 3. Optimization
        res = minimize(
            energy_and_grad, 
            x0.flatten(), 
            method='L-BFGS-B', 
            jac=True, 
            options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-10}
        )
        
        x_opt = res.x.reshape(N, 3)
        
        # 4. Final Projection
        norms = np.linalg.norm(x_opt, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        x_final = x_opt / norms
        
        # Calculate final energy for comparison
        # Use math.fsum for the final score calculation to match evaluator?
        # The optimizer minimized the np.sum version.
        # We return the points. The evaluator calculates the score.
        # We just need to return the best points.
        return x_final, res.fun

    best_x = None
    best_energy = float('inf')
    
    # Multi-start with 5 different seeds
    for seed in range(5):
        np.random.seed(seed)
        # 1. Initial Configuration: Fibonacci Spiral
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        x0 = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        
        # Add noise
        x0 = x0 + 0.001 * np.random.randn(N, 3)
        norms = np.linalg.norm(x0, axis=1, keepdims=True)
        x0 = x0 / norms
        
        x_res, e_res = _run_optimization(x0)
        
        if e_res < best_energy:
            best_energy = e_res
            best_x = x_res
            
    return best_x

# EVOLVE-BLOCK-END
