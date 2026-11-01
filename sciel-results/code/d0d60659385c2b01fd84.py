"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import math
import random

def construct():
    # Initialize with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    x = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    def calc_energy(x):
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, 1.0)
        iu = np.triu_indices(N, k=1)
        return -np.sum(np.log(dist[iu]))
        
    current_energy = calc_energy(x)
    
    # Simulated Annealing Parameters
    T_start = 10.0
    T_end = 1e-6
    num_steps = 200000
    # Geometric cooling
    # T(t) = T_start * (T_end/T_start)^(t/num_steps)
    cooling_factor = (T_end / T_start) ** (1.0 / num_steps)
    
    T = T_start
    step_size = 0.2
    
    for step in range(num_steps):
        k = random.randint(0, N - 1)
        
        # Generate random tangent vector
        rand_vec = np.random.randn(3)
        dot = np.dot(rand_vec, x[k])
        tangent = rand_vec - dot * x[k]
        norm_t = np.linalg.norm(tangent)
        if norm_t < 1e-10:
            continue
        tangent /= norm_t
        
        x_new_k = x[k] + step_size * tangent
        x_new_k = x_new_k / np.linalg.norm(x_new_k)
        
        # Calculate energy change
        diff_old = x[k] - x
        d_old = np.linalg.norm(diff_old, axis=1)
        d_old[k] = 1.0
        old_contrib = -np.sum(np.log(d_old))
        
        diff_new = x_new_k - x
        d_new = np.linalg.norm(diff_new, axis=1)
        d_new[k] = 1.0
        new_contrib = -np.sum(np.log(d_new))
        
        delta_e = new_contrib - old_contrib
        
        if delta_e < 0 or random.random() < math.exp(-delta_e / T):
            x[k] = x_new_k
            current_energy += delta_e
            
        T *= cooling_factor
        
    # Final polish with L-BFGS-B
    def energy_and_grad(x_flat):
        x = x_flat.reshape(N, 3)
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        x = x / norms
        
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, 1.0)
        
        iu = np.triu_indices(N, k=1)
        log_dists = np.log(dist[iu])
        energy = -np.sum(log_dists)
        
        # Vectorized gradient
        inv_dist_sq = 1.0 / (dist * dist)
        np.fill_diagonal(inv_dist_sq, 0.0)
        
        sum_inv = np.sum(inv_dist_sq, axis=1)
        weighted_sum = inv_dist_sq @ x
        
        grad = - ( x * sum_inv[:, np.newaxis] - weighted_sum )
        
        # Project to tangent space
        for k in range(N):
            dot = np.dot(grad[k], x[k])
            grad[k] = grad[k] - dot * x[k]
            
        return energy, grad.flatten()

    res = minimize(
        lambda x: energy_and_grad(x)[0],
        x.flatten(),
        jac=lambda x: energy_and_grad(x)[1],
        method='L-BFGS-B',
        options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-10}
    )
    
    x_opt = res.x.reshape(N, 3)
    norms = np.linalg.norm(x_opt, axis=1, keepdims=True)
    x_opt = x_opt / norms
    
    return x_opt
# EVOLVE-BLOCK-END
