import numpy as np
import time

N = 150

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # 5s safety margin
    
    # Initial points: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    x = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Function to compute energy and gradient
    def energy_and_grad(x):
        # x is (N, 3)
        # Compute pairwise distances squared
        gram = x @ x.T
        dist2 = 2.0 - 2.0 * gram
        # Ensure diagonal is 1 to avoid issues, though we mask it
        np.fill_diagonal(dist2, 1.0)
        
        # W[k,j] = 1/dist2[k,j]
        W = 1.0 / dist2
        np.fill_diagonal(W, 0.0)
        
        # Gradient of E is sum_{j!=k} (x_j - x_k) / dist2[k,j]
        # We compute repulsion vector R = sum_{j!=k} (x_k - x_j) / dist2[k,j] = -grad_E
        row_sums = np.sum(W, axis=1)
        repulsion = x * row_sums[:, np.newaxis] - (x @ W)
        
        # Project repulsion onto tangent space
        # rep_tan = rep - (rep . x) x
        rep_dot_x = np.sum(repulsion * x, axis=1)
        rep_tan = repulsion - rep_dot_x[:, np.newaxis] * x
        
        # Energy
        # E = -0.25 * sum_{i != j} log(dist2_ij)
        log_dist2 = np.log(dist2)
        np.fill_diagonal(log_dist2, 0.0)
        energy = -0.25 * np.sum(log_dist2)
        
        return energy, rep_tan

    current_energy, _ = energy_and_grad(x)
    best_x = x.copy()
    best_energy = current_energy
    
    # Optimization loop
    step_size = 0.1
    no_improve_count = 0
    max_no_improve = 100
    
    while time.time() < deadline:
        energy, rep_tan = energy_and_grad(x)
        
        if energy < best_energy:
            best_energy = energy
            best_x = x.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                break
        
        # Backtracking line search
        # We want to minimize E, so we move in direction of -grad_E = rep_tan
        # new_x = x + alpha * rep_tan
        alpha = step_size
        improved = False
        for _ in range(20):
            new_x = x + alpha * rep_tan
            # Normalize
            norms = np.linalg.norm(new_x, axis=1)
            norms[norms < 1e-10] = 1.0
            new_x = new_x / norms[:, np.newaxis]
            
            new_energy, _ = energy_and_grad(new_x)
            
            if new_energy < energy:
                x = new_x
                improved = True
                break
            alpha *= 0.5
        
        if not improved:
            break
            
        # Adapt step size
        if alpha < step_size * 0.8:
            step_size = alpha
        elif alpha == step_size:
            step_size = min(step_size * 1.1, 1.0)

    # Final check
    norms = np.linalg.norm(best_x, axis=1)
    best_x = best_x / norms[:, np.newaxis]
    
    return best_x
