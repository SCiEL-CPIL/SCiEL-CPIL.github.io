import numpy as np
import time
from scipy.optimize import minimize

N = 130

def _fibonacci_sphere(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _compute_covering_radius(P, Y):
    """Compute covering radius using test points Y."""
    dots = Y @ P.T
    dots = np.clip(dots, -1.0, 1.0)
    D = np.sqrt(np.clip(2.0 - 2.0 * dots, 0.0, 4.0))
    min_D = np.min(D, axis=1)
    return np.max(min_D)

def _potential_objective(params, Y, temperature):
    """Soft-max based potential field objective."""
    P = params.reshape(-1, 3)
    # Normalize to unit sphere
    norms = np.linalg.norm(P, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    P = P / norms
    
    # Compute distances from Y to P
    dots = Y @ P.T
    dots = np.clip(dots, -1.0, 1.0)
    D = np.sqrt(np.clip(2.0 - 2.0 * dots, 0.0, 4.0))
    
    # Soft-max approximation of max
    # D shape: (N_test, N)
    # For each row, compute soft-max
    # Use: soft_max(D, beta) = (1/beta) * log(sum(exp(beta * D)))
    # But D can be up to ~2, and we have 20000 x 130 matrix
    # Use log-sum-exp for numerical stability
    beta = 1.0 / temperature
    
    # For memory efficiency, process in chunks
    N_test = Y.shape[0]
    chunk_size = 2000
    soft_max_vals = np.empty(N_test)
    
    for start in range(0, N_test, chunk_size):
        end = min(start + chunk_size, N_test)
        D_chunk = D[start:end, :]
        
        # Log-sum-exp trick
        max_D = np.max(D_chunk, axis=1, keepdims=True)
        exp_vals = np.exp(beta * (D_chunk - max_D))
        sum_exp = np.sum(exp_vals, axis=1)
        soft_max_chunk = max_D.squeeze() + np.log(sum_exp) / beta
        soft_max_vals[start:end] = soft_max_chunk
    
    return np.mean(soft_max_vals)

def _potential_gradient(params, Y, temperature):
    """Gradient of the soft-max potential objective."""
    P = params.reshape(-1, 3)
    norms = np.linalg.norm(P, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    P = P / norms
    
    dots = Y @ P.T
    dots = np.clip(dots, -1.0, 1.0)
    D = np.sqrt(np.clip(2.0 - 2.0 * dots, 0.0, 4.0))
    
    beta = 1.0 / temperature
    N_test = Y.shape[0]
    N = P.shape[0]
    
    # Gradient of soft-max w.r.t. D
    # d(soft_max)/dD_ij = exp(beta * (D_ij - max_i)) / sum_k exp(beta * (D_ik - max_i))
    # But we need gradient w.r.t. P
    
    # Compute weights w[i,j] = exp(beta*(D[i,j] - max_i)) / sum_k exp(beta*(D[i,k] - max_i))
    max_D = np.max(D, axis=1, keepdims=True)
    exp_vals = np.exp(beta * (D - max_D))
    sum_exp = np.sum(exp_vals, axis=1, keepdims=True)
    weights = exp_vals / sum_exp  # shape: (N_test, N)
    
    # dD/dP_j = -(Y_i - P_j) / D_ij
    # grad_P[j] = sum_i weights[i,j] * (-(Y_i - P_j) / D_ij)
    
    grad = np.zeros_like(P)
    chunk_size = 2000
    
    for start in range(0, N_test, chunk_size):
        end = min(start + chunk_size, N_test)
        D_chunk = D[start:end, :]
        W_chunk = weights[start:end, :]
        Y_chunk = Y[start:end, :]
        
        # For each j, grad[j] += sum_i W[i,j] * (-(Y_i - P_j) / D_ij)
        # = -sum_i W[i,j] * Y_i / D_ij + sum_i W[i,j] * P_j / D_ij
        
        # Compute S1[j] = sum_i W[i,j] / D_ij
        # Compute S2[j] = sum_i W[i,j] * Y_i / D_ij
        
        with np.errstate(divide='ignore', invalid='ignore'):
            inv_D = 1.0 / np.maximum(D_chunk, 1e-10)
        
        S1 = np.sum(W_chunk * inv_D, axis=0)  # shape: (N,)
        S2 = W_chunk.T @ (Y_chunk * inv_D.T)  # shape: (N, 3)
        
        grad += -S2 + P * S1[:, np.newaxis]
    
    return grad.flatten()

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Generate test points
    N_test = 15000
    Y = _fibonacci_sphere(N_test)
    
    # Initial configuration: Fibonacci spiral
    P = _fibonacci_sphere(N).copy()
    
    best_P = P.copy()
    best_radius = _compute_covering_radius(P, Y)
    
    # Multi-stage optimization with dynamic temperature
    # Stage 1: Coarse optimization with high temperature
    # Stage 2: Fine optimization with lower temperature
    # Stage 3: Local refinement with gradient descent
    
    temperatures = [0.05, 0.03, 0.02, 0.01, 0.005]
    iters_per_stage = [50, 50, 50, 50, 50]
    
    for stage, (temp, n_iters) in enumerate(zip(temperatures, iters_per_stage)):
        if time.time() > deadline:
            break
        
        # Use L-BFGS-B for optimization
        x0 = P.flatten()
        
        def objective(x):
            return _potential_objective(x, Y, temp)
        
        def gradient(x):
            return _potential_gradient(x, Y, temp)
        
        # Run L-BFGS-B
        result = minimize(
            objective,
            x0,
            jac=gradient,
            method='L-BFGS-B',
            options={'maxiter': n_iters, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        
        P = result.x.reshape(-1, 3)
        
        # Normalize
        norms = np.linalg.norm(P, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        P = P / norms
        
        # Evaluate actual covering radius
        current_radius = _compute_covering_radius(P, Y)
        if current_radius < best_radius:
            best_radius = current_radius
            best_P = P.copy()
    
    # Final local refinement: move points toward worst holes
    if time.time() < deadline:
        for refine_iter in range(30):
            if time.time() > deadline:
                break
            
            # Find worst points
            dots = Y @ best_P.T
            dots = np.clip(dots, -1.0, 1.0)
            D = np.sqrt(np.clip(2.0 - 2.0 * dots, 0.0, 4.0))
            min_D = np.min(D, axis=1)
            nearest_idx = np.argmin(D, axis=1)
            
            # Select worst 100 test points
            worst_indices = np.argpartition(min_D, -100)[-100:]
            worst_Y = Y[worst_indices]
            worst_nearest = nearest_idx[worst_indices]
            worst_D = min_D[worst_indices]
            
            # Move points toward worst holes
            step_size = 0.005
            for idx in range(len(worst_indices)):
                y = worst_Y[idx]
                k = worst_nearest[idx]
                d = worst_D[idx]
                if d > 1e-6:
                    dir_vec = (y - best_P[k]) / d
                    # Project to tangent space
                    tang = dir_vec - np.dot(dir_vec, best_P[k]) * best_P[k]
                    best_P[k] += step_size * tang
            
            # Normalize
            norms = np.linalg.norm(best_P, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-10)
            best_P = best_P / norms
            
            # Check improvement
            current_radius = _compute_covering_radius(best_P, Y)
            if current_radius < best_radius:
                best_radius = current_radius
    
    # Final normalization
    norms = np.linalg.norm(best_P, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    best_P = best_P / norms
    
    return best_P
