import numpy as np
import time

N = 130

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 140.0  # 10s safety margin
    
    # 1. Initialization: Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    x = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # 2. Optimization Loop
    # We will minimize a combination of Thomson Energy and a Smooth Covering Radius surrogate.
    
    # Sample points for covering radius estimation
    # Use a Fibonacci spiral for the test points too, but denser
    N_test = 5000
    j = np.arange(N_test) + 0.5
    z_t = 1.0 - 2.0 * j / N_test
    phi_t = np.pi * (1.0 + np.sqrt(5.0)) * j
    r_t = np.sqrt(1.0 - z_t * z_t)
    y = np.column_stack([r_t * np.cos(phi_t), r_t * np.sin(phi_t), z_t])
    
    def thomson_energy_and_grad(x):
        # Pairwise distances
        # x: (N, 3)
        # d2[i, j] = |x_i - x_j|^2
        d2 = np.sum(x**2, axis=1)[:, None] + np.sum(x**2, axis=1)[None, :] - 2.0 * x @ x.T
        # Avoid division by zero
        np.fill_diagonal(d2, 1e-10)
        d = np.sqrt(d2)
        
        # Energy: sum_{i != j} 1/d_ij
        # Note: This is 2 * sum_{i < j} 1/d_ij
        # We minimize half of it to get standard Thomson energy
        E = np.sum(1.0 / d) - N # Subtract diagonal terms (1/1e-5 approx)
        
        # Gradient: d/dx_k sum_{i != j} 1/|x_i - x_j|
        # grad_k = - sum_{i != j} (x_k - x_j) / |x_k - x_j|^3
        # Vectorized:
        # For each k, grad[k] = - sum_j (x_k - x_j) / d_kj^3
        # 1/d^3 = 1 / (d2 * sqrt(d2))
        inv_d3 = 1.0 / (d2 * d)
        np.fill_diagonal(inv_d3, 0.0)
        
        # grad = - x @ inv_d3 + sum(inv_d3, axis=1)[:, None] * x ? No.
        # grad[k] = - sum_j inv_d3[k,j] * (x_k - x_j)
        # grad[k] = - x_k * sum_j inv_d3[k,j] + sum_j inv_d3[k,j] * x_j
        # sum_j inv_d3[k,j] * x_j = inv_d3 @ x
        # sum_j inv_d3[k,j] = np.sum(inv_d3, axis=1)
        
        grad = - (x * np.sum(inv_d3, axis=1)[:, None]) + (inv_d3 @ x)
        return E, grad

    def covering_surrogate_and_grad(x, y):
        # y: (M, 3)
        # d2_test[m, i] = |y_m - x_i|^2
        d2_test = np.sum(y**2, axis=1)[:, None] + np.sum(x**2, axis=1)[None, :] - 2.0 * y @ x.T
        d_test = np.sqrt(d2_test)
        
        # For each y_m, find min distance to any x_i
        # min_d2[m] = min_i d2_test[m, i]
        min_d2 = np.min(d2_test, axis=1)
        
        # Smooth Max: (1/tau) * log(sum(exp(tau * min_d2)))
        # We want to minimize the covering radius, so we minimize this value.
        tau = 50.0 # Sharpness of the max approximation
        # For numerical stability
        max_val = np.max(tau * min_d2)
        log_sum_exp = max_val + np.log(np.sum(np.exp(tau * min_d2 - max_val)))
        E_cover = log_sum_exp / tau
        
        # Gradient with respect to x
        # d/dx_i E_cover = (1/tau) * d/dx_i log(sum_m exp(tau * min_d2[m]))
        # = (1/tau) * (1/sum_exp) * sum_m exp(tau * min_d2[m]) * tau * d/dx_i min_d2[m]
        # = (1/sum_exp) * sum_m exp(tau * min_d2[m]) * d/dx_i min_d2[m]
        
        # Let weights w_m = exp(tau * min_d2[m]) / sum_exp
        # w_m is the "importance" of each test point y_m in the max approximation.
        
        exp_vals = np.exp(tau * min_d2 - max_val)
        sum_exp = np.sum(exp_vals)
        w = exp_vals / sum_exp # (M,)
        
        # d/dx_i min_d2[m]
        # min_d2[m] = min_k |y_m - x_k|^2
        # The derivative is only non-zero for the index k* that achieves the minimum.
        # d/dx_k |y_m - x_k|^2 = -2(y_m - x_k)
        
        # We need to sum over m: w_m * d/dx_i min_d2[m]
        # For a fixed i, the contribution comes from all m such that argmin_k d2_test[m,k] == i.
        
        argmins = np.argmin(d2_test, axis=1) # (M,)
        
        grad = np.zeros_like(x)
        # For each test point m, add its contribution to the gradient of x[argmin[m]]
        # Contribution: w[m] * (-2 * (y[m] - x[argmin[m]]))
        
        # Vectorized update using scatter-add
        # We can use np.add.at
        # diffs = y - x[argmins] # (M, 3)
        # contribs = -2.0 * w[:, None] * diffs # (M, 3)
        # np.add.at(grad, argmins, contribs)
        
        diffs = y - x[argmins]
        contribs = -2.0 * w[:, None] * diffs
        np.add.at(grad, argmins, contribs)
        
        return E_cover, grad

    # Optimization parameters
    step_size = 0.01
    lambda_thomson = 1.0
    lambda_cover = 1.0
    max_iter = 500
    
    best_x = x.copy()
    best_score = np.inf
    
    # Simple Riemannian Gradient Descent
    for it in range(max_iter):
        if time.time() > deadline:
            break
            
        E_th, G_th = thomson_energy_and_grad(x)
        E_co, G_co = covering_surrogate_and_grad(x, y)
        
        total_E = lambda_thomson * E_th + lambda_cover * E_co
        total_G = lambda_thomson * G_th + lambda_cover * G_co
        
        # Project gradient to tangent space
        # v = G - (G . x) x
        dot = np.sum(total_G * x, axis=1, keepdims=True)
        G_tangent = total_G - dot * x
        
        # Update
        x = x - step_size * G_tangent
        
        # Normalize
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
        
        # Check convergence / improvement
        # We track the actual covering radius on the test set to monitor progress
        d2_test = np.sum(y**2, axis=1)[:, None] + np.sum(x**2, axis=1)[None, :] - 2.0 * y @ x.T
        min_d2 = np.min(d2_test, axis=1)
        current_covering = np.sqrt(np.max(min_d2))
        
        if current_covering < best_score:
            best_score = current_covering
            best_x = x.copy()
            
        # Adaptive step size: if energy increases, reduce step
        # (Simplified: just use fixed step, L-BFGS would be better but this is fast)
        
    return best_x
# EVOLVE-BLOCK-END
