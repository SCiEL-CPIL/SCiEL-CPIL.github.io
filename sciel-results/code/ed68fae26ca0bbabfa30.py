import numpy as np
import time
from scipy.optimize import minimize

N = 150

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin

    # Initial points: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    X_init = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Flatten for optimizer
    x0 = X_init.flatten()
    
    def energy_and_grad(x_flat):
        X = x_flat.reshape(N, 3)
        # Normalize
        norms = np.linalg.norm(X, axis=1)
        # Avoid division by zero
        norms[norms < 1e-12] = 1e-12
        P = X / norms[:, np.newaxis]
        
        # Calculate pairwise distances
        # d_ij^2 = 2 - 2 * P_i . P_j
        dot_prod = P @ P.T
        d2 = 2.0 - 2.0 * dot_prod
        np.fill_diagonal(d2, 1.0) # Avoid self interaction
        
        # Clip to avoid log(0) or negative args due to float precision
        d2 = np.clip(d2, 1e-12, None)
        d = np.sqrt(d2)
        
        # Energy: sum_{i<j} log(1/d_ij) = - sum_{i<j} log(d_ij)
        # We sum over all i != j and divide by 2
        # Mask diagonal
        mask = ~np.eye(N, dtype=bool)
        log_d = np.log(d)
        # Sum only upper triangle to avoid double counting? 
        # Or sum all and divide by 2.
        # Using fsum for precision
        log_d_sum = np.sum(log_d[mask])
        energy = -0.5 * log_d_sum
        
        # Gradient
        # grad_E_k = - sum_{j != k} (P_k - P_j) / d_kj^2
        # Vectorized:
        # diff = P[:, np.newaxis, :] - P[np.newaxis, :, :] # (N, N, 3)
        # grad = - sum_j diff[k, j] / d2[k, j]
        
        diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
        # d2 is (N, N)
        # We need to divide each vector by d2[k, j]
        # d2_inv = 1 / d2
        # d2_inv[diag] = 0
        d2_inv = np.zeros_like(d2)
        np.fill_diagonal(d2_inv, 0.0)
        # Avoid division by zero in d2_inv
        d2_inv[d2 < 1e-12] = 0.0
        
        # grad_k = - sum_j (P_k - P_j) * (1/d2_kj)
        # diff is (N, N, 3). d2_inv is (N, N).
        # We want to multiply each vector in diff[k, j] by d2_inv[k, j]
        # Then sum over j (axis 1)
        # d2_inv_expanded = d2_inv[:, :, np.newaxis]
        # weighted_diff = diff * d2_inv[:, :, np.newaxis]
        # grad_P = - np.sum(weighted_diff, axis=1)
        
        weighted_diff = diff * d2_inv[:, :, np.newaxis]
        grad_P = -np.sum(weighted_diff, axis=1)
        
        # Project gradient to tangent space
        # grad_X = grad_P - (grad_P . P) P
        # grad_P . P is a vector of dot products
        dot_grad_P = np.sum(grad_P * P, axis=1)
        grad_X = grad_P - dot_grad_P[:, np.newaxis] * P
        
        # The gradient with respect to X (Cartesian) is the same as grad_X?
        # Let's verify.
        # f(X) = E(P(X)).
        # d f / d X = (d P / d X)^T d E / d P.
        # P = X / ||X||.
        # d P_i / d X_j = (1/||X||) (delta_ij - X_i X_j / ||X||^2).
        # This is the projection matrix onto the tangent space.
        # So yes, grad_X = grad_P projected.
        
        return energy, grad_X.flatten()
    
    best_energy = float('inf')
    best_x = x0
    
    # Optimization
    # L-BFGS-B is good for this
    try:
        res = minimize(
            energy_and_grad, 
            x0, 
            method='L-BFGS-B', 
            jac=True, # Since we return (f, g)
            options={'maxiter': 500, 'gtol': 1e-10, 'maxfun': 5000}
        )
        if time.time() < deadline:
            best_energy = res.fun
            best_x = res.x
    except Exception:
        pass
    
    # Polish pass: Run again from best_x with tighter tolerance
    if time.time() < deadline - 10:
        try:
            res2 = minimize(
                energy_and_grad, 
                best_x, 
                method='L-BFGS-B', 
                jac=True,
                options={'maxiter': 1000, 'gtol': 1e-12, 'maxfun': 10000}
            )
            if res2.fun < best_energy:
                best_energy = res2.fun
                best_x = res2.x
        except Exception:
            pass

    # Final check: try a random restart?
    # If time permits, generate a random valid start and optimize.
    # But Fibonacci is usually very good.
    
    X_final = best_x.reshape(N, 3)
    # Normalize again to be safe
    norms = np.linalg.norm(X_final, axis=1)
    X_final = X_final / norms[:, np.newaxis]
    
    return X_final
