import numpy as np
import math

N = 150

def construct():
    # Initial Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    X = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Normalize just in case
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    
    def energy(X):
        # Compute pairwise squared distances
        # X is (N, 3)
        # D is (N, N)
        # D[i,j] = ||X[i] - X[j]||^2
        # Using (a-b)^2 = a^2 + b^2 - 2ab
        # X^2 is all 1
        # D = 1 + 1 - 2 * X @ X.T
        XX = X @ X.T
        D = 2.0 - 2.0 * XX
        np.fill_diagonal(D, 0.0)
        
        # Energy is - sum_{i<j} log(sqrt(D[i,j])) = -0.5 * sum_{i<j} log(D[i,j])
        # We only need upper triangle
        iu = np.triu_indices(N, 1)
        d_upper = D[iu]
        # Avoid log(0)
        d_upper = np.maximum(d_upper, 1e-15)
        return -0.5 * np.sum(np.log(d_upper))
        
    def rgrad(X):
        # Compute Riemannian Gradient
        # g_k = WX_k + ( (N-1)/2 - W_row_sum[k] ) * x_k
        # W_ij = 1 / ||x_i - x_j||^2 for i != j, 0 otherwise
        
        XX = X @ X.T
        D = 2.0 - 2.0 * XX
        np.fill_diagonal(D, 1.0) # Avoid div by 0, we will zero out diagonal later
        
        W = 1.0 / D
        np.fill_diagonal(W, 0.0)
        
        W_row_sum = np.sum(W, axis=1)
        WX = W @ X
        
        coeff = (N - 1) / 2.0 - W_row_sum
        G = WX + coeff[:, np.newaxis] * X
        
        return G

    def exp_map(X, D):
        # Exponential map on sphere
        # X is (N, 3), D is (N, 3) tangent vectors
        # Exp_x(v) = x cos(||v||) + v/||v|| sin(||v||)
        norms = np.linalg.norm(D, axis=1, keepdims=True)
        # Handle zero norms
        safe_norms = np.where(norms < 1e-10, 1.0, norms)
        cos_term = np.cos(safe_norms)
        sin_term = np.sin(safe_norms) / safe_norms
        
        # If norm was 0, D is 0, so result should be X
        # The formula gives X * 1 + 0 = X. Correct.
        return X * cos_term + D * sin_term
        
    def parallel_transport(X_old, X_new, V):
        # Transport V (tangent at X_old) to X_new
        # PT = V + (X_new - X_old) * (V dot X_old)
        # V is (N, 3)
        dots = np.sum(V * X_old, axis=1, keepdims=True)
        return V + (X_new - X_old) * dots

    E_curr = energy(X)
    G_curr = rgrad(X)
    
    # Check if gradient is zero
    if np.linalg.norm(G_curr) < 1e-10:
        return X

    D_curr = -G_curr
    alpha = 1.0
    
    max_iter = 2000
    c1 = 1e-4
    rho = 0.5
    
    for _ in range(max_iter):
        # Line Search
        # Condition: E(X_new) <= E_curr + c1 * alpha * <G_curr, D_curr>
        # <G_curr, D_curr> is negative (descent direction)
        grad_dot_dir = np.sum(G_curr * D_curr)
        
        # If grad_dot_dir is positive, we have an error, reset direction
        if grad_dot_dir > 0:
            D_curr = -G_curr
            grad_dot_dir = np.sum(G_curr * D_curr)
            
        X_trial = exp_map(X_curr, alpha * D_curr)
        E_trial = energy(X_trial)
        
        # Armijo condition
        while E_trial > E_curr + c1 * alpha * grad_dot_dir:
            alpha *= rho
            X_trial = exp_map(X_curr, alpha * D_curr)
            E_trial = energy(X_trial)
            if alpha < 1e-8:
                break
                
        if alpha < 1e-8:
            break
            
        X_new = X_trial
        E_new = E_trial
        G_new = rgrad(X_new)
        
        # Check convergence
        if np.linalg.norm(G_new) < 1e-8:
            break
            
        # Update Conjugate Gradient
        if _ > 0:
            # Transport G_curr to X_new to compute beta
            G_curr_transported = parallel_transport(X_curr, X_new, G_curr)
            
            # Polak-Ribiere Beta
            num = np.sum((G_new - G_curr_transported) * G_new)
            den = np.sum(G_curr_transported * G_curr_transported)
            
            if den > 1e-12:
                beta = num / den
                if beta < 0:
                    beta = 0
            else:
                beta = 0
                
            # Transport D_curr to X_new
            D_curr_transported = parallel_transport(X_curr, X_new, D_curr)
            
            D_new = -G_new + beta * D_curr_transported
        else:
            D_new = -G_new
            
        X_curr = X_new
        E_curr = E_new
        G_curr = G_new
        D_curr = D_new
        
        # Reset alpha if it became very small, or keep it
        # Usually keep alpha for next step, but maybe cap it
        if alpha < 0.1:
            alpha = 0.1
        elif alpha > 10.0:
            alpha = 10.0

    return X_curr

# EVOLVE-BLOCK-START
import numpy as np


def construct():
    # Golden-angle (Fibonacci) spiral points on the unit sphere: a good but improvable start.
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    X = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Normalize just in case
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    
    def energy(X):
        # Compute pairwise squared distances
        XX = X @ X.T
        D = 2.0 - 2.0 * XX
        np.fill_diagonal(D, 0.0)
        
        # Energy is - sum_{i<j} log(sqrt(D[i,j])) = -0.5 * sum_{i<j} log(D[i,j])
        iu = np.triu_indices(N, 1)
        d_upper = D[iu]
        d_upper = np.maximum(d_upper, 1e-15)
        return -0.5 * np.sum(np.log(d_upper))
        
    def rgrad(X):
        XX = X @ X.T
        D = 2.0 - 2.0 * XX
        np.fill_diagonal(D, 1.0)
        
        W = 1.0 / D
        np.fill_diagonal(W, 0.0)
        
        W_row_sum = np.sum(W, axis=1)
        WX = W @ X
        
        coeff = (N - 1) / 2.0 - W_row_sum
        G = WX + coeff[:, np.newaxis] * X
        
        return G

    def exp_map(X, D):
        norms = np.linalg.norm(D, axis=1, keepdims=True)
        safe_norms = np.where(norms < 1e-10, 1.0, norms)
        cos_term = np.cos(safe_norms)
        sin_term = np.sin(safe_norms) / safe_norms
        
        return X * cos_term + D * sin_term
        
    def parallel_transport(X_old, X_new, V):
        dots = np.sum(V * X_old, axis=1, keepdims=True)
        return V + (X_new - X_old) * dots

    E_curr = energy(X)
    G_curr = rgrad(X)
    
    if np.linalg.norm(G_curr) < 1e-10:
        return X

    D_curr = -G_curr
    alpha = 1.0
    
    max_iter = 500
    c1 = 1e-4
    rho = 0.5
    
    for _ in range(max_iter):
        grad_dot_dir = np.sum(G_curr * D_curr)
        
        if grad_dot_dir > 0:
            D_curr = -G_curr
            grad_dot_dir = np.sum(G_curr * D_curr)
            
        X_trial = exp_map(X_curr, alpha * D_curr)
        E_trial = energy(X_trial)
        
        while E_trial > E_curr + c1 * alpha * grad_dot_dir:
            alpha *= rho
            X_trial = exp_map(X_curr, alpha * D_curr)
            E_trial = energy(X_trial)
            if alpha < 1e-8:
                break
                
        if alpha < 1e-8:
            break
            
        X_new = X_trial
        E_new = E_trial
        G_new = rgrad(X_new)
        
        if np.linalg.norm(G_new) < 1e-8:
            break
            
        if _ > 0:
            G_curr_transported = parallel_transport(X_curr, X_new, G_curr)
            
            num = np.sum((G_new - G_curr_transported) * G_new)
            den = np.sum(G_curr_transported * G_curr_transported)
            
            if den > 1e-12:
                beta = num / den
                if beta < 0:
                    beta = 0
            else:
                beta = 0
                
            D_curr_transported = parallel_transport(X_curr, X_new, D_curr)
            
            D_new = -G_new + beta * D_curr_transported
        else:
            D_new = -G_new
            
        X_curr = X_new
        E_curr = E_new
        G_curr = G_new
        D_curr = D_new
        
        if alpha < 0.1:
            alpha = 0.1
        elif alpha > 10.0:
            alpha = 10.0

    return X_curr
# EVOLVE-BLOCK-END
