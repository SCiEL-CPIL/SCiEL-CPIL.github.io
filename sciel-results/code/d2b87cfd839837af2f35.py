import numpy as np
import time

N = 150

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin for safety
    
    # 1. Initial Fibonacci Sphere
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    X = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Function to compute energy
    def log_energy(X):
        # Compute pairwise distances squared
        # X is (N, 3)
        # D2[i, j] = ||X[i] - X[j]||^2
        # Use broadcasting: (N, 1, 3) - (1, N, 3) -> (N, N, 3)
        diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
        D2 = np.sum(diff * diff, axis=2)
        # Mask diagonal
        np.fill_diagonal(D2, 1.0) # Avoid div by 0, we will subtract later or ignore
        # Energy is sum_{i<j} log(1/d) = - sum_{i<j} log(d) = -0.5 * sum_{i!=j} log(d)
        # d = sqrt(D2)
        # log(d) = 0.5 * log(D2)
        # So Energy = -0.5 * sum_{i!=j} 0.5 * log(D2) = -0.25 * sum_{i!=j} log(D2)
        # Wait, sum_{i<j} log(1/d) = sum_{i<j} -log(d).
        # sum_{i!=j} log(d) = 2 * sum_{i<j} log(d).
        # So sum_{i<j} log(1/d) = -0.5 * sum_{i!=j} log(d).
        # log(d) = 0.5 log(D2).
        # So Energy = -0.25 * sum_{i!=j} log(D2).
        
        # We need to be careful with the diagonal. D2[i,i]=0, log(0)=-inf.
        # We set diagonal to 1.0, so log(1)=0. This effectively removes them from the sum.
        log_D2 = np.log(D2)
        total_log_D2 = np.sum(log_D2)
        return -0.25 * total_log_D2

    # Function to compute gradient
    def gradient(X):
        diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
        D2 = np.sum(diff * diff, axis=2)
        np.fill_diagonal(D2, 1.0)
        
        # W[i,j] = 1/D2[i,j]
        W = 1.0 / D2
        np.fill_diagonal(W, 0.0)
        
        # F_i = sum_j W[i,j] (X_j - X_i)
        # F_i = sum_j W[i,j] X_j - X_i sum_j W[i,j]
        # S1 = W @ X  (N, 3)
        S1 = W @ X
        # S2 = W @ ones (N,)
        S2 = np.sum(W, axis=1)
        
        F = S1 - X * S2[:, np.newaxis]
        
        # Project to tangent space
        # dot_FX = sum(F * X, axis=1)
        dot_FX = np.sum(F * X, axis=1)
        F_proj = F - X * dot_FX[:, np.newaxis]
        
        return F_proj

    # Initial energy
    E_curr = log_energy(X)
    
    # Gradient Descent with Backtracking Line Search
    eta = 0.1
    min_eta = 1e-8
    max_iter = 10000
    
    for k in range(max_iter):
        if time.time() > deadline:
            break
        
        F = gradient(X)
        
        # Check for convergence
        # If max gradient norm is small, break
        g_norm = np.max(np.linalg.norm(F, axis=1))
        if g_norm < 1e-8:
            break
        
        # Backtracking line search
        # Try to reduce energy
        success = False
        eta_try = eta
        while eta_try > min_eta:
            X_new = X + eta_try * F
            # Normalize
            norms = np.linalg.norm(X_new, axis=1)
            # Avoid zero norms
            norms[norms < 1e-12] = 1.0
            X_new = X_new / norms[:, np.newaxis]
            
            E_new = log_energy(X_new)
            
            if E_new < E_curr:
                X = X_new
                E_curr = E_new
                eta = eta_try * 1.5 # Increase step if successful
                success = True
                break
            else:
                eta_try *= 0.5
        
        if not success:
            # If even min_eta didn't work, we might be stuck or precision limit
            # Try a smaller step or break
            if eta_try <= min_eta:
                 # Try one more time with a tiny step just in case
                 X_new = X + min_eta * F
                 norms = np.linalg.norm(X_new, axis=1)
                 norms[norms < 1e-12] = 1.0
                 X_new = X_new / norms[:, np.newaxis]
                 E_new = log_energy(X_new)
                 if E_new < E_curr:
                     X = X_new
                     E_curr = E_new
                 else:
                     break
            else:
                 # This shouldn't happen if min_eta is small enough, but just in case
                 break

    # Final normalization and check
    norms = np.linalg.norm(X, axis=1)
    X = X / norms[:, np.newaxis]
    
    return X
# EVOLVE-BLOCK-END
