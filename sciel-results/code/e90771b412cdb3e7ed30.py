import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 7]

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying 
    a 2x3 matrix by a 3x7 matrix.
    
    The output factor W follows the official trace convention W[k*n+i, r].
    U has shape (n*m, rank)
    V has shape (m*p, rank)
    W has shape (p*n, rank)
    
    We first construct the schoolbook decomposition (rank 42).
    Then, we attempt to find a lower rank decomposition (rank 41) using a
    smooth optimization approach (Levenberg-Marquardt style via scipy) on the
    non-convex landscape of tensor decomposition. If successful, we return
    the lower rank result. Otherwise, we return the schoolbook result.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    # 1. Construct Schoolbook Decomposition (Rank 42)
    rank_schoolbook = n * m * p
    U_school = np.zeros((n * m, rank_schoolbook))
    V_school = np.zeros((m * p, rank_schoolbook))
    W_school = np.zeros((p * n, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
                
    # Store the best result found so far
    best_rank = rank_schoolbook
    best_U = U_school
    best_V = V_school
    best_W = W_school
    
    # Precompute the target tensor T
    # T is a 3-tensor of shape (p*n, n*m, m*p)
    # T[k*n+i, i*m+j, j*p+k] = 1
    T = np.zeros((p * n, n * m, m * p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[k * n + i, i * m + j, j * p + k] = 1.0
                
    # 2. Attempt to find a lower rank decomposition (Rank 41)
    # We use a smooth optimization approach. The objective is to minimize
    # the squared error between the reconstructed tensor and T.
    # Parameters are flattened U, V, W matrices.
    
    target_rank = rank_schoolbook - 1  # Try 41
    
    if time.time() < deadline - 30:  # Check if we have enough time
        # Initialize parameters
        # We can try to initialize by taking the schoolbook decomposition
        # and perturbing it, or by using random initialization.
        # A common strategy is to use the "identity" mapping of the first R
        # columns of the schoolbook decomposition, but that won't work directly
        # because the rank is lower.
        # Instead, we use random initialization with multiple restarts.
        
        num_restarts = 5
        found_better = False
        
        for restart in range(num_restarts):
            if time.time() > deadline - 15:
                break
                
            # Random initialization
            U_rand = np.random.randn(n * m, target_rank)
            V_rand = np.random.randn(m * p, target_rank)
            W_rand = np.random.randn(p * n, target_rank)
            
            # Scale to avoid large gradients
            U_rand /= np.sqrt(np.mean(U_rand**2)) + 1e-6
            V_rand /= np.sqrt(np.mean(V_rand**2)) + 1e-6
            W_rand /= np.sqrt(np.mean(W_rand**2)) + 1e-6
            
            # Flatten parameters
            x0 = np.concatenate([U_rand.flatten(), V_rand.flatten(), W_rand.flatten()])
            
            def objective(x):
                U = x[:n*m*target_rank].reshape(n*m, target_rank)
                V = x[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
                W = x[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
                
                # Reconstruct tensor
                # T_recon[a, b, c] = sum_r W[a, r] * U[b, r] * V[c, r]
                # Using einsum for efficiency
                T_recon = np.einsum('ar,br,cr->abc', W, U, V)
                
                # Squared error
                error = np.sum((T_recon - T)**2)
                return error
            
            # Use L-BFGS-B or BFGS. L-BFGS-B is good for large problems, but BFGS
            # might be better for this small one. Let's use L-BFGS-B with a limit.
            # We also add a small regularization to help convergence? No, we want exact.
            
            try:
                # Set a time limit for this specific optimization call if possible,
                # but scipy doesn't directly support wall-clock limits easily.
                # We'll rely on the outer loop deadline check.
                # To prevent hanging, we limit the number of iterations.
                res = minimize(objective, x0, method='L-BFGS-B', 
                               options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8})
                
                # Check if we found a solution with very small error
                if res.fun < 1e-6:  # Threshold for "exact" in floating point
                    # Verify with higher precision
                    U_opt = res.x[:n*m*target_rank].reshape(n*m, target_rank)
                    V_opt = res.x[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
                    W_opt = res.x[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
                    
                    T_recon = np.einsum('ar,br,cr->abc', W_opt, U_opt, V_opt)
                    if np.allclose(T_recon, T, atol=1e-9):
                        best_rank = target_rank
                        best_U = U_opt
                        best_V = V_opt
                        best_W = W_opt
                        found_better = True
                        break  # Break out of restarts if we found a good one
                        
            except Exception:
                pass

    # 3. Return the best result
    # Ensure we return lists as per the interface
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
