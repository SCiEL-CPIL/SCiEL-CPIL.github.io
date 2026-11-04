import numpy as np
import time
import sys
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    U_shape = (n * m, rank_schoolbook)
    V_shape = (m * p, rank_schoolbook)
    W_shape = (p * n, rank_schoolbook)
    
    # Generate schoolbook solution as fallback
    U_sb = np.zeros(U_shape)
    V_sb = np.zeros(V_shape)
    W_sb = np.zeros(W_shape)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
                
    # Function to verify if a set of matrices U, V, W with rank r is a valid decomposition
    def verify(U, V, W):
        # U: (n*m, r), V: (m*p, r), W: (p*n, r)
        # Check sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 for all i,j,k
        # Also check entries are close to half-integers
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for c in range(U.shape[1]):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(val - 1.0) > 1e-6:
                        return False
        # Check half-integer constraint
        for mat in [U, V, W]:
            for val in mat:
                if abs(val * 2 - round(val * 2)) > 1e-6:
                    return False
        return True

    # Search for lower rank
    start_time = time.time()
    total_time_limit = 120.0  # seconds - use most of the available time
    
    # We will try to find a solution for decreasing target ranks
    target_ranks = [rank_schoolbook - 1, rank_schoolbook - 2, rank_schoolbook - 3]
    
    # Function to try finding a solution for a given rank
    def try_find_solution(target_rank, time_budget):
        num_vars = (n*m + m*p + p*n) * target_rank
        
        def unpack_vars(x):
            # x is flat array
            U_flat = x[:n*m*target_rank]
            V_flat = x[n*m*target_rank : n*m*target_rank + m*p*target_rank]
            W_flat = x[n*m*target_rank + m*p*target_rank :]
            
            U = U_flat.reshape((n*m, target_rank))
            V = V_flat.reshape((m*p, target_rank))
            W = W_flat.reshape((p*n, target_rank))
            return U, V, W

        def objective(x):
            U, V, W = unpack_vars(x)
            res = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0
                        for c in range(target_rank):
                            val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                        res += (val - 1.0)**2
            return res

        # Generate multiple initializations
        np.random.seed(42 + target_rank)
        
        # Strategy 1: Start from schoolbook and zero out some columns
        # Strategy 2: Random initialization
        # Strategy 3: Perturbed schoolbook
        
        best_solution = None
        best_fun = float('inf')
        
        for restart in range(100):  # More restarts
            if time.time() - start_time > total_time_limit - 5:
                break
                
            # Different initialization strategies
            if restart % 3 == 0:
                # Start from schoolbook with some columns zeroed
                x0 = np.zeros(num_vars)
                U0 = np.zeros((n*m, target_rank))
                V0 = np.zeros((m*p, target_rank))
                W0 = np.zeros((p*n, target_rank))
                
                # Copy first target_rank columns from schoolbook
                for c in range(target_rank):
                    U0[:, c] = U_sb[:, c]
                    V0[:, c] = V_sb[:, c]
                    W0[:, c] = W_sb[:, c]
                
                # Add small noise
                x0 = np.concatenate([
                    (U0 + np.random.randn(n*m, target_rank) * 0.1).flatten(),
                    (V0 + np.random.randn(m*p, target_rank) * 0.1).flatten(),
                    (W0 + np.random.randn(p*n, target_rank) * 0.1).flatten()
                ])
            elif restart % 3 == 1:
                # Random initialization
                x0 = np.random.randn(num_vars) * 0.5
            else:
                # Sparse initialization
                x0 = np.random.randn(num_vars)
                x0[np.random.choice(num_vars, int(num_vars * 0.7), replace=False)] = 0
                
            # Use L-BFGS-B
            try:
                res = minimize(objective, x0, method='L-BFGS-B', 
                              options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8})
                
                if res.fun < best_fun:
                    best_fun = res.fun
                    
                    if res.fun < 1e-10:
                        U, V, W = unpack_vars(res.x)
                        # Snap to half-integers
                        U_snap = np.round(U * 2) / 2
                        V_snap = np.round(V * 2) / 2
                        W_snap = np.round(W * 2) / 2
                        
                        if verify(U_snap, V_snap, W_snap):
                            best_solution = (U_snap, V_snap, W_snap)
                            break
            except Exception:
                pass
                
        return best_solution

    # Try each target rank with allocated time
    for target_rank in target_ranks:
        if time.time() - start_time > total_time_limit - 10:
            break
            
        solution = try_find_solution(target_rank, 30.0)
        
        if solution is not None:
            U, V, W = solution
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
    
    # Return schoolbook
    return {
        "U": U_sb.tolist(),
        "V": V_sb.tolist(),
        "W": W_sb.tolist()
    }
