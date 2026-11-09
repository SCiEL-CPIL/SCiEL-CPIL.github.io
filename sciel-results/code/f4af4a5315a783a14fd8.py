"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 45
    deadline = time.time() + 130.0  # 20s safety margin before 150s limit
    
    # Schoolbook baseline
    U_sb = np.zeros((n*m, schoolbook_rank))
    V_sb = np.zeros((m*p, schoolbook_rank))
    W_sb = np.zeros((p*n, schoolbook_rank))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                c = i * (m*p) + j * p + k
                U_sb[i*m+j, c] = 1.0
                V_sb[j*p+k, c] = 1.0
                W_sb[k*n+i, c] = 1.0
    
    def verify(U, V, W, num_tests=50):
        """Verify that the decomposition exactly represents matrix multiplication."""
        np.random.seed(42)
        for _ in range(num_tests):
            A = np.random.randn(n, m)
            B = np.random.randn(m, p)
            C_true = A @ B
            A_flat = A.flatten()
            B_flat = B.flatten()
            f = U.T @ A_flat  # shape (rank,)
            g = V.T @ B_flat  # shape (rank,)
            W_reshaped = W.T.reshape(-1, p, n)  # (rank, p, n)
            C_recon = np.einsum('c,c,cpi->ip', f, g, W_reshaped)
            if not np.allclose(C_recon, C_true, atol=1e-10):
                return False
        return True
    
    def is_half_integer(arr, tol=1e-8):
        return np.allclose(arr * 2, np.round(arr * 2), atol=tol)
    
    best_rank = schoolbook_rank
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Try to find lower rank using L-BFGS-B optimization
    target_ranks = [44, 42, 40, 38, 36, 34, 32, 30]
    for target_rank in target_ranks:
        if time.time() > deadline - 15:
            break
        
        found = False
        for attempt in range(2):
            if time.time() > deadline - 5:
                break
            
            np.random.seed(attempt * 999 + target_rank)
            
            U_init = np.random.randn(n*m, target_rank) * 0.3
            V_init = np.random.randn(m*p, target_rank) * 0.3
            W_init = np.random.randn(p*n, target_rank) * 0.3
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            def objective(x):
                U = x[:n*m*target_rank].reshape(n*m, target_rank)
                V = x[n*m*target_rank : n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
                W = x[n*m*target_rank + m*p*target_rank :].reshape(p*n, target_rank)
                
                total_err = 0.0
                for seed in [0, 1, 2, 3, 4]:
                    np.random.seed(seed)
                    A = np.random.randn(n, m)
                    B = np.random.randn(m, p)
                    C_true = A @ B
                    A_flat = A.flatten()
                    B_flat = B.flatten()
                    f = U.T @ A_flat
                    g = V.T @ B_flat
                    W_reshaped = W.T.reshape(-1, p, n)
                    C_recon = np.einsum('c,c,cpi->ip', f, g, W_reshaped)
                    total_err += np.sum((C_recon - C_true)**2)
                
                return total_err
            
            result = minimize(objective, x0, method='L-BFGS-B', 
                            options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8})
            
            if result.fun < 1e-10:
                U_opt = result.x[:n*m*target_rank].reshape(n*m, target_rank)
                V_opt = result.x[n*m*target_rank : n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
                W_opt = result.x[n*m*target_rank + m*p*target_rank :].reshape(p*n, target_rank)
                
                U_round = np.round(U_opt * 2) / 2
                V_round = np.round(V_opt * 2) / 2
                W_round = np.round(W_opt * 2) / 2
                
                if verify(U_round, V_round, W_round):
                    best_rank = target_rank
                    best_U, best_V, best_W = U_round, V_round, W_round
                    found = True
                    break
        
        if found:
            continue
        else:
            continue
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
