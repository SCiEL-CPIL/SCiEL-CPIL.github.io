"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]
import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_school = n * m * p
    U0 = np.zeros((n * m, rank_school))
    V0 = np.zeros((m * p, rank_school))
    W0 = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i * m + j, col] = 1
                V0[j * p + k, col] = 1
                W0[k * n + i, col] = 1
                col += 1
    
    start_time = time.time()
    deadline = start_time + 130  # Leave 20s safety margin before 150s limit
    
    best_rank = rank_school
    best_U, best_V, best_W = U0, V0, W0
    
    # Precompute index mappings for vectorized operations
    # U indices: (i, j) -> i*m + j
    # V indices: (j, k) -> j*p + k  
    # W indices: (k, i) -> k*n + i
    
    # For vectorized ALS, we need to build the Gram matrices efficiently
    # U update: A_U[i,j] = sum_k outer(V[j,k], W[k,i])
    # V update: A_V[j,k] = sum_i outer(U[i,j], W[k,i])
    # W update: A_W[k,i] = sum_j outer(U[i,j], V[j,k])
    
    def compute_error(U_curr, V_curr, W_curr):
        """Compute sum of squared errors for all (i,j,k) triples."""
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_curr[i*m+j, :], V_curr[j*p+k, :] * W_curr[k*n+i, :])
                    err += (val - 1)**2
        return err
    
    # Try target ranks from lower to higher (we want minimum rank)
    # Lower bound is max(n*m, m*p, p*n) = max(15, 25, 15) = 25
    # Try from 30 upward, but also try some higher values
    target_ranks = list(range(30, 80))
    
    for target_rank in target_ranks:
        if time.time() > deadline:
            break
        
        max_attempts = 5
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            np.random.seed(attempt * 10000 + target_rank * 7 + 42)
            U_curr = np.random.randn(n * m, target_rank) * 0.3
            V_curr = np.random.randn(m * p, target_rank) * 0.3
            W_curr = np.random.randn(p * n, target_rank) * 0.3
            
            converged = False
            max_iterations = 30
            prev_err = float('inf')
            
            for iteration in range(max_iterations):
                if time.time() > deadline:
                    break
                
                # --- Update U ---
                U_new = np.zeros_like(U_curr)
                for i in range(n):
                    for j in range(m):
                        A_mat = np.zeros((target_rank, target_rank))
                        for k in range(p):
                            A_mat += np.outer(V_curr[j*p+k, :], W_curr[k*n+i, :])
                        reg = 1e-6 * np.eye(target_rank)
                        try:
                            U_new[i*m+j, :] = np.linalg.solve(A_mat + reg, np.ones(target_rank))
                        except:
                            U_new[i*m+j, :] = np.linalg.lstsq(A_mat + reg, np.ones(target_rank), rcond=None)[0]
                U_curr = U_new
                
                # --- Update V ---
                V_new = np.zeros_like(V_curr)
                for j in range(m):
                    for k in range(p):
                        A_mat = np.zeros((target_rank, target_rank))
                        for i in range(n):
                            A_mat += np.outer(U_curr[i*m+j, :], W_curr[k*n+i, :])
                        reg = 1e-6 * np.eye(target_rank)
                        try:
                            V_new[j*p+k, :] = np.linalg.solve(A_mat + reg, np.ones(target_rank))
                        except:
                            V_new[j*p+k, :] = np.linalg.lstsq(A_mat + reg, np.ones(target_rank), rcond=None)[0]
                V_curr = V_new
                
                # --- Update W ---
                W_new = np.zeros_like(W_curr)
                for k in range(p):
                    for i in range(n):
                        A_mat = np.zeros((target_rank, target_rank))
                        for j in range(m):
                            A_mat += np.outer(U_curr[i*m+j, :], V_curr[j*p+k, :])
                        reg = 1e-6 * np.eye(target_rank)
                        try:
                            W_new[k*n+i, :] = np.linalg.solve(A_mat + reg, np.ones(target_rank))
                        except:
                            W_new[k*n+i, :] = np.linalg.lstsq(A_mat + reg, np.ones(target_rank), rcond=None)[0]
                W_curr = W_new
                
                # Check convergence
                err = compute_error(U_curr, V_curr, W_curr)
                
                if err < 1e-12:
                    converged = True
                    break
                if abs(prev_err - err) < 1e-14 and err < 1e-8:
                    converged = True
                    break
                prev_err = err
            
            if converged:
                # Polish pass with tighter tolerance
                for polish_iter in range(10):
                    if time.time() > deadline:
                        break
                    
                    U_new = np.zeros_like(U_curr)
                    for i in range(n):
                        for j in range(m):
                            A_mat = np.zeros((target_rank, target_rank))
                            for k in range(p):
                                A_mat += np.outer(V_curr[j*p+k, :], W_curr[k*n+i, :])
                            reg = 1e-10 * np.eye(target_rank)
                            try:
                                U_new[i*m+j, :] = np.linalg.solve(A_mat + reg, np.ones(target_rank))
                            except:
                                U_new[i*m+j, :] = np.linalg.lstsq(A_mat + reg, np.ones(target_rank), rcond=None)[0]
                    U_curr = U_new
                    
                    V_new = np.zeros_like(V_curr)
                    for j in range(m):
                        for k in range(p):
                            A_mat = np.zeros((target_rank, target_rank))
                            for i in range(n):
                                A_mat += np.outer(U_curr[i*m+j, :], W_curr[k*n+i, :])
                            reg = 1e-10 * np.eye(target_rank)
                            try:
                                V_new[j*p+k, :] = np.linalg.solve(A_mat + reg, np.ones(target_rank))
                            except:
                                V_new[j*p+k, :] = np.linalg.lstsq(A_mat + reg, np.ones(target_rank), rcond=None)[0]
                    V_curr = V_new
                    
                    W_new = np.zeros_like(W_curr)
                    for k in range(p):
                        for i in range(n):
                            A_mat = np.zeros((target_rank, target_rank))
                            for j in range(m):
                                A_mat += np.outer(U_curr[i*m+j, :], V_curr[j*p+k, :])
                            reg = 1e-10 * np.eye(target_rank)
                            try:
                                W_new[k*n+i, :] = np.linalg.solve(A_mat + reg, np.ones(target_rank))
                            except:
                                W_new[k*n+i, :] = np.linalg.lstsq(A_mat + reg, np.ones(target_rank), rcond=None)[0]
                    W_curr = W_new
                
                best_rank = target_rank
                best_U, best_V, best_W = U_curr.copy(), V_curr.copy(), W_curr.copy()
                break
    
    if best_rank < rank_school:
        U_out = best_U.tolist()
        V_out = best_V.tolist()
        W_out = best_W.tolist()
    else:
        U_out = U0.tolist()
        V_out = V0.tolist()
        W_out = W0.tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
