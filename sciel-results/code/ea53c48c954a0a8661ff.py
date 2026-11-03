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
    deadline = start_time + 140  # Leave 10s safety margin before 150s limit
    
    def check_decomposition(U, V, W, tol=1e-8):
        """Check if U, V, W exactly decompose the tensor."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def polish_solution(U, V, W, max_iter=100, tol=1e-12):
        """High-precision polish pass with tighter tolerance."""
        rank = U.shape[1]
        U_curr = U.astype(np.float64).copy()
        V_curr = V.astype(np.float64).copy()
        W_curr = W.astype(np.float64).copy()
        
        for iteration in range(max_iter):
            if time.time() > deadline:
                break
            
            # Update U
            U_new = np.zeros_like(U_curr)
            for i in range(n):
                for j in range(m):
                    A_mat = np.zeros((rank, rank))
                    for k in range(p):
                        A_mat += np.outer(V_curr[j*p+k, :], W_curr[k*n+i, :])
                    try:
                        U_new[i*m+j, :] = np.linalg.solve(A_mat + 1e-12*np.eye(rank), np.ones(rank))
                    except:
                        U_new[i*m+j, :] = np.linalg.lstsq(A_mat + 1e-12*np.eye(rank), np.ones(rank), rcond=None)[0]
            U_curr = U_new
            
            # Update V
            V_new = np.zeros_like(V_curr)
            for j in range(m):
                for k in range(p):
                    A_mat = np.zeros((rank, rank))
                    for i in range(n):
                        A_mat += np.outer(U_curr[i*m+j, :], W_curr[k*n+i, :])
                    try:
                        V_new[j*p+k, :] = np.linalg.solve(A_mat + 1e-12*np.eye(rank), np.ones(rank))
                    except:
                        V_new[j*p+k, :] = np.linalg.lstsq(A_mat + 1e-12*np.eye(rank), np.ones(rank), rcond=None)[0]
            V_curr = V_new
            
            # Update W
            W_new = np.zeros_like(W_curr)
            for k in range(p):
                for i in range(n):
                    A_mat = np.zeros((rank, rank))
                    for j in range(m):
                        A_mat += np.outer(U_curr[i*m+j, :], V_curr[j*p+k, :])
                    try:
                        W_new[k*n+i, :] = np.linalg.solve(A_mat + 1e-12*np.eye(rank), np.ones(rank))
                    except:
                        W_new[k*n+i, :] = np.linalg.lstsq(A_mat + 1e-12*np.eye(rank), np.ones(rank), rcond=None)[0]
            W_curr = W_new
            
            # Check convergence
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U_curr[i*m+j, :] * V_curr[j*p+k, :] * W_curr[k*n+i, :])
                        err += (val - 1)**2
            
            if err < tol:
                break
        
        return U_curr, V_curr, W_curr
    
    def try_rank(target_rank, seed, max_iterations=50):
        """Try to find a decomposition at the given rank."""
        np.random.seed(seed)
        U_curr = np.random.randn(n * m, target_rank) * 0.1
        V_curr = np.random.randn(m * p, target_rank) * 0.1
        W_curr = np.random.randn(p * n, target_rank) * 0.1
        
        for iteration in range(max_iterations):
            if time.time() > deadline:
                return None
            
            # Update U with regularization
            U_new = np.zeros_like(U_curr)
            for i in range(n):
                for j in range(m):
                    A_mat = np.zeros((target_rank, target_rank))
                    for k in range(p):
                        A_mat += np.outer(V_curr[j*p+k, :], W_curr[k*n+i, :])
                    try:
                        U_new[i*m+j, :] = np.linalg.solve(A_mat + 1e-8*np.eye(target_rank), np.ones(target_rank))
                    except:
                        U_new[i*m+j, :] = np.linalg.lstsq(A_mat + 1e-8*np.eye(target_rank), np.ones(target_rank), rcond=None)[0]
            U_curr = U_new
            
            # Update V
            V_new = np.zeros_like(V_curr)
            for j in range(m):
                for k in range(p):
                    A_mat = np.zeros((target_rank, target_rank))
                    for i in range(n):
                        A_mat += np.outer(U_curr[i*m+j, :], W_curr[k*n+i, :])
                    try:
                        V_new[j*p+k, :] = np.linalg.solve(A_mat + 1e-8*np.eye(target_rank), np.ones(target_rank))
                    except:
                        V_new[j*p+k, :] = np.linalg.lstsq(A_mat + 1e-8*np.eye(target_rank), np.ones(target_rank), rcond=None)[0]
            V_curr = V_new
            
            # Update W
            W_new = np.zeros_like(W_curr)
            for k in range(p):
                for i in range(n):
                    A_mat = np.zeros((target_rank, target_rank))
                    for j in range(m):
                        A_mat += np.outer(U_curr[i*m+j, :], V_curr[j*p+k, :])
                    try:
                        W_new[k*n+i, :] = np.linalg.solve(A_mat + 1e-8*np.eye(target_rank), np.ones(target_rank))
                    except:
                        W_new[k*n+i, :] = np.linalg.lstsq(A_mat + 1e-8*np.eye(target_rank), np.ones(target_rank), rcond=None)[0]
            W_curr = W_new
            
            # Check convergence
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U_curr[i*m+j, :] * V_curr[j*p+k, :] * W_curr[k*n+i, :])
                        err += (val - 1)**2
            
            if err < 1e-10:
                # Polish for exactness
                U_polished, V_polished, W_polished = polish_solution(U_curr, V_curr, W_curr)
                if check_decomposition(U_polished, V_polished, W_polished, tol=1e-9):
                    return (U_polished, V_polished, W_polished)
        
        return None
    
    best_rank = rank_school
    best_U, best_V, best_W = U0, V0, W0
    
    # Target ranks: start from reasonable lower bounds
    # Lower bound: max(n*m, m*p, p*n) = max(15, 25, 15) = 25
    # Try ranks from 70 down to 25
    target_ranks = list(range(70, 24, -1))
    
    for target_rank in target_ranks:
        if time.time() > deadline:
            break
        
        # Try multiple seeds
        found = False
        for seed in range(5):
            if time.time() > deadline:
                break
            
            result = try_rank(target_rank, seed * 1000 + target_rank)
            if result is not None:
                best_rank = target_rank
                best_U, best_V, best_W = result
                found = True
                break
        
        if found:
            # Try to go lower
            continue
    
    # If we found a lower rank, use it; otherwise use schoolbook
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
