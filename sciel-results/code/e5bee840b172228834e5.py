import numpy as np
import time
import itertools

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    R_school = n * m * p  # 18
    
    # Build the target tensor T[i, j, k] = 1 for all (i,j,k)
    # Indices: i in [0, n), j in [0, m), k in [0, p)
    # Note: The task says U (n*m), V (m*p), W (p*n)
    # T[i, j, k] corresponds to U[i*m+j], V[j*p+k], W[k*n+i]
    T = np.ones((n, m, p))
    
    # Build schoolbook solution as fallback
    U_school = np.zeros((n * m, R_school))
    V_school = np.zeros((m * p, R_school))
    W_school = np.zeros((p * n, R_school))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = R_school
    
    start_time = time.time()
    deadline = 140
    
    # Try to find a lower rank solution using a structured search
    # We know R(2,3,3) is likely 14 or 15.
    
    # Candidate supports: 
    # A support is a triple (u_idx, v_idx, w_idx) 
    # u_idx in [0, n*m), v_idx in [0, m*p), w_idx in [0, p*n)
    # For a rank-1 term to be non-zero for T[i,j,k], 
    # u_idx must be i*m+j, v_idx must be j*p+k, w_idx must be k*n+i.
    # This means the support is "aligned" with the tensor indices.
    # However, in a general tensor decomposition, the supports can be arbitrary.
    # But for matrix multiplication, the "natural" supports are the basis elements.
    # Strassen's algorithm uses "mixed" supports.
    
    # We will try to find a set of R supports such that the system is solvable.
    # For each candidate support (u, v, w), it contributes to T[i,j,k] if 
    # u == i*m+j, v == j*p+k, w == k*n+i.
    # This is only true for the "diagonal" supports.
    # Off-diagonal supports (where u, v, w don't match the tensor indices directly)
    # are used in Strassen.
    # But wait, the tensor T is all-ones.
    # The condition for a rank-1 term (u, v, w) to contribute to T[i,j,k] is 
    # that the term is present in the decomposition.
    # The value is U[u] * V[v] * W[w].
    # We need sum_r U[u_r] V[v_r] W[w_r] delta_{u_r, i*m+j} delta_{v_r, j*p+k} delta_{w_r, k*n+i} = 1?
    # NO. The tensor decomposition is:
    # T[i,j,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r].
    # This is a constraint on the **entries** of U, V, W.
    # It does NOT restrict the "support" in the sense of which entries are non-zero.
    # U[i*m+j, r] can be non-zero for any r.
    # So the "support" concept applies to the **entries** of U, V, W.
    
    # The current ALS approach is actually correct in formulation.
    # The issue is convergence.
    
    # Let's try a different optimization: 
    # Minimize the error using scipy.optimize.least_squares on the flattened parameters.
    # This is a non-convex problem, but with a good initialization, it might work.
    
    # However, a more robust approach for small problems is to use 
    # "Iterative Proportional Fitting" or similar, but that's for non-negative.
    
    # Let's stick to ALS but improve the initialization and convergence criteria.
    # Also, try ranks 14, 15, 16, 17.
    
    for target_rank in range(14, R_school + 1):
        if time.time() - start_time > deadline:
            break
        
        found = False
        max_restarts = 50 if target_rank <= 15 else 20
        
        for restart in range(max_restarts):
            if time.time() - start_time > deadline:
                break
            
            np.random.seed(restart * 1000 + target_rank)
            
            # Initialize with random sparse matrices?
            # Or random dense?
            U = np.random.randn(n * m, target_rank)
            V = np.random.randn(m * p, target_rank)
            W = np.random.randn(p * n, target_rank)
            
            # Normalize
            for r in range(target_rank):
                n_u = np.linalg.norm(U[:, r])
                n_v = np.linalg.norm(V[:, r])
                n_w = np.linalg.norm(W[:, r])
                if n_u < 1e-10 or n_v < 1e-10 or n_w < 1e-10:
                    continue
                total_norm = (n_u * n_v * n_w) ** (1/3)
                U[:, r] /= (n_u / total_norm)
                V[:, r] /= (n_v / total_norm)
                W[:, r] /= (n_w / total_norm)
            
            # ALS with early stopping and error tracking
            prev_error = np.inf
            converged = False
            error = np.inf
            
            for iteration in range(300):
                if time.time() - start_time > deadline:
                    break
                
                # Update U
                for i in range(n):
                    for j in range(m):
                        A = np.zeros((p, target_rank))
                        for k in range(p):
                            A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                        try:
                            U[i * m + j, :] = np.linalg.lstsq(A.T, T[i, j, :].reshape(-1, 1), rcond=None)[0].flatten()
                        except:
                            pass
                
                # Update V
                for j in range(m):
                    for k in range(p):
                        B = np.zeros((n, target_rank))
                        for i in range(n):
                            B[i, :] = U[i * m + j, :] * W[k * n + i, :]
                        try:
                            V[j * p + k, :] = np.linalg.lstsq(B.T, T[:, j, k].reshape(-1, 1), rcond=None)[0].flatten()
                        except:
                            pass
                
                # Update W
                for k in range(p):
                    for i in range(n):
                        C = np.zeros((m, target_rank))
                        for j in range(m):
                            C[j, :] = U[i * m + j, :] * V[j * p + k, :]
                        try:
                            W[k * n + i, :] = np.linalg.lstsq(C.T, T[i, :, k].reshape(-1, 1), rcond=None)[0].flatten()
                        except:
                            pass
                
                # Compute Error
                error = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for r in range(target_rank):
                                val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                            error += (val - T[i, j, k]) ** 2
                error = np.sqrt(error)
                
                if error < 1e-8:
                    converged = True
                    break
                
                if error > prev_error * 1.01:
                    # Diverging or stuck
                    break
                prev_error = error
            
            if converged or error < 1e-6:
                # Verify
                max_err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for r in range(target_rank):
                                val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                            max_err = max(max_err, abs(val - T[i, j, k]))
                
                if max_err < 1e-5:
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    best_rank = target_rank
                    found = True
                    break
        
        if found:
            break
    
    # Round to half-integers
    def round_to_half(arr):
        rounded = np.round(arr * 2) / 2
        if np.max(np.abs(rounded - arr)) < 1e-4:
            return rounded
        return arr
    
    best_U = round_to_half(best_U)
    best_V = round_to_half(best_V)
    best_W = round_to_half(best_W)
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
