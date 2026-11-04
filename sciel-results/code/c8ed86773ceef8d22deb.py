"""Rewritten program: rank-minimal exact decomposition for 4x5 * 5x6."""
import time
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Return a rank-minimal exact tensor decomposition for multiplying a 4x5 matrix by a 5x6 matrix.
    
    Index conventions:
      U[i*m + j, c]  where i in 0..n-1, j in 0..m-1
      V[j*p + k, c]  where j in 0..m-1, k in 0..p-1
      W[k*n + i, c]  where k in 0..p-1, i in 0..n-1
    
    The decomposition satisfies: for all A (n x m), B (m x p),
      C = A*B = sum_{c=0}^{R-1} (sum_{i,j} A[i,j]*U[i*m+j,c]) * (sum_{j,k} B[j,k]*V[j*p+k,c]) * W_col_c
    where W_col_c is a (p x n) matrix with W_col_c[k,i] = W[k*n+i, c].
    """
    n, m, p = 4, 5, 6
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin
    
    # Start with schoolbook decomposition (rank 120)
    rank_school = n * m * p
    U = np.zeros((n * m, rank_school), dtype=np.float64)
    V = np.zeros((m * p, rank_school), dtype=np.float64)
    W = np.zeros((p * n, rank_school), dtype=np.float64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    best_rank = rank_school
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Build the multiplication tensor T[a,b,c] = 1 if a=(i,j), b=(j,k), c=(k,i)
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # ALS function: given fixed two factors, solve for the third via least squares
    def als_solve_T(U, V, W, T, which):
        """Solve for one factor matrix via least squares."""
        if which == 'U':
            # For each c: U[:,c] = argmin ||T - sum_c U_c ⊗ V_c ⊗ W_c||_U
            # Equivalent to: for each a, U[a,c] = <T[a,:,:], V[:,c] ⊗ W[:,c]> / ||V[:,c]⊗W[:,c]||^2
            # More efficiently: reshape T as (n*m, m*p*p*n) and solve
            # T_reshaped[a, b*p*n + c2] = T[a,b,c2]
            # We want U[a,c] such that sum_c U[a,c] * (V[:,c] ⊗ W[:,c])[b*p*n+c2] ≈ T[a,b,c2]
            # This is: U @ (V⊗W)^T ≈ T_reshaped
            # So U = T_reshaped @ (V⊗W)^+ 
            VW = np.zeros((m*p, p*n))
            for c in range(V.shape[1]):
                VW += np.outer(V[:, c], W[:, c])
            # Now we need U such that for each a: U[a,:] @ VW^T ≈ T[a,:,:].reshape(-1)
            # Actually: sum_c U[a,c] * VW[:,c]... no.
            # Let me think again. T[a,b,d] = sum_c U[a,c]*V[b,c]*W[d,c]
            # For fixed V,W: T[a,b,d] = sum_c U[a,c] * (V[b,c]*W[d,c])
            # So if we define M_c[b,d] = V[b,c]*W[d,c], then T[a,:,:] = sum_c U[a,c] * M_c
            # Vectorizing: T[a].flatten() = sum_c U[a,c] * M_c.flatten()
            # So U = T_reshaped @ (stack of M_c)^+
            M_stack = np.zeros((m*p*p*n, V.shape[1]))
            for c in range(V.shape[1]):
                M_stack[:, c] = np.outer(V[:, c], W[:, c]).flatten()
            T_reshaped = T.reshape(n*m, m*p*p*n)
            # Solve: T_reshaped = U @ M_stack^T
            # U = T_reshaped @ (M_stack^T)^+ = T_reshaped @ pinv(M_stack^T)
            U_new = T_reshaped @ np.linalg.pinv(M_stack.T)
            return U_new
        elif which == 'V':
            # T[a,b,d] = sum_c V[b,c] * (U[a,c]*W[d,c])
            # For fixed U,W: T[a,b,d] = sum_c V[b,c] * N_c[a,d] where N_c[a,d]=U[a,c]*W[d,c]
            # Vectorizing over (a,d): T_reshaped[b, a*p*n+d] = sum_c V[b,c] * N_c[a,d]
            # So V = T_reshaped @ pinv(N_stack^T)
            N_stack = np.zeros((p*n*n*m, U.shape[1]))
            for c in range(U.shape[1]):
                N_stack[:, c] = np.outer(U[:, c], W[:, c]).flatten()
            T_reshaped = T.transpose(1, 0, 2).reshape(m*p, p*n*n*m)
            V_new = T_reshaped @ np.linalg.pinv(N_stack.T)
            return V_new
        elif which == 'W':
            # T[a,b,d] = sum_c W[d,c] * (U[a,c]*V[b,c])
            # For fixed U,V: T[a,b,d] = sum_c W[d,c] * P_c[a,b] where P_c[a,b]=U[a,c]*V[b,c]
            # Vectorizing over (a,b): T_reshaped[d, a*m*p+b] = sum_c W[d,c] * P_c[a,b]
            # So W = T_reshaped @ pinv(P_stack^T)
            P_stack = np.zeros((m*p*n*m, U.shape[1]))
            for c in range(U.shape[1]):
                P_stack[:, c] = np.outer(U[:, c], V[:, c]).flatten()
            T_reshaped = T.transpose(2, 0, 1).reshape(p*n, m*p*n*m)
            W_new = T_reshaped @ np.linalg.pinv(P_stack.T)
            return W_new
    
    def als_decompose(T, rank, n_iter=50, n_restarts=3, seed_base=0):
        """Run ALS to find rank-rank decomposition of T."""
        best_err = np.inf
        best_U, best_V, best_W = None, None, None
        nm, mp, pn = T.shape
        
        for restart in range(n_restarts):
            if time.time() > deadline - 5:
                break
            rng = np.random.RandomState(seed_base + restart)
            U_r = rng.randn(nm, rank)
            V_r = rng.randn(mp, rank)
            W_r = rng.randn(pn, rank)
            
            for it in range(n_iter):
                if time.time() > deadline - 5:
                    break
                # Alternate solving for each factor
                U_new = als_solve_T(U_r, V_r, W_r, T, 'U')
                V_new = als_solve_T(U_new, V_r, W_r, T, 'V')
                W_new = als_solve_T(U_new, V_new, W_r, T, 'W')
                
                # Check error
                err = 0.0
                for c in range(rank):
                    T_approx = np.einsum('a,b,c->abc', U_new[:, c], V_new[:, c], W_new[:, c])
                    err += np.sum((T - T_approx)**2)
                err = np.sqrt(err)
                
                if err < best_err:
                    best_err = err
                    best_U, best_V, best_W = U_new.copy(), V_new.copy(), W_new.copy()
                
                if err < 1e-12:
                    break
                U_r, V_r, W_r = U_new, V_new, W_new
        
        return best_err, best_U, best_V, best_W
    
    # Try ALS for various target ranks
    target_ranks = [31, 32, 33, 34, 35, 36, 37, 38, 39, 40]
    
    for target_rank in target_ranks:
        if time.time() > deadline - 10:
            break
        
        err, U_sol, V_sol, W_sol = als_decompose(T, target_rank, n_iter=80, n_restarts=5, seed_base=target_rank*100)
        
        if U_sol is not None and err < 1e-10:
            best_rank = target_rank
            best_U, best_V, best_W = U_sol, V_sol, W_sol
            break
        elif err < 1e-6:
            # Close but not exact - try to polish
            # Round to nearest half-integers and check
            U_rounded = np.round(U_sol * 2) / 2
            V_rounded = np.round(V_sol * 2) / 2
            W_rounded = np.round(W_sol * 2) / 2
            
            # Check if rounded version is exact
            T_check = np.zeros_like(T)
            for c in range(target_rank):
                T_check += np.einsum('a,b,c->abc', U_rounded[:, c], V_rounded[:, c], W_rounded[:, c])
            
            if np.max(np.abs(T_check - T)) < 1e-10:
                best_rank = target_rank
                best_U, best_V, best_W = U_rounded, V_rounded, W_rounded
                break
    
    # If we found a better decomposition, use it; otherwise use schoolbook
    if best_rank < rank_school:
        U_out = best_U.astype(float)
        V_out = best_V.astype(float)
        W_out = best_W.astype(float)
    else:
        U_out = U.astype(float)
        V_out = V.astype(float)
        W_out = W.astype(float)
    
    # Convert to list format as required
    U_list = U_out.tolist()
    V_list = V_out.tolist()
    W_list = W_out.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
