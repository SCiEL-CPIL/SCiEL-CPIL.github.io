import numpy as np
import time
import random

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Build the target tensor T
    # T is of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    def _verify(U, V, W):
        # Check if U, V, W decompose T
        # Sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
        # Vectorized check
        # Reshape U to (n, m, R), V to (m, p, R), W to (p, n, R)
        R = U.shape[1]
        U_r = U.T.reshape(n, m, R)
        V_r = V.T.reshape(m, p, R)
        W_r = W.T.reshape(p, n, R)
        
        # Compute sum_r U[i,j,r] V[j,k,r] W[k,i,r]
        # This is a contraction over r
        # Result shape (n, m, p)
        # We can do:
        # temp = np.einsum('ijr,jkr->ikr', U_r, V_r) ? No.
        # We need sum_r U[i,j,r] V[j,k,r] W[k,i,r]
        # Let's use einsum: 'ijr,jkr,kir->ijk'
        computed = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
        
        # Target T_ijk is 1 for all i,j,k
        target = np.ones((n, m, p))
        
        return np.allclose(computed, target, atol=1e-6)

    def _als_decompose(R, max_iters=20, tol=1e-8):
        # Initialize U, V, W
        U = np.random.randn(n * m, R)
        V = np.random.randn(m * p, R)
        W = np.random.randn(p * n, R)
        
        for it in range(max_iters):
            # Update W
            # For each (k, i), solve for W[k*n+i, :]
            # Equation: sum_r W[k*n+i, r] (U[i*m+j, r] V[j*p+k, r]) = 1
            # This is a linear system for each (k, i)
            # Let A_r = U[i*m+j, r] V[j*p+k, r] for all j
            # We have m equations for each (k, i)
            # sum_r W[k*n+i, r] A_r = 1 (for all j)
            # This is: W_row @ A_vec = 1_vec
            # Where A_vec is of length R, 1_vec is of length m? No.
            # The equation is for each j.
            # So we have a system:
            # [ A_1(1), A_1(2), ..., A_1(m) ]
            # [ A_2(1), A_2(2), ..., A_2(m) ]
            # ...
            # [ A_m(1), A_m(2), ..., A_m(m) ]
            # Wait, the sum is over r.
            # Let's fix (k, i).
            # For each j, we have: sum_r W[k*n+i, r] U[i*m+j, r] V[j*p+k, r] = 1
            # Let M_jr = U[i*m+j, r] V[j*p+k, r]
            # Then sum_r W[k*n+i, r] M_jr = 1
            # This is a linear system M W_row = 1
            # M is of shape (m, R)
            # W_row is of shape (R,)
            # 1 is of shape (m,)
            
            W_new = np.zeros_like(W)
            for k in range(p):
                for i in range(n):
                    M = np.zeros((m, R))
                    for j in range(m):
                        M[j, :] = U[i * m + j, :] * V[j * p + k, :]
                    # Solve M W_row = 1
                    # Use least squares
                    try:
                        W_row, _, _, _ = np.linalg.lstsq(M, np.ones(m), rcond=None)
                        W_new[k * n + i, :] = W_row
                    except:
                        W_new[k * n + i, :] = W[k * n + i, :]
            W = W_new
            
            # Update U
            # Similarly for U
            U_new = np.zeros_like(U)
            for i in range(n):
                for j in range(m):
                    M = np.zeros((p, R))
                    for k in range(p):
                        M[k, :] = V[j * p + k, :] * W[k * n + i, :]
                    try:
                        U_row, _, _, _ = np.linalg.lstsq(M, np.ones(p), rcond=None)
                        U_new[i * m + j, :] = U_row
                    except:
                        U_new[i * m + j, :] = U[i * m + j, :]
            U = U_new
            
            # Update V
            V_new = np.zeros_like(V)
            for j in range(m):
                for k in range(p):
                    M = np.zeros((n, R))
                    for i in range(n):
                        M[i, :] = U[i * m + j, :] * W[k * n + i, :]
                    try:
                        V_row, _, _, _ = np.linalg.lstsq(M, np.ones(n), rcond=None)
                        V_new[j * p + k, :] = V_row
                    except:
                        V_new[j * p + k, :] = V[j * p + k, :]
            V = V_new
            
            # Check convergence
            if _verify(U, V, W):
                return U, V, W
                
        return None

    start_time = time.time()
    time_limit = 100.0 # Leave margin
    
    best_result = None
    best_rank = schoolbook_rank
    
    # Try ranks from schoolbook_rank - 1 down
    for R in range(schoolbook_rank - 1, 40, -1):
        if time.time() - start_time > time_limit:
            break
            
        # Try a few restarts
        found = False
        for restart in range(5):
            if time.time() - start_time > time_limit:
                break
                
            U, V, W = _als_decompose(R, max_iters=10)
            if U is not None and _verify(U, V, W):
                best_result = (U, V, W)
                best_rank = R
                found = True
                break
        
        if found:
            # We found a lower rank, we can stop or try to go lower
            # Given time, we might stop here to be safe
            # But let's try a bit lower if time permits
            if time.time() - start_time < time_limit - 10:
                continue
            else:
                break

    if best_result is not None:
        U, V, W = best_result
        return {"U": U, "V": V, "W": W}
    else:
        # Return schoolbook
        U_sb = np.zeros((n * m, schoolbook_rank))
        V_sb = np.zeros((m * p, schoolbook_rank))
        W_sb = np.zeros((p * n, schoolbook_rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j, col] = 1
                    V_sb[j * p + k, col] = 1
                    W_sb[k * n + i, col] = 1
                    col += 1
        return {"U": U_sb, "V": V_sb, "W": W_sb}
