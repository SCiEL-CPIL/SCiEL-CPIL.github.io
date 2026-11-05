import numpy as np
import time
import random

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120  # 120s limit
    
    # Build the target tensor T for 3x3 by 3x7 multiplication
    # T[i, j, k] = 1 for all i in [0,3), j in [0,3), k in [0,7)
    # This represents C[i,k] = sum_j A[i,j] * B[j,k]
    
    # We'll use a constructive approach based on Strassen-like decompositions
    # For 3x3x3: rank 21 is achievable
    # For 3x3x7: we can extend the 3x3x3 result
    
    # First, let's try to find a rank-21 decomposition for the 3x3x3 core
    # Then extend to handle the 7 columns
    
    # Actually, let me try a direct ALS approach but with better initialization
    # and more efficient computation
    
    # Build unfolded matrices
    T_1 = np.zeros((n*m, (m*p)*(p*n)))
    T_2 = np.zeros((m*p, (n*m)*(p*n)))
    T_3 = np.zeros((p*n, (n*m)*(m*p)))
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                
                col_1 = idx_v * (p*n) + idx_w
                T_1[idx_u, col_1] = 1
                
                col_2 = idx_u * (p*n) + idx_w
                T_2[idx_v, col_2] = 1
                
                col_3 = idx_u * (m*p) + idx_v
                T_3[idx_w, col_3] = 1

    best_U, best_V, best_W = None, None, None
    best_rank = n * m * p  # 63
    
    # Try ranks from 21 upward (known lower bounds suggest 21 might work for 3x3x3)
    # For 3x3x7, the rank should be between 21 and 63
    
    for rank in range(21, 40):
        if time.time() > deadline - 30:
            break
            
        # Multiple restarts
        for restart in range(5):
            if time.time() > deadline - 30:
                    break
                    
            # Initialize with structured random
            U = np.random.randn(n*m, rank)
            V = np.random.randn(m*p, rank)
            W = np.random.randn(p*n, rank)
            
            # ALS steps with early stopping
            for step in range(30):
                if time.time() > deadline - 30:
                    break
                    
                # Solve for U: T_1 = U * (V ⊙ W)^T where ⊙ is Khatri-Rao
                M_vw = np.zeros(((m*p)*(p*n), rank))
                for c in range(rank):
                    M_vw[:, c] = np.kron(V[:, c], W[:, c])
                
                try:
                    U = np.linalg.lstsq(M_vw, T_1, rcond=None)[0].T
                except:
                    break
                    
                # Solve for V
                M_uw = np.zeros(((n*m)*(p*n), rank))
                for c in range(rank):
                    M_uw[:, c] = np.kron(U[:, c], W[:, c])
                try:
                    V = np.linalg.lstsq(M_uw, T_2, rcond=None)[0].T
                except:
                    break
                    
                # Solve for W
                M_uv = np.zeros(((n*m)*(m*p), rank))
                for c in range(rank):
                    M_uv[:, c] = np.kron(U[:, c], V[:, c])
                try:
                    W = np.linalg.lstsq(M_uv, T_3, rcond=None)[0].T
                except:
                    break
                    
            # Check error
            M_vw = np.zeros(((m*p)*(p*n), rank))
            for c in range(rank):
                M_vw[:, c] = np.kron(V[:, c], W[:, c])
            T_1_rec = U @ M_vw.T
            err = np.linalg.norm(T_1 - T_1_rec)
            
            if err < 1e-8:
                # Try to round to half-integers
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                # Verify exactness
                M_vw_r = np.zeros(((m*p)*(p*n), rank))
                for c in range(rank):
                    M_vw_r[:, c] = np.kron(V_rounded[:, c], W_rounded[:, c])
                T_1_rec_r = U_rounded @ M_vw_r.T
                err_r = np.linalg.norm(T_1 - T_1_rec_r)
                
                if err_r < 1e-8:
                    best_U, best_V, best_W = U_rounded, V_rounded, W_rounded
                    best_rank = rank
                    break  # Found lower rank, break inner loop
        
        if best_rank < n*m*p:
            break
    
    if best_U is not None:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
