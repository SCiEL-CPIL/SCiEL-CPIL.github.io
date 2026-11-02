import numpy as np
import time
import itertools
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    n, m, p = 2, 4, 6
    
    # Build the base rank-42 Strassen decomposition
    base_rank = 42
    U = [[0] * base_rank for _ in range(n * m)]
    V = [[0] * base_rank for _ in range(m * p)]
    W = [[0] * base_rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_block(A_idx, B_idx, C_idx, start_col):
        a, b, c, d = A_idx
        e, f, g, h = B_idx
        c11, c12, c21, c22 = C_idx
        
        # M1 = (a+d)(e+h)
        U[a][start_col] += 1; U[d][start_col] += 1
        V[e][start_col] += 1; V[h][start_col] += 1
        W[c11][start_col] += 1; W[c22][start_col] += 1
        start_col += 1
        
        # M2 = (c+d)e
        U[c][start_col] += 1; U[d][start_col] += 1
        V[e][start_col] += 1
        W[c21][start_col] += 1; W[c22][start_col] -= 1
        start_col += 1
        
        # M3 = a(f-h)
        U[a][start_col] += 1
        V[f][start_col] += 1; V[h][start_col] -= 1
        W[c12][start_col] += 1; W[c22][start_col] += 1
        start_col += 1
        
        # M4 = d(g-e)
        U[d][start_col] += 1
        V[g][start_col] += 1; V[e][start_col] -= 1
        W[c11][start_col] += 1; W[c21][start_col] += 1
        start_col += 1
        
        # M5 = (a+b)h
        U[a][start_col] += 1; U[b][start_col] += 1
        V[h][start_col] += 1
        W[c11][start_col] -= 1; W[c12][start_col] += 1
        start_col += 1
        
        # M6 = (c-a)(e+f)
        U[c][start_col] += 1; U[a][start_col] -= 1
        V[e][start_col] += 1; V[f][start_col] += 1
        W[c22][start_col] += 1
        start_col += 1
        
        # M7 = (b-d)(g+h)
        U[b][start_col] += 1; U[d][start_col] -= 1
        V[g][start_col] += 1; V[h][start_col] += 1
        W[c11][start_col] += 1
        start_col += 1
        
        return start_col
    
    for k in range(3):
        A_L_idx = [0, 1, 4, 5]
        B_Tk_idx = [0*6 + 2*k, 0*6 + 2*k + 1, 1*6 + 2*k, 1*6 + 2*k + 1]
        C_idx = [2*k*2 + 0, (2*k+1)*2 + 0, 2*k*2 + 1, (2*k+1)*2 + 1]
        col = add_strassen_block(A_L_idx, B_Tk_idx, C_idx, col)
        
        A_R_idx = [2, 3, 6, 7]
        B_Bk_idx = [2*6 + 2*k, 2*6 + 2*k + 1, 3*6 + 2*k, 3*6 + 2*k + 1]
        col = add_strassen_block(A_R_idx, B_Bk_idx, C_idx, col)
    
    # Now search for linear dependencies to reduce rank
    # The decomposition is valid if for all i, j, k:
    # sum_r U[i,r] * V[j,r] * W[k,r] = 1 if j == k, else 0
    # where i = a*m + b (row a, col b of A), j = b*p + c (row b, col c of B), k = c*n + a (row c, col a of C)
    
    # Convert to numpy arrays for efficient computation
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # Verify the base decomposition
    def verify(U_arr, V_arr, W_arr, n, m, p, tol=1e-10):
        for a in range(n):
            for c in range(p):
                for b in range(m):
                    i = a * m + b
                    j = b * p + c
                    k = c * n + a
                    val = 0.0
                    for r in range(U_arr.shape[1]):
                        val += U_arr[i, r] * V_arr[j, r] * W_arr[k, r]
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    # Search for dependencies: can we express one column as a linear combination of others?
    # For a column r to be removable, there must exist coefficients alpha_s (s != r)
    # such that for all (i, j, k): U[i,r]*V[j,r]*W[k,r] = sum_s alpha_s * U[i,s]*V[j,s]*W[k,s]
    
    # Build the "feature" vectors for each column
    # Each column r defines a function f_r(i,j,k) = U[i,r]*V[j,r]*W[k,r]
    # We want to find if any f_r is in the span of the others
    
    num_i = n * m  # 8
    num_j = m * p  # 24
    num_k = p * n  # 12
    total_features = num_i * num_j * num_k  # 2304
    
    # Build feature matrix: each row is a feature (i,j,k), each column is a rank term
    # F[i,j,k, r] = U[i,r] * V[j,r] * W[k,r]
    F = np.zeros((total_features, base_rank), dtype=np.float64)
    idx = 0
    for i in range(num_i):
        for j in range(num_j):
            for k in range(num_k):
                for r in range(base_rank):
                    F[idx, r] = U_arr[i, r] * V_arr[j, r] * W_arr[k, r]
                idx += 1
    
    # For each column r, check if it's in the span of the others
    # by solving the linear system F[:, S] * alpha = F[:, r]
    # where S = {0, ..., 41} \ {r}
    
    best_rank = base_rank
    best_indices = list(range(base_rank))
    
    # Try removing each single column
    for r in range(base_rank):
        if time.time() > deadline:
            break
        S = [s for s in range(base_rank) if s != r]
        F_S = F[:, S]
        F_r = F[:, r]
        
        # Check if F_r is in the column space of F_S
        # Use least squares
        try:
            alpha, residuals, rank_S, sv = np.linalg.lstsq(F_S, F_r, rcond=None)
            # Check if the residual is small
            reconstructed = F_S @ alpha
            residual_norm = np.linalg.norm(F_r - reconstructed)
            if residual_norm < 1e-10:
                # Column r is redundant!
                new_rank = base_rank - 1
                if new_rank < best_rank:
                    best_rank = new_rank
                    best_indices = S
        except:
            pass
    
    # Now try removing pairs of columns
    if time.time() < deadline:
        current_indices = list(range(base_rank))
        improved = True
        while improved and time.time() < deadline:
            improved = False
            new_indices = list(current_indices)
            for r_idx in range(len(new_indices)):
                if time.time() > deadline:
                    break
                r = new_indices[r_idx]
                S = [s for s in new_indices if s != r]
                if len(S) == 0:
                    continue
                F_S = F[:, S]
                F_r = F[:, r]
                
                try:
                    alpha, residuals, rank_S, sv = np.linalg.lstsq(F_S, F_r, rcond=None)
                    reconstructed = F_S @ alpha
                    residual_norm = np.linalg.norm(F_r - reconstructed)
                    if residual_norm < 1e-10:
                        new_indices = S
                        improved = True
                        break
                except:
                    pass
            if improved:
                current_indices = new_indices
                best_rank = len(current_indices)
    
    # Build the reduced matrices
    if best_rank < base_rank:
        U_final = [[U_arr[i, r] for r in best_indices] for i in range(n * m)]
        V_final = [[V_arr[j, r] for r in best_indices] for j in range(m * p)]
        W_final = [[W_arr[k, r] for r in best_indices] for k in range(p * n)]
    else:
        U_final = U
        V_final = V
        W_final = W
    
    # Verify the final result
    U_check = np.array(U_final, dtype=np.float64)
    V_check = np.array(V_final, dtype=np.float64)
    W_check = np.array(W_final, dtype=np.float64)
    
    if not verify(U_check, V_check, W_check, n, m, p):
        # Fall back to the original if verification fails
        U_final = U
        V_final = V
        W_final = W
    
    return {"U": U_final, "V": V_final, "W": W_final}
