"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

import numpy as np
import time
import random
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 135  # 15s safety margin before 150s kill
    
    U_dim = n * m  # 8
    V_dim = m * p  # 24
    W_dim = p * n  # 12
    
    # Build the target tensor T of shape (U_dim, V_dim, W_dim)
    # T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k
    T = np.zeros((U_dim, V_dim, W_dim), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
                
    # Schoolbook solution as fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(U_dim)]
    V_school = [[0] * rank_school for _ in range(V_dim)]
    W_school = [[0] * rank_school for _ in range(W_dim)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    def verify(U_mat, V_mat, W_mat, tol=1e-10):
        """Check if U,V,W exactly decompose T (within tol)."""
        for c in range(W_dim):
            recon = np.zeros((U_dim, V_dim), dtype=np.float64)
            for r in range(U_mat.shape[1]):
                recon += U_mat[:, r] * V_mat[:, r] * W_mat[r, c]
            if np.max(np.abs(recon - T[:, :, c])) > tol:
                return False
        return True

    # Strassen decomposition for 2x2x2
    # u, v, w are vectors of length 4 (flattened 2x2 matrices)
    # Order: 00, 01, 10, 11
    strassen_terms = [
        # (u, v, w)
        (np.array([1, 0, 0, 1]), np.array([1, 0, 0, 1]), np.array([1, 0, 0, 1])),
        (np.array([0, 0, 1, 1]), np.array([1, 0, 0, 0]), np.array([0, 0, 1, -1])),
        (np.array([1, 0, 0, 0]), np.array([0, 1, 0, -1]), np.array([0, 1, 0, 1])),
        (np.array([0, 0, 0, 1]), np.array([0, -1, 1, 0]), np.array([1, 0, 1, 0])),
        (np.array([1, 1, 0, 0]), np.array([0, 0, 0, 1]), np.array([-1, 1, 0, 0])),
        (np.array([0, -1, 1, 0]), np.array([1, 1, 0, 0]), np.array([0, 0, 0, 1])),
        (np.array([0, 1, -1, 0]), np.array([0, 0, 1, 1]), np.array([1, 0, 0, 0])),
    ]
    
    # We use 3 blocks of 2 columns for B, so 3 * 14 = 42 rank
    # Each block uses 2 Strassen decompositions (one for A_L B_T, one for A_R B_B)
    rank_strassen = 42
    U_mat = np.zeros((U_dim, rank_strassen), dtype=np.float64)
    V_mat = np.zeros((V_dim, rank_strassen), dtype=np.float64)
    W_mat = np.zeros((W_dim, rank_strassen), dtype=np.float64)
    
    col_idx = 0
    for b in range(3):
        # b corresponds to column block 2b, 2b+1
        for prod_idx in range(2):
            # prod_idx 0: A_L B_T
            # prod_idx 1: A_R B_B
            for u_loc, v_loc, w_loc in strassen_terms:
                # Map u_loc to global U
                # A_L uses global indices 0,1,4,5 (A[0,0], A[0,1], A[1,0], A[1,1])
                # A_R uses global indices 2,3,6,7 (A[0,2], A[0,3], A[1,2], A[1,3])
                if prod_idx == 0:
                    # Local 0->Global 0, 1->1, 2->4, 3->5
                    U_mat[0, col_idx] += u_loc[0]
                    U_mat[1, col_idx] += u_loc[1]
                    U_mat[4, col_idx] += u_loc[2]
                    U_mat[5, col_idx] += u_loc[3]
                else:
                    # Local 0->Global 2, 1->3, 2->6, 3->7
                    U_mat[2, col_idx] += u_loc[0]
                    U_mat[3, col_idx] += u_loc[1]
                    U_mat[6, col_idx] += u_loc[2]
                    U_mat[7, col_idx] += u_loc[3]
                
                # Map v_loc to global V
                # B_T uses rows 0,1 and cols 2b, 2b+1
                # B_B uses rows 2,3 and cols 2b, 2b+1
                # Global V index = j * p + k
                if prod_idx == 0:
                    # Local 0->B[0, 2b], 1->B[0, 2b+1], 2->B[1, 2b], 3->B[1, 2b+1]
                    V_mat[0 * p + 2*b, col_idx] += v_loc[0]
                    V_mat[0 * p + 2*b+1, col_idx] += v_loc[1]
                    V_mat[1 * p + 2*b, col_idx] += v_loc[2]
                    V_mat[1 * p + 2*b+1, col_idx] += v_loc[3]
                else:
                    # Local 0->B[2, 2b], 1->B[2, 2b+1], 2->B[3, 2b], 3->B[3, 2b+1]
                    V_mat[2 * p + 2*b, col_idx] += v_loc[0]
                    V_mat[2 * p + 2*b+1, col_idx] += v_loc[1]
                    V_mat[3 * p + 2*b, col_idx] += v_loc[2]
                    V_mat[3 * p + 2*b+1, col_idx] += v_loc[3]
                
                # Map w_loc to global W
                # W index = k * n + i
                # Local 0->C[0, 2b], 1->C[0, 2b+1], 2->C[1, 2b], 3->C[1, 2b+1]
                # C[i, k] -> W[k * n + i]
                W_mat[2*b * n + 0, col_idx] += w_loc[0]
                W_mat[(2*b+1) * n + 0, col_idx] += w_loc[1]
                W_mat[2*b * n + 1, col_idx] += w_loc[2]
                W_mat[(2*b+1) * n + 1, col_idx] += w_loc[3]
                
                col_idx += 1
    
    # Verify the constructed solution
    if verify(U_mat, V_mat, W_mat, tol=1e-9):
        U_list = [[float(U_mat[idx, c]) for c in range(rank_strassen)] for idx in range(U_dim)]
        V_list = [[float(V_mat[idx, c]) for c in range(rank_strassen)] for idx in range(V_dim)]
        W_list = [[float(W_mat[idx, c]) for c in range(rank_strassen)] for idx in range(W_dim)]
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        # Fallback to schoolbook
        return {"U": U_school, "V": V_school, "W": W_school}
# EVOLVE-BLOCK-END
