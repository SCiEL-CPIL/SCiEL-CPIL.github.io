import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115
    
    # Schoolbook decomposition
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    # Build the 3-mode unfolding of the tensor
    dim_A = n * m
    dim_B = m * p
    dim_C = p * n
    
    # Unfold along mode 1: T_(1) is dim_A x (dim_B * dim_C)
    T_1 = np.zeros((dim_A, dim_B * dim_C), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_1[a, b * dim_C + c] = 1.0
    
    # Compute the rank of T_1
    try:
        U1, s1, Vt1 = np.linalg.svd(T_1, full_matrices=False)
        tol = 1e-10
        rank_T1 = int(np.sum(s1 > tol))
        
        if rank_T1 < rank_sb and time.time() < deadline:
            # Try to construct a decomposition of rank rank_T1
            Vt1_2d = Vt1[:rank_T1, :].reshape(rank_T1, dim_B, dim_C)
            
            # Check if each slice Vt1_2d[r, :, :] has rank 1
            all_rank_1 = True
            for r in range(rank_T1):
                slice_matrix = Vt1_2d[r, :, :]
                s_slice = np.linalg.svd(slice_matrix, compute_uv=False)
                if len(s_slice) > 1 and s_slice[1] > tol:
                    all_rank_1 = False
                    break
            
            if all_rank_1:
                # Construct the decomposition
                U_cols = []
                V_cols = []
                W_cols = []
                
                for r in range(rank_T1):
                    U_cols.append(U1[:, r] * s1[r])
                    slice_matrix = Vt1_2d[r, :, :]
                    u_slice, s_slice, vt_slice = np.linalg.svd(slice_matrix, full_matrices=False)
                    v_vec = u_slice[:, 0] * s_slice[0]
                    w_vec = vt_slice[0, :]
                    V_cols.append(v_vec)
                    W_cols.append(w_vec)
                
                U_arr = np.column_stack(U_cols)
                V_arr = np.column_stack(V_cols)
                W_arr = np.column_stack(W_cols)
                
                U_list = [[float(x) for x in row] for row in U_arr]
                V_list = [[float(x) for x in row] for row in V_arr]
                W_list = [[float(x) for x in row] for row in W_arr]
                
                # Verify the decomposition
                T_check = np.zeros((dim_A, dim_B, dim_C), dtype=float)
                for r in range(rank_T1):
                    for a in range(dim_A):
                        for b in range(dim_B):
                            for c in range(dim_C):
                                T_check[a, b, c] += U_list[a][r] * V_list[b][r] * W_list[c][r]
                
                error = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            a = i * m + j
                            b = j * p + k
                            c = k * n + i
                            if abs(T_check[a, b, c] - 1.0) > 1e-8:
                                error += 1
                
                if error == 0:
                    return {"U": U_list, "V": V_list, "W": W_list}
    except Exception:
        pass
    
    return {"U": U_sb, "V": V_sb, "W": W_sb}
# EVOLVE-BLOCK-END
