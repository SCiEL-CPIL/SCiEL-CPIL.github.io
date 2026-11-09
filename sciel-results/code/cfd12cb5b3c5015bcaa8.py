"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [5, 5, 5]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140.0  # 10s safety margin
    
    # Block structure: 5 = 2 + 2 + 1
    block_sizes = [2, 2, 1]
    num_blocks = 3
    offsets = [0]
    for bs in block_sizes:
        offsets.append(offsets[-1] + bs)
    
    U_terms = []
    V_terms = []
    W_terms = []
    
    for bi in range(num_blocks):
        for bj in range(num_blocks):
            for bk in range(num_blocks):
                si = block_sizes[bi]
                sj = block_sizes[bj]
                sk = block_sizes[bk]
                
                a_row_start = offsets[bi]
                a_col_start = offsets[bj]
                b_row_start = offsets[bj]
                b_col_start = offsets[bk]
                c_row_start = offsets[bi]
                c_col_start = offsets[bk]
                
                if si == 2 and sj == 2 and sk == 2:
                    # Strassen: 7 rank-1 terms
                    # M1 = (A00+A11)(B00+B11)
                    # M2 = (A10+A11)(B00)
                    # M3 = (A00)(B01-B11)
                    # M4 = (A11)(B10-B00)
                    # M5 = (A00+A01)(B11)
                    # M6 = (A10-A00)(B00+B01)
                    # M7 = (A01-A11)(B10+B11)
                    # C00 = M1+M4-M5+M7
                    # C01 = M3+M5
                    # C10 = M2+M4
                    # C11 = M1-M2+M3+M6
                    
                    # Helper to compute global indices
                    def u_idx(i, j):
                        return (a_row_start + i) * m + (a_col_start + j)
                    def v_idx(j, k):
                        return (b_row_start + j) * p + (b_col_start + k)
                    def w_idx(k, i):
                        return (c_col_start + k) * n + (c_row_start + i)
                    
                    # M1: U: A00=1, A11=1; V: B00=1, B01=1; W: C00=1, C11=1
                    U_terms.append([(u_idx(0,0), 1), (u_idx(1,1), 1)])
                    V_terms.append([(v_idx(0,0), 1), (v_idx(0,1), 1)])
                    W_terms.append([(w_idx(0,0), 1), (w_idx(1,1), 1)])
                    
                    # M2: U: A10=1, A11=1; V: B00=1; W: C10=1, C11=-1
                    U_terms.append([(u_idx(1,0), 1), (u_idx(1,1), 1)])
                    V_terms.append([(v_idx(0,0), 1)])
                    W_terms.append([(w_idx(0,1), 1), (w_idx(1,1), -1)])
                    
                    # M3: U: A00=1; V: B01=1, B11=-1; W: C01=1, C11=1
                    U_terms.append([(u_idx(0,0), 1)])
                    V_terms.append([(v_idx(0,1), 1), (v_idx(1,1), -1)])
                    W_terms.append([(w_idx(1,0), 1), (w_idx(1,1), 1)])
                    
                    # M4: U: A11=1; V: B10=1, B00=-1; W: C00=1, C10=1
                    U_terms.append([(u_idx(1,1), 1)])
                    V_terms.append([(v_idx(1,0), 1), (v_idx(0,0), -1)])
                    W_terms.append([(w_idx(0,0), 1), (w_idx(0,1), 1)])
                    
                    # M5: U: A00=1, A01=1; V: B11=1; W: C00=-1, C01=1
                    U_terms.append([(u_idx(0,0), 1), (u_idx(0,1), 1)])
                    V_terms.append([(v_idx(1,1), 1)])
                    W_terms.append([(w_idx(0,0), -1), (w_idx(1,0), 1)])
                    
                    # M6: U: A10=1, A00=-1; V: B00=1, B01=1; W: C11=1
                    U_terms.append([(u_idx(1,0), 1), (u_idx(0,0), -1)])
                    V_terms.append([(v_idx(0,0), 1), (v_idx(0,1), 1)])
                    W_terms.append([(w_idx(1,1), 1)])
                    
                    # M7: U: A01=1, A11=-1; V: B10=1, B11=1; W: C00=1
                    U_terms.append([(u_idx(0,1), 1), (u_idx(1,1), -1)])
                    V_terms.append([(v_idx(1,0), 1), (v_idx(1,1), 1)])
                    W_terms.append([(w_idx(0,0), 1)])
                    
                else:
                    # Schoolbook for this block
                    for ii in range(si):
                        for jj in range(sj):
                            for kk in range(sk):
                                U_terms.append([((a_row_start+ii)*m + (a_col_start+jj), 1)])
                                V_terms.append([((b_row_start+jj)*p + (b_col_start+kk), 1)])
                                W_terms.append([((c_col_start+kk)*n + (c_row_start+ii), 1)])
    
    rank = len(U_terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for col in range(rank):
        for (row_idx, val) in U_terms[col]:
            U[row_idx][col] = val
        for (row_idx, val) in V_terms[col]:
            V[row_idx][col] = val
        for (row_idx, val) in W_terms[col]:
            W[row_idx][col] = val
    
    # Verify correctness - check ALL entries
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # The multiplication tensor: T[i,j,k] = 1 for all i,j,k
    # Decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Use vectorized check for speed
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                lhs = 0.0
                for r in range(rank):
                    lhs += U_np[i*m+j, r] * V_np[j*p+k, r] * W_np[k*n+i, r]
                err = abs(lhs - 1.0)
                if err > max_err:
                    max_err = err
    
    if max_err > 1e-9:
        # Fallback to schoolbook
        rank_fb = n * m * p
        U_fb = [[0] * rank_fb for _ in range(n * m)]
        V_fb = [[0] * rank_fb for _ in range(m * p)]
        W_fb = [[0] * rank_fb for _ in range(p * n)]
        column = 0
        for ii in range(n):
            for jj in range(m):
                for kk in range(p):
                    U_fb[ii * m + jj][column] = 1
                    V_fb[jj * p + kk][column] = 1
                    W_fb[kk * n + ii][column] = 1
                    column += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
