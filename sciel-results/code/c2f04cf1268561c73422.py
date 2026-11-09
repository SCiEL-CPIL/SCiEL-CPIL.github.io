"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [5, 5, 5]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140.0  # 10s safety margin
    
    block_sizes = [2, 2, 1]
    num_blocks = 3
    
    # Precompute block offsets
    block_offsets = [0]
    for s in block_sizes:
        block_offsets.append(block_offsets[-1] + s)
    
    # Build the decomposition term by term
    # Each term is (U_entries, V_entries, W_entries) where each is a list of (index, coeff)
    terms = []
    
    for bi in range(num_blocks):
        for bj in range(num_blocks):
            for bk in range(num_blocks):
                si = block_sizes[bi]
                sj = block_sizes[bj]
                sk = block_sizes[bk]
                
                a_row0 = block_offsets[bi]
                a_col0 = block_offsets[bj]
                b_row0 = block_offsets[bj]
                b_col0 = block_offsets[bk]
                c_row0 = block_offsets[bi]
                c_col0 = block_offsets[bk]
                
                if si == 2 and sj == 2 and sk == 2:
                    # Strassen: 7 rank-1 terms
                    # Local indices: A(ii,jj), B(jj,kk), C(ii,kk) for ii,jj,kk in {0,1}
                    # Global: A(a_row0+ii, a_col0+jj), B(b_row0+jj, b_col0+kk), C(c_row0+ii, c_col0+kk)
                    
                    # Helper: U index for A(ii,jj) = (a_row0+ii)*5 + (a_col0+jj)
                    # Helper: V index for B(jj,kk) = (b_row0+jj)*5 + (b_col0+kk)
                    # Helper: W index for C(ii,kk) = (c_col0+kk)*5 + (c_row0+ii)
                    
                    def u_idx(ii, jj):
                        return (a_row0 + ii) * 5 + (a_col0 + jj)
                    
                    def v_idx(jj, kk):
                        return (b_row0 + jj) * 5 + (b_col0 + kk)
                    
                    def w_idx(ii, kk):
                        return (c_col0 + kk) * 5 + (c_row0 + ii)
                    
                    # M1 = (A00+A11)(B00+B11)
                    u1 = [(u_idx(0,0), 1), (u_idx(1,1), 1)]
                    v1 = [(v_idx(0,0), 1), (v_idx(0,1), 1)]
                    w1 = [(w_idx(0,0), 1), (w_idx(1,1), 1)]
                    terms.append((u1, v1, w1))
                    
                    # M2 = (A10+A11)(B00)
                    u2 = [(u_idx(1,0), 1), (u_idx(1,1), 1)]
                    v2 = [(v_idx(0,0), 1)]
                    w2 = [(w_idx(1,0), 1)]
                    terms.append((u2, v2, w2))
                    
                    # M3 = (A00)(B01-B11)
                    u3 = [(u_idx(0,0), 1)]
                    v3 = [(v_idx(0,1), 1), (v_idx(1,1), -1)]
                    w3 = [(w_idx(0,1), 1)]
                    terms.append((u3, v3, w3))
                    
                    # M4 = (A11)(B10-B00)
                    u4 = [(u_idx(1,1), 1)]
                    v4 = [(v_idx(1,0), 1), (v_idx(0,0), -1)]
                    w4 = [(w_idx(1,0), 1)]
                    terms.append((u4, v4, w4))
                    
                    # M5 = (A00+A01)(B11)
                    u5 = [(u_idx(0,0), 1), (u_idx(0,1), 1)]
                    v5 = [(v_idx(1,1), 1)]
                    w5 = [(w_idx(0,0), -1), (w_idx(0,1), 1)]
                    terms.append((u5, v5, w5))
                    
                    # M6 = (A10-A00)(B00+B01)
                    u6 = [(u_idx(1,0), 1), (u_idx(0,0), -1)]
                    v6 = [(v_idx(0,0), 1), (v_idx(0,1), 1)]
                    w6 = [(w_idx(1,1), 1)]
                    terms.append((u6, v6, w6))
                    
                    # M7 = (A01-A11)(B10+B11)
                    u7 = [(u_idx(0,1), 1), (u_idx(1,1), -1)]
                    v7 = [(v_idx(1,0), 1), (v_idx(1,1), 1)]
                    w7 = [(w_idx(0,0), 1)]
                    terms.append((u7, v7, w7))
                    
                else:
                    # Schoolbook for mixed blocks
                    for ii in range(si):
                        for jj in range(sj):
                            for kk in range(sk):
                                gi = a_row0 + ii
                                gj = a_col0 + jj
                                gk = b_col0 + kk
                                u_entry = (gi * 5 + gj, 1)
                                v_entry = (gj * 5 + gk, 1)
                                w_entry = (gk * 5 + gi, 1)
                                terms.append(([u_entry], [v_entry], [w_entry]))
    
    rank = len(terms)
    
    # Build the factor matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r, (u_entries, v_entries, w_entries) in enumerate(terms):
        for (idx, coeff) in u_entries:
            U[idx][r] = coeff
        for (idx, coeff) in v_entries:
            V[idx][r] = coeff
        for (idx, coeff) in w_entries:
            W[idx][r] = coeff
    
    # Verify correctness
    if time.time() - start_time < deadline - 5:
        A = np.random.randint(-2, 3, size=(n, m))
        B = np.random.randint(-2, 3, size=(m, p))
        C_expected = A @ B
        
        U_arr = np.array(U)
        V_arr = np.array(V)
        W_arr = np.array(W)
        
        C_computed = np.zeros((n, p))
        for r in range(rank):
            u_col = U_arr[:, r]
            v_col = V_arr[:, r]
            w_col = W_arr[:, r]
            
            # U index: i*m + j, so reshape to (n, m)
            A_u = u_col.reshape(n, m)
            # V index: j*p + k, so reshape to (m, p)
            B_v = v_col.reshape(m, p)
            # W index: k*n + i, so reshape to (p, n)
            C_w = w_col.reshape(p, n)
            
            # C[i,k] = sum_j A_u[i,j] * B_v[j,k] * C_w[k,i]
            # = (A_u @ B_v)[i,k] * C_w[k,i]
            AB = A_u @ B_v
            C_computed += AB * C_w.T
        
        if not np.allclose(C_computed, C_expected):
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
# EVOLVE-BLOCK-END
