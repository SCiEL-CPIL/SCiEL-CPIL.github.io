import numpy as np
import time
def _strassen_2x2x2_terms():
    """
    Returns the 7 rank-1 terms for 2x2x2 matrix multiplication.
    Each term is a tuple (u, v, w) where u, v, w are dicts mapping indices to coefficients.
    Indices are 0..3 for 2x2 matrices.
    U index: i*2+j
    V index: j*2+k
    W index: k*2+i
    """
    terms = []
    # M1 = (A00+A11)(B00+B11) -> C00+=1, C11+=1
    terms.append(({0:1, 3:1}, {0:1, 1:1, 2:1, 3:1}, {0:1, 3:1}))
    # M2 = (A10+A11)B00 -> C10+=1, C11+=-1
    terms.append(({2:1, 3:1}, {0:1}, {1:1, 3:-1}))
    # M3 = A00(B01-B11) -> C01+=1, C11+=1
    terms.append(({0:1}, {1:1, 3:-1}, {2:1, 3:1}))
    # M4 = A11(B10-B00) -> C10+=1, C00+=1
    terms.append(({3:1}, {2:1, 0:-1}, {1:1, 0:1}))
    # M5 = (A00+A01)B11 -> C01+=1, C00+=-1
    terms.append(({0:1, 1:1}, {3:1}, {2:1, 0:-1}))
    # M6 = (A10-A00)(B00+B01) -> C11+=1
    terms.append(({2:1, 0:-1}, {0:1, 1:1}, {3:1}))
    # M7 = (A01-A11)(B10+B11) -> C00+=1
    terms.append(({1:1, 3:-1}, {2:1, 3:1}, {0:1}))
    return terms

def _build_strassen_4x4x4():
    """
    Builds U, V, W for 4x4x4 matrix multiplication using Strassen.
    Rank = 49.
    """
    n = 4
    rank = 49
    U = [[0] * rank for _ in range(n*n)]
    V = [[0] * rank for _ in range(n*n)]
    W = [[0] * rank for _ in range(n*n)]
    
    terms_2x2 = _strassen_2x2x2_terms()
    
    # Outer Strassen terms for 4x4x4
    # A = [[A11, A12], [A21, A22]]
    # B = [[B11, B12], [B21, B22]]
    # C = A*B
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # Block indices: 0=11, 1=12, 2=21, 3=22
    outer_specs = [
        ({0:1, 3:1}, {0:1, 3:1}, {0:1, 3:1}), # M1 -> C11, C22
        ({2:1, 3:1}, {0:1}, {2:1, 3:-1}),      # M2 -> C21, C22
        ({0:1}, {1:1, 3:-1}, {1:1, 3:1}),      # M3 -> C12, C22
        ({3:1}, {2:1, 0:-1}, {2:1, 0:1}),      # M4 -> C21, C11
        ({0:1, 1:1}, {3:1}, {1:1, 0:-1}),      # M5 -> C12, C11
        ({2:1, 0:-1}, {0:1, 1:1}, {3:1}),      # M6 -> C22
        ({1:1, 3:-1}, {2:1, 3:1}, {0:1}),      # M7 -> C11
    ]
    
    col = 0
    for a_coeff, b_coeff, c_coeff in outer_specs:
        for u_2, v_2, w_2 in terms_2x2:
            # Expand each 2x2 term into 4x4 terms
            # For each local index in u_2, v_2, w_2:
            # Map local (li, lj) to global indices based on a_coeff/b_coeff/c_coeff blocks
            
            # Helper to get global indices for a block
            def get_block_indices(block_id, local_i, local_j, matrix_type):
                row_start = (block_id // 2) * 2
                col_start = (block_id % 2) * 2
                if matrix_type == 'A' or matrix_type == 'B':
                    return (row_start + local_i) * 4 + (col_start + local_j)
                else: # W
                    i = row_start + local_i
                    k = col_start + local_j
                    return k * 4 + i
            
            # Process U
            for li in range(2):
                for lj in range(2):
                    local_idx = li * 2 + lj
                    if local_idx in u_2:
                        val = u_2[local_idx]
                        for b_id, coeff in a_coeff.items():
                            g_idx = get_block_indices(b_id, li, lj, 'A')
                            U[g_idx][col] += val * coeff
            
            # Process V
            for lj in range(2): # local j
                for lk in range(2): # local k
                    local_idx = lj * 2 + lk
                    if local_idx in v_2:
                        val = v_2[local_idx]
                        for b_id, coeff in b_coeff.items():
                            g_idx = get_block_indices(b_id, lj, lk, 'B')
                            V[g_idx][col] += val * coeff
            
            # Process W
            for li in range(2): # local i
                for lk in range(2): # local k
                    local_idx = lk * 2 + li # W index k*2+i
                    if local_idx in w_2:
                        val = w_2[local_idx]
                        for b_id, coeff in c_coeff.items():
                            g_idx = get_block_indices(b_id, li, lk, 'W')
                            W[g_idx][col] += val * coeff
            
            col += 1
            
    return U, V, W, rank

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = 4, 4, 8
    
    # Default: Schoolbook
    best_rank = n * m * p
    best_U = [[0] * best_rank for _ in range(n * m)]
    best_V = [[0] * best_rank for _ in range(m * p)]
    best_W = [[0] * best_rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i * m + j][column] = 1
                best_V[j * p + k][column] = 1
                best_W[k * n + i][column] = 1
                column += 1
                
    try:
        # Build Strassen 4x4x4
        U4, V4, W4, rank4 = _build_strassen_4x4x4()
        
        # Verify 4x4x4
        valid = True
        for _ in range(5):
            A = np.random.randint(-10, 10, (4, 4))
            B = np.random.randint(-10, 10, (4, 4))
            C_exp = A @ B
            C_comp = np.zeros((4, 4))
            for r in range(rank4):
                u = np.array(U4)[:, r]
                v = np.array(V4)[:, r]
                w = np.array(W4)[:, r]
                # C[i,k] += sum_j U[i,j,r] * V[j,k,r] * W[k,i,r]
                # Vectorized:
                # u is 16x1, reshape to 4x4
                u_mat = u.reshape(4, 4)
                v_mat = v.reshape(4, 4)
                w_mat = w.reshape(4, 4) # Note: w is indexed k*4+i, so reshape(4,4) gives W[k,i]
                # We need sum_j u_mat[i,j] * v_mat[j,k] * w_mat[k,i]
                # This is not a simple matrix product.
                # C[i,k] = sum_j (u_mat[i,j] * v_mat[j,k]) * w_mat[k,i]
                # Let T = u_mat @ v_mat (4x4)
                # C[i,k] = T[i,k] * w_mat[k,i]
                T = u_mat @ v_mat
                C_comp += T * w_mat.T # w_mat[k,i] is w_mat.T[i,k]
                
            if not np.allclose(C_comp, C_exp, atol=1e-9):
                valid = False
                break
                
        if valid:
            # Build 4x4x8
            rank_total = 2 * rank4
            U_full = [[0] * rank_total for _ in range(16)]
            V_full = [[0] * rank_total for _ in range(32)]
            W_full = [[0] * rank_total for _ in range(32)]
            
            # Left half
            for i in range(16):
                for r in range(rank4):
                    U_full[i][r] = U4[i][r]
            for j in range(4):
                for k in range(4):
                    for r in range(rank4):
                        V_full[j*8+k][r] = V4[j*4+k][r]
            for k in range(4):
                for i in range(4):
                    for r in range(rank4):
                        W_full[k*4+i][r] = W4[k*4+i][r]
                        
            # Right half
            for i in range(16):
                for r in range(rank4):
                    U_full[i][rank4+r] = U4[i][r]
            for j in range(4):
                for k in range(4):
                    for r in range(rank4):
                        V_full[j*8+(k+4)][rank4+r] = V4[j*4+k][r]
            for k in range(4):
                for i in range(4):
                    for r in range(rank4):
                        W_full[(k+4)*4+i][rank4+r] = W4[k*4+i][r]
                        
            # Verify 4x4x8
            valid_full = True
            for _ in range(5):
                A = np.random.randint(-10, 10, (4, 4))
                B = np.random.randint(-10, 10, (4, 8))
                C_exp = A @ B
                C_comp = np.zeros((4, 8))
                for r in range(rank_total):
                    u = np.array(U_full)[:, r]
                    v = np.array(V_full)[:, r]
                    w = np.array(W_full)[:, r]
                    
                    u_mat = u.reshape(4, 4)
                    v_mat = v.reshape(4, 8)
                    w_mat = w.reshape(8, 4) # w is k*4+i, so reshape(8,4) gives W[k,i]
                    
                    T = u_mat @ v_mat # 4x8
                    # C[i,k] = T[i,k] * w_mat[k,i]
                    C_comp += T * w_mat.T # w_mat[k,i] is w_mat.T[i,k]
                    
                if not np.allclose(C_comp, C_exp, atol=1e-9):
                    valid_full = False
                    break
                    
            if valid_full and rank_total < best_rank:
                best_rank = rank_total
                best_U = U_full
                best_V = V_full
                best_W = W_full
                
    except Exception:
        pass
        
    if time.time() > deadline:
        return {"U": best_U, "V": best_V, "W": best_W}
        
    return {"U": best_U, "V": best_V, "W": best_W}
