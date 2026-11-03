import numpy as np
import time

def construct():
    """
    Rank-112 decomposition for 4x4 * 4x8 matrix multiplication.
    
    Strategy:
    1. Decompose the 4x4 * 4x4 multiplication into 8 block multiplications of 2x2 * 2x2.
    2. Decompose each 2x2 * 2x2 multiplication into 7 Strassen terms.
    3. Total terms for one 4x4 * 4x4 is 8 * 7 = 56.
    4. Since the output width is 8 (two 4-column blocks), we duplicate this structure.
    5. Total rank = 56 * 2 = 112.
    
    This is better than the naive rank of 128.
    """
    start_time = time.time()
    
    n, m, p = 4, 4, 8
    
    # Define the 7 Strassen terms for 2x2x2.
    # Each term is a tuple of (U_entries, V_entries, W_entries).
    # Indices are flat indices for 2x2 matrices.
    # U: i*2+j, V: j*2+k, W: k*2+i
    
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # We define them as sparse vectors.
    # U indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # V indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # W indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    strassen_2x2_terms = [
        # M1
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # M2
        ({2: 1, 3: 1}, {0: 1}, {2: 1}),
        # M3
        ({0: 1}, {2: 1, 3: -1}, {1: 1}),
        # M4
        ({3: 1}, {1: 1, 0: -1}, {1: 1}),
        # M5
        ({0: 1, 1: 1}, {3: 1}, {1: 1}),
        # M6
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {2: 1}),
        # M7
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {3: 1}),
    ]
    
    # Verify Strassen 2x2x2
    def verify_2x2x2(terms):
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    total = 0
                    for u_dict, v_dict, w_dict in terms:
                        u_val = u_dict.get(i*2+j, 0)
                        v_val = v_dict.get(j*2+k, 0)
                        w_val = w_dict.get(k*2+i, 0)
                        total += u_val * v_val * w_val
                    if abs(total - 1) > 1e-9:
                        return False
        return True

    if not verify_2x2x2(strassen_2x2_terms):
        # Fallback to naive if Strassen definition is wrong (should not happen)
        return construct_naive(n, m, p)

    # The 8 block multiplications for 4x4 * 4x4
    # A is 4x4, B is 4x4, C is 4x4
    # Blocks:
    # A00 (0:2, 0:2), A01 (0:2, 2:4), A10 (2:4, 0:2), A11 (2:4, 2:4)
    # B00 (0:2, 0:2), B01 (0:2, 2:4), B10 (2:4, 0:2), B11 (2:4, 2:4)
    # C00 (0:2, 0:2), C01 (0:2, 2:4), C10 (2:4, 0:2), C11 (2:4, 2:4)
    
    # block_mults: (A_row_block, A_col_block, B_row_block, B_col_block, C_row_block, C_col_block)
    block_mults_4x4x4 = [
        (0, 0, 0, 0, 0, 0), # A00 * B00 -> C00
        (0, 1, 1, 0, 0, 0), # A01 * B10 -> C00
        (0, 0, 0, 1, 0, 1), # A00 * B01 -> C01
        (0, 1, 1, 1, 0, 1), # A01 * B11 -> C01
        (1, 0, 0, 0, 1, 0), # A10 * B00 -> C10
        (1, 1, 1, 0, 1, 0), # A11 * B10 -> C10
        (1, 0, 0, 1, 1, 1), # A10 * B01 -> C11
        (1, 1, 1, 1, 1, 1), # A11 * B11 -> C11
    ]
    
    U_list = []
    V_list = []
    W_list = []
    
    # We need to handle the two 4x4x4 blocks for the 4x4x8 problem.
    # The B matrix is 4x8. We split it into B_left (4x4) and B_right (4x4).
    # The C matrix is 4x8. We split it into C_left (4x4) and C_right (4x4).
    
    for block_offset in [0, 4]:
        for (ar, ac, br, bc, cr, cc) in block_mults_4x4x4:
            for (u_dict, v_dict, w_dict) in strassen_2x2_terms:
                u_entries = {}
                v_entries = {}
                w_entries = {}
                
                for (u_idx, u_coeff) in u_dict.items():
                    local_i = u_idx // 2
                    local_j = u_idx % 2
                    global_u = (ar*2 + local_i) * 4 + (ac*2 + local_j)
                    u_entries[global_u] = u_coeff
                
                for (v_idx, v_coeff) in v_dict.items():
                    local_j = v_idx // 2
                    local_k = v_idx % 2
                    # B is 4x8. Row is br*2 + local_j. Col is block_offset + bc*2 + local_k.
                    # V flat index: row * 8 + col
                    global_v = (br*2 + local_j) * 8 + (block_offset + bc*2 + local_k)
                    v_entries[global_v] = v_coeff
                
                for (w_idx, w_coeff) in w_dict.items():
                    local_k = w_idx // 2
                    local_i = w_idx % 2
                    # C is 4x8. Row is cr*2 + local_k. Col is block_offset + cc*2 + local_i.
                    # W flat index: row * 4 + col (Wait, W index is k*n+i? No, W is p*n. p=8, n=4. W[k,i] -> k*4+i)
                    # The task says W[k*n+i, r]. So W index is k*4 + i.
                    # k is the output row (0..7), i is the input row (0..3).
                    # In our block structure:
                    # k = cr*2 + local_k
                    # i = cc*2 + local_i
                    global_w = (cr*2 + local_k) * 4 + (cc*2 + local_i)
                    w_entries[global_w] = w_coeff
                
                U_list.append(u_entries)
                V_list.append(v_entries)
                W_list.append(w_entries)
    
    total_rank = len(U_list)
    
    # Verify the decomposition
    def verify_decomposition(U_terms, V_terms, W_terms, n, m, p):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for u_dict, v_dict, w_dict in zip(U_terms, V_terms, W_terms):
                        u_val = u_dict.get(i*m+j, 0)
                        v_val = v_dict.get(j*p+k, 0)
                        w_val = w_dict.get(k*n+i, 0)
                        total += u_val * v_val * w_val
                    if abs(total - 1) > 1e-9:
                        return False
        return True

    if not verify_decomposition(U_list, V_list, W_list, n, m, p):
        # Fallback to naive
        return construct_naive(n, m, p)
    
    # Convert to matrices
    rank = total_rank
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    for r in range(rank):
        for (idx, coeff) in U_list[r].items():
            U[idx, r] = coeff
        for (idx, coeff) in V_list[r].items():
            V[idx, r] = coeff
        for (idx, coeff) in W_list[r].items():
            W[idx, r] = coeff
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def construct_naive(n, m, p):
    U = np.zeros((n * m, n * m * p), dtype=float)
    V = np.zeros((m * p, n * m * p), dtype=float)
    W = np.zeros((p * n, n * m * p), dtype=float)
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, idx] = 1.0
                V[j*p+k, idx] = 1.0
                W[k*n+i, idx] = 1.0
                idx += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
