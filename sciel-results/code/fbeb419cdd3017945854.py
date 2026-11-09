import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = 4, 4, 5
    
    def time_remaining():
        return deadline - time.time()

    def build_target_tensor():
        T = np.zeros((n, m, p), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] = 1.0
        return T

    TARGET_T = build_target_tensor()

    def verify(U, V, W):
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)

        if U_arr.ndim != 2 or V_arr.ndim != 2 or W_arr.ndim != 2:
            return False

        rank = U_arr.shape[1]
        if V_arr.shape[1] != rank or W_arr.shape[1] != rank:
            return False

        U_3d = U_arr.reshape(n, m, rank)
        V_3d = V_arr.reshape(m, p, rank)
        W_3d = W_arr.reshape(p, n, rank)

        T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)

        if not np.allclose(T, TARGET_T, atol=1e-6):
            return False
        return True

    def get_schoolbook():
        schoolbook_rank = n * m * p
        U = np.zeros((n * m, schoolbook_rank), dtype=float)
        V = np.zeros((m * p, schoolbook_rank), dtype=float)
        W = np.zeros((p * n, schoolbook_rank), dtype=float)
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W

    # Build rank-65 decomposition:
    # 49 terms from recursive Strassen for 4x4x4 (cols 0..3)
    # 16 terms for the 5th column (k=4)
    
    U = np.zeros((n * m, 65), dtype=float)
    V = np.zeros((m * p, 65), dtype=float)
    W = np.zeros((p * n, 65), dtype=float)
    
    # Strassen 2x2: 7 multiplications
    # C = A * B where A, B are 2x2
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For 4x4x4: partition into 2x2 blocks
    # A rows 0-1,2-3; cols 0-1,2-3
    # B rows 0-1,2-3; cols 0-1,2-3
    # C rows 0-1,2-3; cols 0-1,2-3
    
    # The 7 outer Strassen terms each produce a 2x2 result.
    # Each 2x2 multiplication uses 7 scalar multiplications (inner Strassen).
    # Total: 7 * 7 = 49 scalar multiplications.
    
    def strassen_2x2_mults():
        mults = []
        u1 = {(0,0): 1, (1,1): 1}
        v1 = {(0,0): 1, (1,1): 1}
        mults.append((u1, v1))
        
        u2 = {(1,0): 1, (1,1): 1}
        v2 = {(0,0): 1}
        mults.append((u2, v2))
        
        u3 = {(0,0): 1}
        v3 = {(0,1): 1, (1,1): -1}
        mults.append((u3, v3))
        
        u4 = {(1,1): 1}
        v4 = {(1,0): 1, (0,0): -1}
        mults.append((u4, v4))
        
        u5 = {(0,0): 1, (0,1): 1}
        v5 = {(1,1): 1}
        mults.append((u5, v5))
        
        u6 = {(1,0): 1, (0,0): -1}
        v6 = {(0,0): 1, (0,1): 1}
        mults.append((u6, v6))
        
        u7 = {(0,1): 1, (1,1): -1}
        v7 = {(1,0): 1, (1,1): 1}
        mults.append((u7, v7))
        
        return mults

    def expand_2x2_to_scalars(u_matrix, v_matrix):
        strassen_mults = strassen_2x2_mults()
        results = []
        for u_dict, v_dict in strassen_mults:
            u_scalar = {}
            v_scalar = {}
            for (ur, uc), coeff_u in u_dict.items():
                for (ai, aj), coeff_ai in u_matrix[ur][uc].items():
                    key = (ai, aj)
                    u_scalar[key] = u_scalar.get(key, 0) + coeff_u * coeff_ai
            for (vr, vc), coeff_v in v_dict.items():
                for (bi, bj), coeff_bi in v_matrix[vr][vc].items():
                    key = (bi, bj)
                    v_scalar[key] = v_scalar.get(key, 0) + coeff_v * coeff_bi
            u_scalar = {k: v for k, v in u_scalar.items() if v != 0}
            v_scalar = {k: v for k, v in v_scalar.items() if v != 0}
            results.append((u_scalar, v_scalar))
        return results

    U1_blocks = []
    for r in range(2):
        for c in range(2):
            U1_blocks.append({(r,c): 1, (r+2,c+2): 1})
    V1_blocks = []
    for r in range(2):
        for c in range(2):
            V1_blocks.append({(r,c): 1, (r+2,c+2): 1})

    U2_blocks = []
    for r in range(2):
        for c in range(2):
            U2_blocks.append({(r+2,c): 1, (r+2,c+2): 1})
    V2_blocks = []
    for r in range(2):
        for c in range(2):
            V2_blocks.append({(r,c): 1})

    U3_blocks = []
    for r in range(2):
        for c in range(2):
            U3_blocks.append({(r,c): 1})
    V3_blocks = []
    for r in range(2):
        for c in range(2):
            V3_blocks.append({(r,c+2): 1, (r+2,c+2): -1})

    U4_blocks = []
    for r in range(2):
        for c in range(2):
            U4_blocks.append({(r+2,c+2): 1})
    V4_blocks = []
    for r in range(2):
        for c in range(2):
            V4_blocks.append({(r+2,c): 1, (r,c): -1})

    U5_blocks = []
    for r in range(2):
        for c in range(2):
            U5_blocks.append({(r,c): 1, (r,c+2): 1})
    V5_blocks = []
    for r in range(2):
        for c in range(2):
            V5_blocks.append({(r+2,c+2): 1})

    U6_blocks = []
    for r in range(2):
        for c in range(2):
            U6_blocks.append({(r+2,c): 1, (r,c): -1})
    V6_blocks = []
    for r in range(2):
        for c in range(2):
            V6_blocks.append({(r,c): 1, (r,c+2): 1})

    U7_blocks = []
    for r in range(2):
        for c in range(2):
            U7_blocks.append({(r,c+2): 1, (r+2,c+2): -1})
    V7_blocks = []
    for r in range(2):
        for c in range(2):
            V7_blocks.append({(r+2,c): 1, (r+2,c+2): 1})

    inner_contrib = {
        0: [(0,0, 1), (1,1, 1)],      
        1: [(1,0, 1), (1,1, -1)],     
        2: [(0,1, 1), (1,1, 1)],      
        3: [(0,0, 1), (1,0, 1)],      
        4: [(0,0, -1), (0,1, 1)],     
        5: [(1,1, 1)],                
        6: [(0,0, 1)],                
    }

    block_offsets = {
        'C00': (0, 0),
        'C01': (0, 2),
        'C10': (2, 0),
        'C11': (2, 2),
    }

    outer_block_map = {
        0: [('C00', 1), ('C11', 1)],
        1: [('C10', 1), ('C11', -1)],
        2: [('C01', 1), ('C11', 1)],
        3: [('C00', 1), ('C10', 1)],
        4: [('C00', -1), ('C01', 1)],
        5: [('C11', 1)],
        6: [('C00', 1)],
    }

    outer_terms = [
        (U1_blocks, V1_blocks),
        (U2_blocks, V2_blocks),
        (U3_blocks, V3_blocks),
        (U4_blocks, V4_blocks),
        (U5_blocks, V5_blocks),
        (U6_blocks, V6_blocks),
        (U7_blocks, V7_blocks),
    ]

    col_idx = 0
    for j in range(7):
        U_blocks, V_blocks = outer_terms[j]
        U_mat = [[U_blocks[r*2+c] for c in range(2)] for r in range(2)]
        V_mat = [[V_blocks[r*2+c] for c in range(2)] for r in range(2)]
        
        inner_mults = expand_2x2_to_scalars(U_mat, V_mat)
        
        for t in range(7):
            u_scalar, v_scalar = inner_mults[t]
            
            w_dict = {}
            for block_name, block_coeff in outer_block_map[j]:
                row_off, col_off = block_offsets[block_name]
                for mr, mc, inner_coeff in inner_contrib[t]:
                    c_row = row_off + mr
                    c_col = col_off + mc
                    coeff = block_coeff * inner_coeff
                    w_idx = c_col * n + c_row
                    w_dict[w_idx] = w_dict.get(w_idx, 0) + coeff
            
            for (ai, aj), coeff in u_scalar.items():
                U[ai * m + aj, col_idx] += coeff
            
            for (bi, bj), coeff in v_scalar.items():
                V[bi * p + bj, col_idx] += coeff
            
            for w_idx, coeff in w_dict.items():
                W[w_idx, col_idx] += coeff
            
            col_idx += 1

    for i in range(n):
        for j in range(m):
            U[i * m + j, col_idx] = 1.0
            V[j * p + 4, col_idx] = 1.0
            W[4 * n + i, col_idx] = 1.0
            col_idx += 1

    # Check for redundant terms (columns where U, V, W are all zero or where the term contributes nothing)
    # Try to find a lower rank by checking if the decomposition is valid
    if verify(U, V, W):
        # Try to reduce rank by removing zero columns
        active = []
        for c in range(65):
            if not (np.allclose(U[:, c], 0) and np.allclose(V[:, c], 0) and np.allclose(W[:, c], 0)):
                active.append(c)
        
        if len(active) < 65:
            U_red = U[:, active]
            V_red = V[:, active]
            W_red = W[:, active]
            if verify(U_red, V_red, W_red):
                return {"U": U_red.tolist(), "V": V_red.tolist(), "W": W_red.tolist()}
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        U_sb, V_sb, W_sb = get_schoolbook()
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
