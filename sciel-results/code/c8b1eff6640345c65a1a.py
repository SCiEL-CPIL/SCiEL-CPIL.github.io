import numpy as np

def construct():
    n, m, p = 4, 4, 5

    # We'll build a rank-65 decomposition:
    # - 49 terms from recursive Strassen for 4x4x4
    # - 16 terms for the 5th column of B (k=4)

    U = np.zeros((n * m, 65), dtype=float)
    V = np.zeros((m * p, 65), dtype=float)
    W = np.zeros((p * n, 65), dtype=float)

    # Define Strassen 2x2 multiplications
    # Each multiplication is (u_dict, v_dict) where u_dict and v_dict map (row,col) -> coeff
    def strassen_2x2_mults():
        mults = []
        # M1 = (A00+A11)(B00+B11)
        mults.append((
            {(0,0): 1, (1,1): 1},
            {(0,0): 1, (1,1): 1}
        ))
        # M2 = (A10+A11)B00
        mults.append((
            {(1,0): 1, (1,1): 1},
            {(0,0): 1}
        ))
        # M3 = A00(B01-B11)
        mults.append((
            {(0,0): 1},
            {(0,1): 1, (1,1): -1}
        ))
        # M4 = A11(B10-B00)
        mults.append((
            {(1,1): 1},
            {(1,0): 1, (0,0): -1}
        ))
        # M5 = (A00+A01)B11
        mults.append((
            {(0,0): 1, (0,1): 1},
            {(1,1): 1}
        ))
        # M6 = (A10-A00)(B00+B01)
        mults.append((
            {(1,0): 1, (0,0): -1},
            {(0,0): 1, (0,1): 1}
        ))
        # M7 = (A01-A11)(B10+B11)
        mults.append((
            {(0,1): 1, (1,1): -1},
            {(1,0): 1, (1,1): 1}
        ))
        return mults

    # How each inner Strassen term contributes to C blocks
    # inner_contrib[t] = list of (row_in_2x2, col_in_2x2, coeff)
    inner_contrib = {
        0: [(0,0, 1), (1,1, 1)],      # M1 -> C00 and C11
        1: [(1,0, 1), (1,1, -1)],     # M2 -> C10 and C11
        2: [(0,1, 1), (1,1, 1)],      # M3 -> C01 and C11
        3: [(0,0, 1), (1,0, 1)],      # M4 -> C00 and C10
        4: [(0,0, -1), (0,1, 1)],     # M5 -> C00 and C01
        5: [(1,1, 1)],                # M6 -> C11
        6: [(0,0, 1)],                # M7 -> C00
    }

    # Which C blocks each outer term contributes to
    outer_block_map = {
        0: [('C00', 1), ('C11', 1)],   # M1
        1: [('C10', 1), ('C11', -1)],  # M2
        2: [('C01', 1), ('C11', 1)],   # M3
        3: [('C00', 1), ('C10', 1)],   # M4
        4: [('C00', -1), ('C01', 1)],  # M5
        5: [('C11', 1)],               # M6
        6: [('C00', 1)],               # M7
    }

    block_offsets = {
        'C00': (0, 0),
        'C01': (0, 2),
        'C10': (2, 0),
        'C11': (2, 2),
    }

    # Define the 7 outer Strassen terms for 4x4x4
    # Each outer term has U_j and V_j as 2x2 block matrices
    # where each block is a linear combination of 4x4 entries

    def make_outer_terms():
        outer_terms = []
        
        # Outer term 1: U1 = A00 + A11, V1 = B00 + B11
        U1_blocks = []
        V1_blocks = []
        for r in range(2):
            for c in range(2):
                U1_blocks.append({(r,c): 1, (r+2,c+2): 1})
                V1_blocks.append({(r,c): 1, (r+2,c+2): 1})
        outer_terms.append((U1_blocks, V1_blocks))
        
        # Outer term 2: U2 = A10 + A11, V2 = B00
        U2_blocks = []
        V2_blocks = []
        for r in range(2):
            for c in range(2):
                U2_blocks.append({(r+2,c): 1, (r+2,c+2): 1})
                V2_blocks.append({(r,c): 1})
        outer_terms.append((U2_blocks, V2_blocks))
        
        # Outer term 3: U3 = A00, V3 = B01 - B11
        U3_blocks = []
        V3_blocks = []
        for r in range(2):
            for c in range(2):
                U3_blocks.append({(r,c): 1})
                V3_blocks.append({(r,c+2): 1, (r+2,c+2): -1})
        outer_terms.append((U3_blocks, V3_blocks))
        
        # Outer term 4: U4 = A11, V4 = B10 - B00
        U4_blocks = []
        V4_blocks = []
        for r in range(2):
            for c in range(2):
                U4_blocks.append({(r+2,c+2): 1})
                V4_blocks.append({(r+2,c): 1, (r,c): -1})
        outer_terms.append((U4_blocks, V4_blocks))
        
        # Outer term 5: U5 = A00 + A01, V5 = B11
        U5_blocks = []
        V5_blocks = []
        for r in range(2):
            for c in range(2):
                U5_blocks.append({(r,c): 1, (r,c+2): 1})
                V5_blocks.append({(r+2,c+2): 1})
        outer_terms.append((U5_blocks, V5_blocks))
        
        # Outer term 6: U6 = A10 - A00, V6 = B00 + B01
        U6_blocks = []
        V6_blocks = []
        for r in range(2):
            for c in range(2):
                U6_blocks.append({(r+2,c): 1, (r,c): -1})
                V6_blocks.append({(r,c): 1, (r,c+2): 1})
        outer_terms.append((U6_blocks, V6_blocks))
        
        # Outer term 7: U7 = A01 - A11, V7 = B10 + B11
        U7_blocks = []
        V7_blocks = []
        for r in range(2):
            for c in range(2):
                U7_blocks.append({(r,c+2): 1, (r+2,c+2): -1})
                V7_blocks.append({(r+2,c): 1, (r+2,c+2): 1})
        outer_terms.append((U7_blocks, V7_blocks))
        
        return outer_terms

    def expand_2x2_to_scalars(U_mat, V_mat):
        """
        U_mat and V_mat are 2x2 arrays where each element is a dict mapping (row,col) -> coeff
        Returns list of 7 (u_scalar, v_scalar) pairs
        """
        strassen_mults = strassen_2x2_mults()
        results = []
        for u_dict, v_dict in strassen_mults:
            u_scalar = {}
            v_scalar = {}
            for (ur, uc), coeff_u in u_dict.items():
                for (ai, aj), coeff_ai in U_mat[ur][uc].items():
                    key = (ai, aj)
                    u_scalar[key] = u_scalar.get(key, 0) + coeff_u * coeff_ai
            for (vr, vc), coeff_v in v_dict.items():
                for (bi, bj), coeff_bi in V_mat[vr][vc].items():
                    key = (bi, bj)
                    v_scalar[key] = v_scalar.get(key, 0) + coeff_v * coeff_bi
            u_scalar = {k: v for k, v in u_scalar.items() if v != 0}
            v_scalar = {k: v for k, v in v_scalar.items() if v != 0}
            results.append((u_scalar, v_scalar))
        return results

    outer_terms = make_outer_terms()
    col_idx = 0

    for j in range(7):
        U_blocks, V_blocks = outer_terms[j]
        # Convert to 2x2 matrix of dicts
        U_mat = [[U_blocks[r*2+c] for c in range(2)] for r in range(2)]
        V_mat = [[V_blocks[r*2+c] for c in range(2)] for r in range(2)]
        
        inner_mults = expand_2x2_to_scalars(U_mat, V_mat)
        
        for t in range(7):
            u_scalar, v_scalar = inner_mults[t]
            
            # Build W for this term
            w_dict = {}
            for block_name, block_coeff in outer_block_map[j]:
                row_off, col_off = block_offsets[block_name]
                for mr, mc, inner_coeff in inner_contrib[t]:
                    c_row = row_off + mr
                    c_col = col_off + mc
                    coeff = block_coeff * inner_coeff
                    w_idx = c_col * n + c_row
                    w_dict[w_idx] = w_dict.get(w_idx, 0) + coeff
            
            # Fill U, V, W
            for (ai, aj), coeff in u_scalar.items():
                U[ai * m + aj, col_idx] += coeff
            
            for (bi, bj), coeff in v_scalar.items():
                V[bi * p + bj, col_idx] += coeff
            
            for w_idx, coeff in w_dict.items():
                W[w_idx, col_idx] += coeff
            
            col_idx += 1

    # 16 terms for the 5th column of B (k=4)
    for i in range(n):
        for j in range(m):
            U[i * m + j, col_idx] = 1.0
            V[j * p + 4, col_idx] = 1.0
            W[4 * n + i, col_idx] = 1.0
            col_idx += 1

    # Verify the decomposition
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
        
        if not np.allclose(T, 1.0, atol=1e-8):
            return False
        return True

    if verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to schoolbook
        schoolbook_rank = n * m * p
        U_sb = np.zeros((n * m, schoolbook_rank), dtype=float)
        V_sb = np.zeros((m * p, schoolbook_rank), dtype=float)
        W_sb = np.zeros((p * n, schoolbook_rank), dtype=float)
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j, col] = 1.0
                    V_sb[j * p + k, col] = 1.0
                    W_sb[k * n + i, col] = 1.0
                    col += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
