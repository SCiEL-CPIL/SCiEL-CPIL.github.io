"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=4, m=5, p=6
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_strassen_block(i0, j0, k0):
        """
        Adds a Strassen decomposition for a 2x2x2 block.
        i0, j0, k0 are the top-left indices of the block.
        """
        # Indices for A (n x m)
        # A00, A01, A10, A11
        u_00 = i0 * m + j0
        u_01 = i0 * m + (j0 + 1)
        u_10 = (i0 + 1) * m + j0
        u_11 = (i0 + 1) * m + (j0 + 1)
        
        # Indices for B (m x p)
        # B00, B01, B10, B11
        v_00 = j0 * p + k0
        v_01 = j0 * p + (k0 + 1)
        v_10 = (j0 + 1) * p + k0
        v_11 = (j0 + 1) * p + (k0 + 1)
        
        # Indices for C (p x n)
        # C00, C01, C10, C11
        # Note: C[k, i]
        w_00 = k0 * n + i0
        w_01 = k0 * n + (i0 + 1)
        w_10 = (k0 + 1) * n + i0
        w_11 = (k0 + 1) * n + (i0 + 1)
        
        # M1 = (A00 + A11) * (B00 + B11)
        # U: 1 at u_00, 1 at u_11
        # V: 1 at v_00, 1 at v_11
        # W: 1 at w_00, 1 at w_11
        U_cols.append({u_00: 1, u_11: 1})
        V_cols.append({v_00: 1, v_11: 1})
        W_cols.append({w_00: 1, w_11: 1})
        
        # M2 = (A10 + A11) * B00
        U_cols.append({u_10: 1, u_11: 1})
        V_cols.append({v_00: 1})
        W_cols.append({w_10: 1}) # C10 = M2 + ...
        
        # M3 = A00 * (B10 - B11)
        U_cols.append({u_00: 1})
        V_cols.append({v_10: 1, v_11: -1})
        W_cols.append({w_01: 1}) # C01 = M3 + ...
        
        # M4 = A11 * (B01 - B00)
        U_cols.append({u_11: 1})
        V_cols.append({v_01: 1, v_00: -1})
        W_cols.append({w_00: 1, w_10: 1}) # C00 += M4, C10 += M4
        
        # M5 = (A00 + A10) * B11
        U_cols.append({u_00: 1, u_10: 1})
        V_cols.append({v_11: 1})
        W_cols.append({w_00: -1, w_01: 1}) # C00 -= M5, C01 += M5
        
        # M6 = (A10 - A00) * (B00 + B01)
        U_cols.append({u_10: 1, u_00: -1})
        V_cols.append({v_00: 1, v_01: 1})
        W_cols.append({w_11: 1}) # C11 += M6
        
        # M7 = (A01 - A11) * (B10 + B11)
        U_cols.append({u_01: 1, u_11: -1})
        V_cols.append({v_10: 1, v_11: 1})
        W_cols.append({w_00: 1, w_11: 1}) # C00 += M7, C11 += M7

    def add_schoolbook_block(i0, j0, k0, di, dj, dk):
        """
        Adds a schoolbook decomposition for a di x dj x dk block.
        i0, j0, k0 are the top-left indices.
        """
        for i in range(di):
            for j in range(dj):
                for k in range(dk):
                    u_idx = (i0 + i) * m + (j0 + j)
                    v_idx = (j0 + j) * p + (k0 + k)
                    w_idx = (k0 + k) * n + (i0 + i)
                    
                    U_cols.append({u_idx: 1})
                    V_cols.append({v_idx: 1})
                    W_cols.append({w_idx: 1})

    # Partitions
    # i: 0:2, 2:4
    # j: 0:2, 2:4, 4:5
    # k: 0:2, 2:4, 4:6
    
    i_slices = [(0, 2), (2, 4)]
    j_slices = [(0, 2), (2, 4), (4, 5)]
    k_slices = [(0, 2), (2, 4), (4, 6)]
    
    for i0, di in i_slices:
        for j0, dj in j_slices:
            for k0, dk in k_slices:
                if di == 2 and dj == 2 and dk == 2:
                    add_strassen_block(i0, j0, k0)
                else:
                    add_schoolbook_block(i0, j0, k0, di, dj, dk)
                    
    # Convert dict lists to matrices
    rank = len(U_cols)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c in range(rank):
        for r, val in U_cols[c].items():
            U[r][c] = val
        for r, val in V_cols[c].items():
            V[r][c] = val
        for r, val in W_cols[c].items():
            W[r][c] = val
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
