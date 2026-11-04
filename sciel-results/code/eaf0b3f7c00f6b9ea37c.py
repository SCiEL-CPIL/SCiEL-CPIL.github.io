"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 7
    # Split m=3 into m1=2, m2=1
    # Part 1: (2x2)*(2x7), split p=7 into 2+2+2+1, use Strassen for (2x2)*(2x2) blocks
    # Part 2: (2x1)*(1x7), rank 14
    
    # Strassen's 7 terms for 2x2*2x2:
    # C00 = M1 - M4 + M6 + M7
    # C01 = M2 + M4
    # C10 = M3 + M5
    # C11 = M1 - M3 + M2 + M5
    # where:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B10-B11)
    # M4 = A11*(B01-B10)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B10)
    # M7 = (A01-A11)*(B01+B11)
    
    # For (2x2)*(2x2), the tensor decomposition has 7 terms.
    # Each term is: U factor (2x2 output), V factor (2x2 A), W factor (2x2 B)
    
    # Strassen terms: (U_row, U_col, V_rows, W_cols, coeff)
    # Actually, let me define the 7 rank-1 tensors for 2x2*2x2.
    # The multiplication: C[i,j] = sum_k A[i,k]*B[k,j]
    # U is indexed by (i,j) for C, V by (i,k) for A, W by (k,j) for B
    
    # Strassen's 7 multiplications:
    # M1: (A00+A11) * (B00+B11)
    #   U contribution: to C00 (+1), C11 (-1)... wait let me be careful.
    
    # Standard Strassen:
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b10-b11)
    # M4 = a11(b01-b10)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b10)
    # M7 = (a01-a11)(b01+b11)
    #
    # c00 = M1 - M4 + M6 + M7
    # c01 = M2 + M4
    # c10 = M3 + M5
    # c11 = M1 - M3 + M2 + M5
    
    # Each Mk is a product of a linear form in A and a linear form in B.
    # For the tensor decomposition, each rank-1 term corresponds to:
    # U vector (in C-space), V vector (in A-space), W vector (in B-space)
    
    # Let me define the 7 terms as (U_coeffs, V_coeffs, W_coeffs)
    # C-space: indices (i,j) -> flat index i*2+j: 0,1,2,3 for (0,0),(0,1),(1,0),(1,1)
    # A-space: indices (i,k) -> flat index i*2+k: 0,1,2,3 for (0,0),(0,1),(1,0),(1,1)
    # B-space: indices (k,j) -> flat index k*2+j: 0,1,2,3 for (0,0),(0,1),(1,0),(1,1)
    
    # M1: V = (1,0,0,1) [a00+a11], W = (1,0,0,1) [b00+b11]
    #   Contribution to C: c00 += M1, c11 -= M1
    #   U = (1, 0, 0, -1)
    
    # M2: V = (0,0,1,1) [a10+a11], W = (1,0,0,0) [b00]
    #   Contribution: c01 += M2, c11 += M2
    #   U = (0, 1, 0, 1)
    
    # M3: V = (1,0,0,0) [a00], W = (0,0,1,-1) [b10-b11]
    #   Contribution: c10 += M3, c11 -= M3
    #   U = (0, 0, 1, -1)
    
    # M4: V = (0,0,0,1) [a11], W = (0,1,-1,0) [b01-b10]
    #   Contribution: c00 -= M4, c01 += M4
    #   U = (-1, 1, 0, 0)
    
    # M5: V = (1,1,0,0) [a00+a01], W = (0,0,0,1) [b11]
    #   Contribution: c10 += M5, c11 += M5
    #   U = (0, 0, 1, 1)
    
    # M6: V = (0,0,1,-1) [a10-a00], W = (1,0,1,0) [b00+b10]
    #   Contribution: c00 += M6
    #   U = (1, 0, 0, 0)
    
    # M7: V = (0,1,-1,0) [a01-a11], W = (0,1,0,1) [b01+b11]
    #   Contribution: c00 += M7
    #   U = (1, 0, 0, 0)
    
    # Wait, c00 = M1 - M4 + M6 + M7, so M6 and M7 both contribute +1 to c00.
    # But M6 U = (1,0,0,0) and M7 U = (1,0,0,0)? That can't be right for a rank-1 decomposition.
    # Each rank-1 term should have its own unique contribution pattern.
    
    # Let me redo: In the tensor decomposition, each rank-1 term is U⊗V⊗W where
    # T = sum_r U_r ⊗ V_r ⊗ W_r
    # T[c_idx, a_idx, b_idx] = sum_r U_r[c_idx] * V_r[a_idx] * W_r[b_idx]
    # And T should equal the multiplication tensor.
    
    # For Strassen, the 7 terms are:
    # r=0 (M1): V=(1,0,0,1), W=(1,0,0,1), U=(1,0,0,-1)
    # r=1 (M2): V=(0,0,1,1), W=(1,0,0,0), U=(0,1,0,1)  
    # r=2 (M3): V=(1,0,0,0), W=(0,0,1,-1), U=(0,0,1,-1)
    # r=3 (M4): V=(0,0,0,1), W=(0,1,-1,0), U=(-1,1,0,0)
    # r=4 (M5): V=(1,1,0,0), W=(0,0,0,1), U=(0,0,1,1)
    # r=5 (M6): V=(0,0,1,-1), W=(1,0,1,0), U=(1,0,0,0)
    # r=6 (M7): V=(0,1,-1,0), W=(0,1,0,1), U=(1,0,0,0)
    
    # Verify M6: U=(1,0,0,0) means only c00 gets contribution. V=(0,0,1,-1)=a10-a00, W=(1,0,1,0)=b00+b10.
    # So c00 += (a10-a00)(b00+b10). From Strassen: c00 = M1-M4+M6+M7. Yes.
    
    # Verify M7: U=(1,0,0,0) means only c00. V=(0,1,-1,0)=a01-a11, W=(0,1,0,1)=b01+b11.
    # c00 += (a01-a11)(b01+b11). Yes.
    
    # But wait, both M6 and M7 have U=(1,0,0,0). That's fine - they're different rank-1 terms
    # because their V and W are different.
    
    strassen_terms = [
        # (U, V, W) for 2x2*2x2
        ([1, 0, 0, -1], [1, 0, 0, 1], [1, 0, 0, 1]),   # M1
        ([0, 1, 0, 1],  [0, 0, 1, 1], [1, 0, 0, 0]),   # M2
        ([0, 0, 1, -1], [1, 0, 0, 0], [0, 0, 1, -1]),  # M3
        ([-1, 1, 0, 0], [0, 0, 0, 1], [0, 1, -1, 0]),  # M4
        ([0, 0, 1, 1],  [1, 1, 0, 0], [0, 0, 0, 1]),   # M5
        ([1, 0, 0, 0],  [0, 0, 1, -1], [1, 0, 1, 0]),  # M6
        ([1, 0, 0, 0],  [0, 1, -1, 0], [0, 1, 0, 1]),  # M7
    ]
    
    # Now build the full decomposition for (2x3)*(3x7)
    # Split m=3 into m1=2 (rows 0,1 of A, rows 0,1 of B) and m2=1 (row 2 of A, row 2 of B)
    # Part 1: (2x2)*(2x7) using the first 2 rows/cols
    # Part 2: (2x1)*(1x7) using row 2
    
    # Full dimensions:
    # C: 2x7, flat index i*7+k, total 14
    # A: 2x3, flat index i*3+j, total 6
    # B: 3x7, flat index j*7+k, total 21
    
    U_list = []
    V_list = []
    W_list = []
    
    # Part 1: (2x2)*(2x7), split p=7 into blocks of size 2,2,2,1
    # For each (2x2)*(2xq) sub-problem, we need to decompose it.
    # For (2x2)*(2x2): use Strassen (7 terms)
    # For (2x2)*(2x1): schoolbook (4 terms)
    
    # (2x2)*(2x1): C[i] = A[i,0]*B[0,0] + A[i,1]*B[1,0]
    # C-space: 2 dims (i=0,1), A-space: 4 dims (i*2+k), B-space: 2 dims (k*1+0=k)
    # T[c, a, b] = 1 if c=i, a=i*2+k, b=k
    # Terms:
    # (i=0,k=0): U=(1,0), V=(1,0,0,0), W=(1,0)
    # (i=0,k=1): U=(1,0), V=(0,1,0,0), W=(0,1)
    # (i=1,k=0): U=(0,1), V=(0,0,1,0), W=(1,0)
    # (i=1,k=1): U=(0,1), V=(0,0,0,1), W=(0,1)
    
    def make_2x2x2_block(col_offset):
        """Return U,V,W vectors for (2x2)*(2x2) block at column offset in the 7-col space."""
        terms = []
        for (u, v, w) in strassen_terms:
            # u is 4-dim (C 2x2), v is 4-dim (A 2x2), w is 4-dim (B 2x2)
            # Map to full spaces:
            # C: i*7+k, for i in 0,1 and k in col_offset..col_offset+1
            u_full = [0] * 14
            for ci in range(2):
                for ck in range(2):
                    u_full[ci * 7 + (col_offset + ck)] = u[ci * 2 + ck]
            
            # A: i*3+j, for i in 0,1 and j in 0,1 (first 2 cols of A)
            v_full = [0] * 6
            for ai in range(2):
                for aj in range(2):
                    v_full[ai * 3 + aj] = v[ai * 2 + aj]
            
            # B: j*7+k, for j in 0,1 and k in col_offset..col_offset+1
            w_full = [0] * 21
            for bj in range(2):
                for bk in range(2):
                    w_full[bj * 7 + (col_offset + bk)] = w[bj * 2 + bk]
            
            terms.append((u_full, v_full, w_full))
        return terms
    
    def make_2x2x1_block(col_offset):
        """Return U,V,W vectors for (2x2)*(2x1) block at column offset."""
        base_terms = [
            ([1, 0], [1, 0, 0, 0], [1, 0]),
            ([1, 0], [0, 1, 0, 0], [0, 1]),
            ([0, 1], [0, 0, 1, 0], [1, 0]),
            ([0, 1], [0, 0, 0, 1], [0, 1]),
        ]
        terms = []
        for (u, v, w) in base_terms:
            u_full = [0] * 14
            for ci in range(2):
                u_full[ci * 7 + col_offset] = u[ci]
            
            v_full = [0] * 6
            for ai in range(2):
                for aj in range(2):
                    v_full[ai * 3 + aj] = v[ai * 2 + aj]
            
            w_full = [0] * 21
            for bj in range(2):
                w_full[bj * 7 + col_offset] = w[bj]
            
            terms.append((u_full, v_full, w_full))
        return terms
    
    # Split p=7 into 2+2+2+1
    col_offsets = [0, 2, 4, 6]
    block_sizes = [2, 2, 2, 1]
    
    for col_off, bs in zip(col_offsets, block_sizes):
        if bs == 2:
            terms = make_2x2x2_block(col_off)
        else:
            terms = make_2x2x1_block(col_off)
        for (u, v, w) in terms:
            U_list.append(u)
            V_list.append(v)
            W_list.append(w)
    
    # Part 2: (2x1)*(1x7), using A row 2 and B row 2
    # C[i,k] = A[i,2] * B[2,k]
    # C-space: 14, A-space: 6, B-space: 21
    # T[c, a, b] = 1 if c=i*7+k, a=i*3+2, b=2*7+k
    # Terms: for each (i,k): U=e_{i*7+k}, V=e_{i*3+2}, W=e_{2*7+k}
    for i in range(2):
        for k in range(7):
            u = [0] * 14
            u[i * 7 + k] = 1
            v = [0] * 6
            v[i * 3 + 2] = 1
            w = [0] * 21
            w[2 * 7 + k] = 1
            U_list.append(u)
            V_list.append(v)
            W_list.append(w)
    
    rank = len(U_list)
    # Convert to matrices (rows = indices, cols = rank)
    U = [[0] * rank for _ in range(14)]
    V = [[0] * rank for _ in range(6)]
    W = [[0] * rank for _ in range(21)]
    
    for r in range(rank):
        for i in range(14):
            U[i][r] = U_list[r][i]
        for i in range(6):
            V[i][r] = V_list[r][i]
        for i in range(21):
            W[i][r] = W_list[r][i]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
