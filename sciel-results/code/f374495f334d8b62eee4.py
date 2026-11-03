"""Rewritten program for matrix multiplication 4x4 x 4x8 with rank-98 decomposition."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-98 exact tensor decomposition for 4x4 * 4x8 matrix multiplication.
    
    The 4x4x8 multiplication tensor decomposes as a direct sum of two 4x4x4 tensors:
    - Output columns 0-3: C[:, 0:4] = A * B[:, 0:4]
    - Output columns 4-7: C[:, 4:8] = A * B[:, 4:8]
    
    Each 4x4x4 sub-tensor is decomposed using the rank-49 Strassen-based construction:
    View 4x4 matrices as 2x2 blocks of 2x2 matrices. Apply Strassen (rank 7) at the
    block level to get 7 block-products, each of which is a 2x2x2 matrix multiplication
    requiring 7 Strassen scalar multiplications. Total: 7 * 7 = 49 terms per sub-tensor.
    
    Total rank: 49 + 49 = 98.
    
    Output format: W[k*n + i, r] where n=4, k in 0..7, i in 0..3.
    """
    n, m, p = DIMENSIONS  # n=4, m=4, p=8
    
    # Strassen 2x2x2 decomposition: 7 rank-1 terms
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # Flattened row-major: index 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # M1 = (a+d)(e+h), M2 = (c+d)e, M3 = a(f-h), M4 = d(g-e)
    # M5 = (a+b)h, M6 = (c-a)(e+f), M7 = (b-d)(g+h)
    # C11 = M1+M4-M5+M7, C12 = M3+M5, C21 = M2+M4, C22 = M1-M2+M3+M6
    strassen_U = [
        [1, 0, 0, 1],   # M1: a+d
        [0, 0, 1, 1],   # M2: c+d
        [1, 0, 0, 0],   # M3: a
        [0, 0, 0, 1],   # M4: d
        [1, 1, 0, 0],   # M5: a+b
        [-1, 0, 1, 0],  # M6: c-a
        [0, 1, 0, -1],  # M7: b-d
    ]
    strassen_V = [
        [1, 0, 0, 1],   # M1: e+h
        [1, 0, 0, 0],   # M2: e
        [0, 1, 0, -1],  # M3: f-h
        [-1, 0, 1, 0],  # M4: g-e
        [0, 0, 0, 1],   # M5: h
        [1, 1, 0, 0],   # M6: e+f
        [0, 0, 1, 1],   # M7: g+h
    ]
    strassen_W = [
        [1, 0, 0, 1],   # M1: C11+1, C22+1
        [0, 0, 1, -1],  # M2: C21+1, C22-1
        [0, 1, 0, 1],   # M3: C12+1, C22+1
        [1, 0, 1, 0],   # M4: C11+1, C21+1
        [-1, 1, 0, 0],  # M5: C11-1, C12+1
        [0, 0, 0, 1],   # M6: C22+1
        [1, 0, 0, 0],   # M7: C11+1
    ]
    
    # Block-level Strassen for 4x4x4 (2x2 blocks of 2x2 matrices)
    # Block indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # S1 = (A00+A11)(B00+B11), S2 = (A10+A11)B00, S3 = A00(B01-B11)
    # S4 = A11(B10-B00), S5 = (A00+A01)B11, S6 = (A10-A00)(B00+B01)
    # S7 = (A01-A11)(B10+B11)
    # C00 = S1+S4-S5+S7, C01 = S3+S5, C10 = S2+S4, C11 = S1-S2+S3+S6
    
    block_A = [
        {0: 1, 3: 1},       # S1
        {2: 1, 3: 1},       # S2
        {0: 1},             # S3
        {3: 1},             # S4
        {0: 1, 1: 1},       # S5
        {2: 1, 0: -1},      # S6
        {1: 1, 3: -1},      # S7
    ]
    block_B = [
        {0: 1, 3: 1},       # S1
        {0: 1},             # S2
        {1: 1, 3: -1},      # S3
        {2: 1, 0: -1},      # S4
        {3: 1},             # S5
        {0: 1, 1: 1},       # S6
        {2: 1, 3: 1},       # S7
    ]
    # For each C-block c_idx, which S-terms contribute and with what coefficient
    c_block_contrib = [
        [(0, 1), (3, 1), (4, -1), (6, 1)],  # C00
        [(2, 1), (4, 1)],                     # C01
        [(1, 1), (3, 1)],                     # C10
        [(0, 1), (1, -1), (2, 1), (5, 1)],   # C11
    ]
    
    def build_4x4_decomp():
        """Build the 49-term rank-1 decomposition for 4x4x4 matrix multiplication.
        Returns U, V, W each of shape (16, 49)."""
        U = [[0] * 49 for _ in range(16)]
        V = [[0] * 49 for _ in range(16)]
        W = [[0] * 49 for _ in range(16)]
        
        for s_idx in range(7):
            for str_idx in range(7):
                term_idx = s_idx * 7 + str_idx
                su = strassen_U[str_idx]
                sv = strassen_V[str_idx]
                sw = strassen_W[str_idx]
                
                # U and V: for each block that contributes to this S-term
                for bi in range(2):
                    for bj in range(2):
                        bidx = bi * 2 + bj
                        a_coeff = block_A[s_idx].get(bidx, 0)
                        b_coeff = block_B[s_idx].get(bidx, 0)
                        if a_coeff == 0 and b_coeff == 0:
                            continue
                        for li in range(2):
                            for lj in range(2):
                                flat_idx = (2 * bi + li) * 4 + (2 * bj + lj)
                                U[flat_idx][term_idx] = a_coeff * su[li * 2 + lj]
                                V[flat_idx][term_idx] = b_coeff * sv[li * 2 + lj]
                
                # W: for each C-block that this S-term contributes to
                for c_idx in range(4):
                    c_coeff = 0
                    for (s2, coeff) in c_block_contrib[c_idx]:
                        if s2 == s_idx:
                            c_coeff = coeff
                            break
                    if c_coeff == 0:
                        continue
                    ci = c_idx // 2
                    cj = c_idx % 2
                    for li in range(2):
                        for lk in range(2):
                            flat_idx = (2 * ci + li) * 4 + (2 * cj + lk)
                            W[flat_idx][term_idx] = c_coeff * sw[li * 2 + lk]
        
        return U, V, W
    
    U4, V4, W4 = build_4x4_decomp()
    
    # Build the full 4x4x8 decomposition (rank 98)
    rank = 98
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(32)]
    W = [[0] * rank for _ in range(32)]
    
    # U: same 49-term pattern repeated for both halves
    for i in range(16):
        for t in range(49):
            U[i][t] = U4[i][t]
            U[i][49 + t] = U4[i][t]
    
    # V: B is 4x8, flattened row-major: B[i,j] -> index i*8+j
    # For j in 0..3 (first half): V[i*8+j, t] = V4[i*4+j, t]
    # For j in 4..7 (second half): V[i*8+j, 49+t] = V4[i*4+(j-4), 49+t]
    for i in range(4):
        for j in range(4):
            for t in range(49):
                V[i * 8 + j][t] = V4[i * 4 + j][t]
        for j in range(4):
            for t in range(49):
                V[i * 8 + (j + 4)][49 + t] = V4[i * 4 + j][49 + t]
    
    # W: output C is 4x8, C[i,k] for i in 0..3, k in 0..7
    # W[k*n+i, r] = W[k*4+i, r]
    # For k in 0..3: W[k*4+i, t] = W4[i*4+k, t]
    # For k in 4..7: W[k*4+i, 49+t] = W4[i*4+(k-4), 49+t]
    for i in range(4):
        for k in range(4):
            for t in range(49):
                W[k * 4 + i][t] = W4[i * 4 + k][t]
        for k in range(4):
            for t in range(49):
                W[(k + 4) * 4 + i][49 + t] = W4[i * 4 + k][49 + t]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
