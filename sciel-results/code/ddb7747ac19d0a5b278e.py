import numpy as np

# We implement Strassen-based bilinear decomposition for 4x4 * 4x9.
# Strategy: treat 4x9 as two 4x4 blocks + one 4x1 column.
# - For each 4x4 block: use rank-49 Strassen decomposition (7 multiplications of 2x2 blocks, each 2x2 rank 7)
# - For the 4x1 column: schoolbook rank 16
# Total rank = 49 + 49 + 16 = 114

def _strassen_2x2_terms():
    """Return the 7 bilinear terms for 2x2 matrix multiplication (rank 7).
    
    Returns a list of 7 tuples, each (U_coeffs, V_coeffs, W_coeffs)
    where each is a dict from (i,j) to coefficient.
    For 2x2: indices (i,j) with i in {0,1}, j in {0,1}
    U: (n*m) = 4 entries, indexed by (i, j) where i is row of A (0,1), j is col of A (0,1)
    V: (m*p) = 4 entries, indexed by (j, k) where j is row of B (0,1), k is col of B (0,1)
    W: (p*n) = 4 entries, indexed by (k, i) where k is row of C (0,1), i is col of C (0,1)
    """
    # Strassen for 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    terms = []
    
    # M1 = (A00+A11)(B00+B11)
    U1 = {(0,0): 1, (1,1): 1}
    V1 = {(0,0): 1, (1,1): 1}
    W1 = {(0,0): 1, (1,1): 1}  # contributes to C00 and C11
    # Actually W should map the scalar product to C entries
    # M1 contributes +1 to C00 and +1 to C11
    terms.append((U1, V1, {(0,0): 1, (1,1): 1}))
    
    # M2 = (A10+A11)B00
    U2 = {(1,0): 1, (1,1): 1}
    V2 = {(0,0): 1}
    W2 = {(1,0): 1}  # contributes to C10
    terms.append((U2, V2, W2))
    
    # M3 = A00(B01-B11)
    U3 = {(0,0): 1}
    V3 = {(0,1): 1, (1,1): -1}
    W3 = {(0,1): 1, (1,1): 1}  # contributes to C01 and C11
    terms.append((U3, V3, W3))
    
    # M4 = A11(B10-B00)
    U4 = {(1,1): 1}
    V4 = {(1,0): 1, (0,0): -1}
    W4 = {(0,0): 1, (1,0): 1}  # contributes to C00 and C10
    terms.append((U4, V4, W4))
    
    # M5 = (A00+A01)B11
    U5 = {(0,0): 1, (0,1): 1}
    V5 = {(1,1): 1}
    W5 = {(0,0): -1, (0,1): 1}  # contributes -1 to C00, +1 to C01
    terms.append((U5, V5, W5))
    
    # M6 = (A10-A00)(B00+B01)
    U6 = {(1,0): 1, (0,0): -1}
    V6 = {(0,0): 1, (0,1): 1}
    W6 = {(1,1): 1}  # contributes to C11
    terms.append((U6, V6, W6))
    
    # M7 = (A01-A11)(B10+B11)
    U7 = {(0,1): 1, (1,1): -1}
    V7 = {(1,0): 1, (1,1): 1}
    W7 = {(0,0): 1}  # contributes to C00
    terms.append((U7, V7, W7))
    
    return terms


def _schoolbook_4x1_terms():
    """Return 16 bilinear terms for 4x1 matrix-vector multiplication.
    
    U: 4x4 entries, V: 4x1 entries, W: 1x4 entries
    For each (i,j) pair: U[i,j]=1, V[j,0]=1, W[0,i]=1
    """
    terms = []
    for i in range(4):
        for j in range(4):
            U = {(i, j): 1}
            V = {(j, 0): 1}
            W = {(0, i): 1}
            terms.append((U, V, W))
    return terms


def _strassen_4x4_terms():
    """Return 49 bilinear terms for 4x4 matrix multiplication using Strassen.
    
    Split 4x4 into four 2x2 blocks:
    A = [[A00, A01], [A10, A11]] where each Aij is 2x2
    B = [[B00, B01], [B10, B11]] where each Bij is 2x2
    
    Use Strassen at the block level (7 block multiplications), each block 
    multiplication uses rank-7 Strassen for 2x2.
    
    Returns list of 49 tuples (U, V, W) where U, V, W are dicts with keys
    being flat indices.
    """
    # Block-level Strassen terms (7 of them)
    # Each maps to (U_block, V_block, W_block) where each block is a 2x2 matrix
    # of coefficients
    
    # Block indices: (bi, bj) where bi in {0,1} is block row, bj in {0,1} is block col
    # For A: block (bi, bj) has rows 2*bi..2*bi+1, cols 2*bj..2*bj+1
    # For B: block (bi, bj) has rows 2*bi..2*bi+1, cols 2*bj..2*bj+1
    # For C: block (bi, bj) has rows 2*bi..2*bi+1, cols 2*bj..2*bj+1
    
    # Strassen block combinations:
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
    
    block_strassen = [
        # (U_blocks, V_blocks, W_blocks)
        # U_blocks: dict from (block_row, block_col) to coeff
        # V_blocks: dict from (block_row, block_col) to coeff
        # W_blocks: dict from (block_row, block_col) to coeff (output block positions)
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),  # M1
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1}),  # M2
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),  # M3
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),  # M4
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),  # M5
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),  # M6
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),  # M7
    ]
    
    # Get the 7 base 2x2 Strassen terms
    base_terms = _strassen_2x2_terms()
    
    all_terms = []
    for block_U, block_V, block_W in block_strassen:
        for base_U, base_V, base_W in base_terms:
            # Expand each block term into individual 4x4 indices
            U_full = {}
            V_full = {}
            W_full = {}
            
            # Expand U
            for (bi, bj), coeff_u_block in block_U.items():
                for (li, lj), coeff_u_base in base_U.items():
                    # Global row: 2*bi + li, global col: 2*bj + lj
                    row = 2 * bi + li
                    col = 2 * bj + lj
                    key = (row, col)
                    U_full[key] = U_full.get(key, 0) + coeff_u_block * coeff_u_base
            
            # Expand V
            for (bi, bj), coeff_v_block in block_V.items():
                for (li, lj), coeff_v_base in base_V.items():
                    # B block (bi,bj): rows 2*bi+li, cols 2*bj+lj
                    row = 2 * bi + li
                    col = 2 * bj + lj
                    key = (row, col)
                    V_full[key] = V_full.get(key, 0) + coeff_v_block * coeff_v_base
            
            # Expand W
            for (bi, bj), coeff_w_block in block_W.items():
                for (li, lj), coeff_w_base in base_W.items():
                    # C block (bi,bj): rows 2*bi+li, cols 2*bj+lj
                    row = 2 * bi + li
                    col = 2 * bj + lj
                    key = (row, col)
                    W_full[key] = W_full.get(key, 0) + coeff_w_block * coeff_w_base
            
            all_terms.append((U_full, V_full, W_full))
    
    return all_terms


def construct():
    n, m, p = 4, 4, 9
    
    # Collect all bilinear terms
    all_terms = []
    
    # Two 4x4 blocks for columns 0-3 and 4-7 of B
    strassen_4x4 = _strassen_4x4_terms()
    
    # First 4x4 block: columns 0-3
    for U, V, W in strassen_4x4:
        # U is for A (4x4), stays the same
        # V is for B block 1 (4x4), columns 0-3
        V_offset = {}
        for (row, col), coeff in V.items():
            V_offset[(row, col)] = coeff  # col 0-3
        
        # W is for C block 1 (4x4), columns 0-3
        W_offset = {}
        for (row, col), coeff in W.items():
            W_offset[(row, col)] = coeff  # col 0-3
        
        all_terms.append((U, V_offset, W_offset))
    
    # Second 4x4 block: columns 4-7
    for U, V, W in strassen_4x4:
        # U is for A (4x4), stays the same
        # V is for B block 2 (4x4), columns 4-7
        V_offset = {}
        for (row, col), coeff in V.items():
            V_offset[(row, col + 4)] = coeff
        
        # W is for C block 2 (4x4), columns 4-7
        W_offset = {}
        for (row, col), coeff in W.items():
            W_offset[(row, col + 4)] = coeff
        
        all_terms.append((U, V_offset, W_offset))
    
    # 4x1 block: column 8
    schoolbook_4x1 = _schoolbook_4x1_terms()
    for U, V, W in schoolbook_4x1:
        # U is for A (4x4), stays the same
        # V is for B column 8 (4x1)
        V_offset = {}
        for (row, col), coeff in V.items():
            V_offset[(row, 8)] = coeff
        
        # W is for C column 8 (4x1)
        W_offset = {}
        for (row, col), coeff in W.items():
            W_offset[(row, 8)] = coeff
        
        all_terms.append((U, V_offset, W_offset))
    
    rank = len(all_terms)
    
    # Build U, V, W matrices
    U_mat = np.zeros((n * m, rank))
    V_mat = np.zeros((m * p, rank))
    W_mat = np.zeros((p * n, rank))
    
    for c, (U_dict, V_dict, W_dict) in enumerate(all_terms):
        for (i, j), coeff in U_dict.items():
            if coeff != 0:
                U_mat[i * m + j, c] = coeff
        for (j, k), coeff in V_dict.items():
            if coeff != 0:
                V_mat[j * p + k, c] = coeff
        for (k, i), coeff in W_dict.items():
            if coeff != 0:
                W_mat[k * n + i, c] = coeff
    
    # Convert to lists
    U_list = U_mat.tolist()
    V_list = V_mat.tolist()
    W_list = W_mat.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
