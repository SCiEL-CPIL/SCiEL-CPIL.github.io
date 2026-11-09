"""Matrix multiplication 6x6x6 rank minimization via Strassen + Laderman 3x3 block decomposition."""
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Partition 6x6 matrices into 2x2 blocks of 3x3.
    Apply Strassen's algorithm (7 block products) at the block level.
    Apply Laderman's 23-term decomposition to each 3x3 block product.
    Total rank = 7 * 23 = 161.
    """
    n, m, p = DIMENSIONS  # 6, 6, 6
    
    # Laderman 23-term decomposition for 3x3 matrix multiplication
    # A flattened: a[0..8] = A[0][0], A[0][1], A[0][2], A[1][0], ..., A[2][2]
    # B flattened: b[0..8] = B[0][0], B[0][1], B[0][2], B[1][0], ..., B[2][2]
    # C flattened: c[0..8] = C[0][0], C[0][1], C[0][2], C[1][0], ..., C[2][2]
    
    # 23 terms: each is (A_coeffs[9], B_coeffs[9])
    laderman_terms = [
        ([1,0,0, 0,0,0, 0,0,0], [0,1,0, 0,-1,0, 0,0,0]),   # m1 = A00*(B01-B11)
        ([1,1,0, 0,0,0, 0,0,0], [0,0,0, 0,1,0, 0,0,0]),     # m2 = (A00+A01)*B11
        ([0,0,0, 1,1,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0]),     # m3 = (A10+A11)*B00
        ([0,0,0, 0,1,0, 0,0,0], [1,0,0, -1,0,0, 0,0,0]),    # m4 = A11*(B10-B00)
        ([1,0,0, 0,0,0, 0,0,1], [1,0,0, 0,0,0, 0,0,0]),     # m5 = (A00+A22)*B00
        ([1,0,0, 0,1,0, -1,0,1], [1,0,0, 0,1,0, 0,0,0]),    # m6 = (A20-A00+A11+A22)*(B00+B11)
        ([0,1,0, 0,-1,0, 0,0,1], [0,0,0, 0,1,0, 0,0,1]),    # m7 = (A01-A11+A22)*(B11+B22)
        ([1,1,0, -1,-1,0, 0,0,0], [0,0,0, 1,1,0, 0,0,0]),   # m8 = (A00+A01-A10-A11)*(B10+B11)
        ([0,0,0, 1,1,0, -1,-1,0], [1,1,0, 0,0,0, 0,0,0]),   # m9 = (A10+A11-A00-A01)*(B00+B01)
        ([0,0,0, 0,0,0, 1,0,1], [1,-1,0, 1,0,0, 0,0,0]),    # m10 = (A20+A22)*(B01-B00+B10)
        ([0,0,0, 0,0,0, 1,0,1], [1,0,-1, 0,0,1, 0,0,1]),    # m11 = (A20+A22)*(B02-B12+B22)
        ([0,1,0, 0,1,0, 0,0,0], [1,0,0, 0,0,0, -1,0,1]),    # m12 = (A01+A11)*(B20-B00+B22)
        ([1,0,0, 0,0,0, 0,0,1], [0,1,-1, 0,0,0, 0,0,1]),    # m13 = (A00+A22)*(B21-B01+B22)
        ([0,0,0, 1,1,0, 0,0,0], [1,0,0, -1,0,0, 0,0,1]),    # m14 = (A10+A11)*(B20-B10+B22)
        ([1,1,0, 0,0,0, 0,0,0], [0,1,-1, 0,0,0, 0,0,1]),    # m15 = (A00+A01)*(B21-B01+B22)
        ([1,0,0, 0,0,0, 0,0,1], [0,0,1, -1,0,0, 0,0,1]),    # m16 = (A00+A22)*(B12-B02+B22)
        ([1,0,0, 0,1,0, -1,0,1], [0,0,1, 0,0,1, 0,0,0]),    # m17 = (A20-A00+A11+A22)*(B02+B12)
        ([0,1,0, 0,-1,0, 0,0,1], [0,1,0, 0,0,0, 1,0,0]),    # m18 = (A01-A11+A22)*(B10+B20)
        ([1,1,0, -1,-1,0, 0,0,0], [0,0,0, 0,0,0, 1,1,0]),   # m19 = (A00+A01-A10-A11)*(B20+B21)
        ([0,0,0, 1,1,0, -1,-1,0], [0,0,1, 1,0,0, 0,0,0]),   # m20 = (A10+A11-A00-A01)*(B02+B01)
        ([0,0,0, 0,0,0, 1,0,1], [0,1,-1, 0,0,0, 0,1,0]),    # m21 = (A20+A22)*(B11-B01+B21)
        ([1,0,0, 0,0,0, 0,0,1], [1,0,0, -1,0,0, 0,1,0]),    # m22 = (A00+A22)*(B10-B00+B20)
        ([0,0,0, 1,1,0, 0,0,0], [0,1,0, -1,0,0, 0,0,1]),    # m23 = (A10+A11)*(B01-B00+B21)
    ]
    
    # Output: C[c_idx] = sum_l laderman_output[c_idx][l] * m_l
    laderman_output = [
        [1,1,1,0,1,-1,0,1,-1,0,0,1,-1,1,0,0,0,-1,1,-1,0,1,0],   # C00
        [0,1,0,1,1,1,-1,0,1,-1,0,0,1,-1,1,1,0,0,-1,1,-1,0,1],   # C01
        [1,0,-1,0,1,0,1,-1,0,0,1,-1,1,0,-1,1,1,0,0,1,-1,0,1],   # C02
        [1,1,1,0,0,1,-1,1,-1,1,0,-1,0,1,-1,0,0,1,-1,0,1,0,0],   # C10
        [0,1,0,1,1,-1,1,-1,1,-1,0,1,-1,0,1,-1,0,1,-1,0,0,0,0],  # C11
        [1,0,-1,0,-1,1,-1,1,-1,1,-1,0,1,0,-1,1,-1,1,0,0,0,0,0], # C12
        [1,1,-1,0,-1,1,1,-1,1,-1,0,-1,0,1,-1,0,0,0,-1,0,0,0,0], # C20
        [0,1,0,-1,-1,-1,-1,1,1,-1,0,0,-1,0,1,1,0,0,1,1,-1,0,0], # C21
        [1,0,-1,0,-1,0,-1,1,1,-1,0,-1,0,-1,0,1,1,1,0,1,0,0,0],  # C22
    ]
    
    # Strassen's algorithm for 2x2 block matrix multiplication
    # 7 terms, each is a product of two linear combinations of 3x3 blocks
    
    # Block layout: A = [[A00, A01], [A10, A11]] where each is 3x3
    # B = [[B00, B01], [B10, B11]]
    # C = A*B = [[C00, C01], [C10, C11]]
    
    # Strassen terms:
    # M0 = (A00 + A11) * (B00 + B11)
    # M1 = (A10 + A11) * B00
    # M2 = A00 * (B01 - B11)
    # M3 = A11 * (B10 - B00)
    # M4 = (A00 + A01) * B11
    # M5 = (A10 - A00) * (B00 + B01)
    # M6 = (A01 - A11) * (B10 + B11)
    
    # Output:
    # C00 = M0 + M3 - M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5
    
    # Each Strassen term is defined as (A_blocks, B_blocks)
    # A_blocks: list of (block_row, block_col, coeff)
    # B_blocks: list of (block_row, block_col, coeff)
    
    strassen_terms = [
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),  # M0
        ([(1,0,1), (1,1,1)], [(0,0,1)]),             # M1
        ([(0,0,1)], [(0,1,1), (1,1,-1)]),            # M2
        ([(1,1,1)], [(1,0,1), (0,0,-1)]),            # M3
        ([(0,0,1), (0,1,1)], [(1,1,1)]),             # M4
        ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)]),   # M5
        ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)]),   # M6
    ]
    
    # Strassen output: C_block = sum over strassen terms
    # (out_block_row, out_block_col): {strassen_term_idx: coeff}
    strassen_output = {
        (0,0): {0: 1, 3: 1, 4: -1, 6: 1},   # C00 = M0 + M3 - M4 + M6
        (0,1): {2: 1, 4: 1},                  # C01 = M2 + M4
        (1,0): {1: 1, 3: 1},                  # C10 = M1 + M3
        (1,1): {0: 1, 1: -1, 2: 1, 5: 1},    # C11 = M0 - M1 + M2 + M5
    }
    
    rank = 7 * 23
    U = [[0] * rank for _ in range(n * m)]  # 36 x 161
    V = [[0] * rank for _ in range(m * p)]  # 36 x 161
    W = [[0] * rank for _ in range(p * n)]  # 36 x 161
    
    # For each Strassen term and Laderman term, fill in U, V, W
    for s_idx, (a_blocks, b_blocks) in enumerate(strassen_terms):
        for l_idx in range(23):
            rank_idx = s_idx * 23 + l_idx
            
            a_coeffs = laderman_terms[l_idx][0]
            b_coeffs = laderman_terms[l_idx][1]
            
            # Fill U
            for br, bc, coeff in a_blocks:
                r0 = br * 3
                c0 = bc * 3
                for i in range(3):
                    for j in range(3):
                        a_flat = i * 3 + j
                        u_row = (r0 + i) * m + (c0 + j)
                        U[u_row][rank_idx] += coeff * a_coeffs[a_flat]
            
            # Fill V
            for br, bc, coeff in b_blocks:
                r0 = br * 3
                c0 = bc * 3
                for i in range(3):
                    for j in range(3):
                        b_flat = i * 3 + j
                        v_row = (r0 + i) * p + (c0 + j)
                        V[v_row][rank_idx] += coeff * b_coeffs[b_flat]
            
            # Fill W
            for (out_br, out_bc), term_coeffs in strassen_output.items():
                if s_idx in term_coeffs:
                    s_coeff = term_coeffs[s_idx]
                    for c_idx in range(9):
                        w_l_coeff = laderman_output[c_idx][l_idx]
                        if w_l_coeff != 0:
                            ci = c_idx // 3
                            cj = c_idx % 3
                            r0 = out_br * 3
                            c0 = out_bc * 3
                            w_row = (r0 + ci) * n + (c0 + cj)
                            W[w_row][rank_idx] += s_coeff * w_l_coeff
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
