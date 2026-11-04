"""Program for rank-minimal exact tensor decomposition of 6x6 matrix multiplication."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 6, 6, 6
    
    # Block size is 3x3.
    # 6x6 matrix is divided into 4 blocks of size 3x3.
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    
    # Rank calculation:
    # Strassen requires 7 products of 3x3 blocks.
    # Each 3x3 product is decomposed using schoolbook (rank 9).
    # Expansion of Strassen products:
    # M1: 4 block products
    # M2: 2 block products
    # M3: 2 block products
    # M4: 2 block products
    # M5: 2 block products
    # M6: 4 block products
    # M7: 4 block products
    # Total 20 block products.
    # Total rank = 20 * 9 = 180.
    
    rank = 180
    U = [[0] * rank for _ in range(n * m)]  # 36 x 180
    V = [[0] * rank for _ in range(m * p)]  # 36 x 180
    W = [[0] * rank for _ in range(p * n)]  # 36 x 180
    
    col = 0
    
    # Helper to get U row index for A[bi][bj][i][j]
    # bi, bj are block indices (0 or 1)
    # i, j are indices within the 3x3 block (0, 1, 2)
    # Global row in A: (bi*3 + i) * 6 + (bj*3 + j)
    def u_row(bi, bj, i, j):
        return (bi * 3 + i) * m + (bj * 3 + j)
        
    # Helper to get V row index for B[bi][bj][i][j]
    # Global row in B: (bi*3 + i) * 6 + (bj*3 + j)
    def v_row(bi, bj, i, j):
        return (bi * 3 + i) * p + (bj * 3 + j)
        
    # Helper to get W row index for C[bi][bj][i][j]
    # W[k*n+i, r] corresponds to C[i, k]
    # C[bi][bj] is at rows 3*bi..3*bi+2, cols 3*bj..3*bj+2
    # C[i, k] where i = 3*bi + i_local, k = 3*bj + k_local
    # W row index = k * n + i = (3*bj + k_local) * 6 + (3*bi + i_local)
    def w_row(bi, bj, i, j):
        # bi, bj are block indices of C
        # i, j are local indices in C block
        # C row index = 3*bi + i
        # C col index = 3*bj + j
        # W row = col * n + row
        return (3 * bj + j) * n + (3 * bi + i)

    # Strassen products definitions
    # Each product is a list of (A_block_idx, B_block_idx, sign)
    # A_block_idx: (bi, bj)
    # B_block_idx: (bi, bj)
    
    # M1 = (A00 + A11)(B00 + B11)
    # Expansion: A00B00 + A00B11 + A11B00 + A11B11
    m1_terms = [((0,0), (0,0), 1), ((0,0), (0,1), 1), ((1,1), (0,0), 1), ((1,1), (0,1), 1)]
    # Note: B indices in M1 are B00 and B11.
    # B00 is (0,0), B11 is (1,1).
    # Wait, B11 is (1,1) in my notation?
    # B = [[B00, B01], [B10, B11]]
    # So B11 is block (1,1).
    # My tuple (bi, bj) for B:
    # B00: (0,0)
    # B01: (0,1)
    # B10: (1,0)
    # B11: (1,1)
    
    # Let's re-verify M1 terms:
    # (A00 + A11)(B00 + B11) = A00B00 + A00B11 + A11B00 + A11B11
    # A00: (0,0), A11: (1,1)
    # B00: (0,0), B11: (1,1)
    m1_terms = [
        ((0,0), (0,0), 1), # A00 B00
        ((0,0), (1,1), 1), # A00 B11
        ((1,1), (0,0), 1), # A11 B00
        ((1,1), (1,1), 1)  # A11 B11
    ]
    
    # M2 = (A10 + A11)B00
    # A10: (1,0), A11: (1,1)
    # B00: (0,0)
    m2_terms = [
        ((1,0), (0,0), 1), # A10 B00
        ((1,1), (0,0), 1)  # A11 B00
    ]
    
    # M3 = A00(B01 - B11)
    # A00: (0,0)
    # B01: (0,1), B11: (1,1)
    m3_terms = [
        ((0,0), (0,1), 1),  # A00 B01
        ((0,0), (1,1), -1)  # A00 (-B11)
    ]
    
    # M4 = A11(B10 - B00)
    # A11: (1,1)
    # B10: (1,0), B00: (0,0)
    m4_terms = [
        ((1,1), (1,0), 1),  # A11 B10
        ((1,1), (0,0), -1)  # A11 (-B00)
    ]
    
    # M5 = (A00 + A01)B11
    # A00: (0,0), A01: (0,1)
    # B11: (1,1)
    m5_terms = [
        ((0,0), (1,1), 1), # A00 B11
        ((0,1), (1,1), 1)  # A01 B11
    ]
    
    # M6 = (A10 - A00)(B00 + B01)
    # A10: (1,0), A00: (0,0)
    # B00: (0,0), B01: (0,1)
    m6_terms = [
        ((1,0), (0,0), 1),  # A10 B00
        ((1,0), (0,1), 1),  # A10 B01
        ((0,0), (0,0), -1), # (-A00) B00
        ((0,0), (0,1), -1)  # (-A00) B01
    ]
    
    # M7 = (A01 - A11)(B10 + B11)
    # A01: (0,1), A11: (1,1)
    # B10: (1,0), B11: (1,1)
    m7_terms = [
        ((0,1), (1,0), 1),  # A01 B10
        ((0,1), (1,1), 1),  # A01 B11
        ((1,1), (1,0), -1), # (-A11) B10
        ((1,1), (1,1), -1)  # (-A11) B11
    ]
    
    # Output block contributions
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We will iterate through each Strassen product, then each term, then the 9 schoolbook terms.
    
    # M1
    for term in m1_terms:
        a_idx, b_idx, sign = term
        for k in range(3):
            for i in range(3):
                for j in range(3):
                    # Term: sign * A[a_idx][i,k] * B[b_idx][k,j]
                    # Contributes to C00 and C11
                    # C00: +1
                    # C11: +1
                    
                    # U
                    U[u_row(a_idx[0], a_idx[1], i, k)][col] = sign
                    # V
                    V[v_row(b_idx[0], b_idx[1], k, j)][col] = 1
                    # W for C00 (block 0,0)
                    W[w_row(0, 0, i, j)][col] = 1
                    # W for C11 (block 1,1)
                    W[w_row(1, 1, i, j)][col] = 1
                    col += 1
                    
    # M2
    for term in m2_terms:
        a_idx, b_idx, sign = term
        for k in range(3):
            for i in range(3):
                for j in range(3):
                    # Contributes to C10 (+1) and C11 (-1)
                    U[u_row(a_idx[0], a_idx[1], i, k)][col] = sign
                    V[v_row(b_idx[0], b_idx[1], k, j)][col] = 1
                    W[w_row(1, 0, i, j)][col] = 1
                    W[w_row(1, 1, i, j)][col] = -1
                    col += 1
                    
    # M3
    for term in m3_terms:
        a_idx, b_idx, sign = term
        for k in range(3):
            for i in range(3):
                for j in range(3):
                    # Contributes to C01 (+1) and C11 (+1)
                    U[u_row(a_idx[0], a_idx[1], i, k)][col] = sign
                    V[v_row(b_idx[0], b_idx[1], k, j)][col] = 1
                    W[w_row(0, 1, i, j)][col] = 1
                    W[w_row(1, 1, i, j)][col] = 1
                    col += 1
                    
    # M4
    for term in m4_terms:
        a_idx, b_idx, sign = term
        for k in range(3):
            for i in range(3):
                for j in range(3):
                    # Contributes to C00 (+1) and C10 (+1)
                    U[u_row(a_idx[0], a_idx[1], i, k)][col] = sign
                    V[v_row(b_idx[0], b_idx[1], k, j)][col] = 1
                    W[w_row(0, 0, i, j)][col] = 1
                    W[w_row(1, 0, i, j)][col] = 1
                    col += 1
                    
    # M5
    for term in m5_terms:
        a_idx, b_idx, sign = term
        for k in range(3):
            for i in range(3):
                for j in range(3):
                    # Contributes to C00 (-1) and C01 (+1)
                    U[u_row(a_idx[0], a_idx[1], i, k)][col] = sign
                    V[v_row(b_idx[0], b_idx[1], k, j)][col] = 1
                    W[w_row(0, 0, i, j)][col] = -1
                    W[w_row(0, 1, i, j)][col] = 1
                    col += 1
                    
    # M6
    for term in m6_terms:
        a_idx, b_idx, sign = term
        for k in range(3):
            for i in range(3):
                for j in range(3):
                    # Contributes to C11 (+1)
                    U[u_row(a_idx[0], a_idx[1], i, k)][col] = sign
                    V[v_row(b_idx[0], b_idx[1], k, j)][col] = 1
                    W[w_row(1, 1, i, j)][col] = 1
                    col += 1
                    
    # M7
    for term in m7_terms:
        a_idx, b_idx, sign = term
        for k in range(3):
            for i in range(3):
                for j in range(3):
                    # Contributes to C00 (+1)
                    U[u_row(a_idx[0], a_idx[1], i, k)][col] = sign
                    V[v_row(b_idx[0], b_idx[1], k, j)][col] = 1
                    W[w_row(0, 0, i, j)][col] = 1
                    col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
