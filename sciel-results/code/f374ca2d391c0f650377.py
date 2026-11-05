"""
Matrix multiplication 5x5x5 tensor decomposition using Strassen's algorithm
for the 4x4 core block and naive decomposition for the remaining parts.

The 5x5 matrix multiplication can be decomposed as:
- A 4x4 core (using Strassen, rank 49)
- Remaining parts involving the 5th row/column (naive, rank 61)
Total rank: 110
"""

import numpy as np

DIMENSIONS = [5, 5, 5]


def _strassen_2x2():
    """
    Return U, V, W matrices for 2x2x2 matrix multiplication using Strassen's algorithm.
    U: shape (4, 7), V: shape (4, 7), W: shape (4, 7)
    Indexing: row = i*p + j for U (i row of A, j col of A)
              row = j*p + k for V (j row of B, k col of B)  
              row = k*n + i for W (k row of C, i col of C) -> but W[k*n+i, r]
    For 2x2: n=m=p=2
    U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    n = m = p = 2
    rank = 7
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen's 7 products for 2x2:
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # C = [[a*e - c*f, a*f + b*h], [c*e + d*g, b*g - a*f + d*h + c*e]]
    
    # M1 = (a+d)(e+h) = ae+ah+de+dh
    # M2 = (c+d)e = ce+de
    # M3 = a(f-h) = af-ah
    # M4 = d(g-e) = dg-de
    # M5 = (a+b)h = ah+bh
    # M6 = (c-a)(e+f) = ce+cf-ae-af
    # M7 = (b-d)(g+h) = bg+bh-dg-dh
    
    # C11 = M1 - M4 + M6 - M7 = ae+ah+de+dh - dg+de + ce+cf-ae-af - bg-bh+dg+dh
    #     = ae+ah+2de+dh-dg+ce+cf-ae-af-bg-bh+dg+dh
    #     = ah+2de+2dh+ce+cf-af-bg-bh
    # Hmm, let me use the standard formulation more carefully.
    
    # Standard Strassen:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 - M4 + M6 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M3 + M5 - M6
    
    # For each Strassen product, we need to express it as a rank-1 tensor term.
    # Each Mi is a product of two linear combinations, so it's a sum of rank-1 terms.
    
    # Let's define the 7 Strassen terms and their contributions to C.
    
    # Term 1: M1 = (a+d)(e+h)
    # a=A[0,0]=U[0], d=A[1,1]=U[3], e=B[0,0]=V[0], h=B[1,1]=V[3]
    # M1 = ae + ah + de + dh
    # This contributes to C11 (+1), C22 (+1)
    
    # Actually, let me just build this directly by enumerating each scalar product.
    
    # For Strassen 2x2, we have 7 "multiplications" but each is actually a product of linear forms.
    # The total rank is 7 because each Mi is a single scalar multiplication in the algorithm,
    # but when expanded, each Mi involves multiple scalar products that share the same
    # linear combination structure.
    
    # The correct way: each of the 7 Strassen products corresponds to one rank-1 term
    # in the tensor decomposition, where the U and V vectors are the coefficients of
    # the linear forms, and W is the contribution to C.
    
    # Let me define the linear forms:
    # L1_A = a + d  (coeff: [1,0,0,1] for [a,b,c,d])
    # L1_B = e + h  (coeff: [1,0,0,1] for [e,f,g,h])
    # L2_A = c + d  (coeff: [0,0,1,1])
    # L2_B = e      (coeff: [1,0,0,0])
    # L3_A = a      (coeff: [1,0,0,0])
    # L3_B = f - h  (coeff: [0,1,0,-1])
    # L4_A = d      (coeff: [0,0,0,1])
    # L4_B = g - e  (coeff: [-1,0,1,0])
    # L5_A = a + b  (coeff: [1,1,0,0])
    # L5_B = h      (coeff: [0,0,0,1])
    # L6_A = c - a  (coeff: [-1,0,1,0])
    # L6_B = e + f  (coeff: [1,1,0,0])
    # L7_A = b - d  (coeff: [0,1,0,-1])
    # L7_B = g + h  (coeff: [0,0,1,1])
    
    # Contributions to C:
    # C11 = M1 - M4 + M6 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M3 + M5 - M6
    
    # So for each term r (0-6), the W vector (for C indices [C11,C12,C21,C22] = [0,1,2,3]):
    # Term 0 (M1): W = [1, 0, 0, 1]
    # Term 1 (M2): W = [0, 0, 1, 0]
    # Term 2 (M3): W = [0, 1, 0, -1]
    # Term 3 (M4): W = [-1, 0, 1, 0]
    # Term 4 (M5): W = [0, 1, 0, 1]
    # Term 5 (M6): W = [1, 0, 0, -1]
    # Term 6 (M7): W = [-1, 0, 0, 0]
    
    U[0, 0] = 1  # a
    U[3, 0] = 1  # d
    V[0, 0] = 1  # e
    V[3, 0] = 1  # h
    W[0, 0] = 1  # C11
    W[3, 0] = 1  # C22
    
    U[2, 1] = 1  # c
    U[3, 1] = 1  # d
    V[0, 1] = 1  # e
    W[2, 1] = 1  # C21
    
    U[0, 2] = 1  # a
    V[1, 2] = 1  # f
    V[3, 2] = -1 # h
    W[1, 2] = 1  # C12
    W[3, 2] = -1 # C22
    
    U[3, 3] = 1  # d
    V[2, 3] = 1  # g
    V[0, 3] = -1 # e
    W[0, 3] = -1 # C11
    W[2, 3] = 1  # C21
    
    U[0, 4] = 1  # a
    U[1, 4] = 1  # b
    V[3, 4] = 1  # h
    W[1, 4] = 1  # C12
    W[3, 4] = 1  # C22
    
    U[2, 5] = 1  # c
    U[0, 5] = -1 # a
    V[0, 5] = 1  # e
    V[1, 5] = 1  # f
    W[0, 5] = 1  # C11
    W[3, 5] = -1 # C22
    
    U[1, 6] = 1  # b
    U[3, 6] = -1 # d
    V[2, 6] = 1  # g
    V[3, 6] = 1  # h
    W[0, 6] = -1 # C11
    
    return U, V, W


def _strassen_4x4():
    """
    Build Strassen decomposition for 4x4x4 matrix multiplication.
    Uses recursive Strassen: 4x4 = 7 * (2x2 multiplications)
    Each 2x2 has rank 7, so total rank = 7*7 = 49.
    """
    n = m = p = 4
    rank = 49
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    U2, V2, W2 = _strassen_2x2()
    
    # 4x4 matrices are divided into four 2x2 blocks:
    # A = [[A11, A12], [A21, A22]] where each Ai is 2x2
    # B = [[B11, B12], [B21, B22]]
    # C = [[C11, C12], [C21, C22]]
    
    # Block indices:
    # A11: rows 0-1, cols 0-1 -> A[0:2, 0:2]
    # A12: rows 0-1, cols 2-3 -> A[0:2, 2:4]
    # A21: rows 2-3, cols 0-1 -> A[2:4, 0:2]
    # A22: rows 2-3, cols 2-3 -> A[2:4, 2:4]
    
    # For each of the 7 Strassen products at the 4x4 level, we need to
    # compute the linear forms on A and B, which are sums of 2x2 blocks.
    # Each such linear form, when multiplied, gives a product of two 2x2 matrices,
    # which we decompose using the 2x2 Strassen decomposition.
    
    # The 7 Strassen products at 4x4 level:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 - M4 + M6 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M3 + M5 - M6
    
    # For each Mi, the A-part is a sum of 2x2 blocks, and B-part is a sum of 2x2 blocks.
    # The product Mi = (sum of A blocks)(sum of B blocks) is a sum of products of individual blocks.
    # Each such product is a 2x2 matrix multiplication, which we decompose with rank 7.
    
    # So for each Mi, we enumerate all pairs of (A_block, B_block) with their coefficients,
    # and for each pair, we add the corresponding 2x2 Strassen terms.
    
    def add_2x2_term(U, V, W, col, a_block, b_block, c_block, scale_a, scale_b, scale_c):
        """
        Add a 2x2 Strassen term to the global matrices.
        a_block, b_block, c_block: (row_start, col_start) for A, B, C blocks
        scale_a, scale_b: coefficients for the A and B linear forms
        scale_c: coefficient for the C contribution
        col: starting column in U, V, W
        """
        for r in range(7):
            for i in range(2):
                for j in range(2):
                    a_idx = (a_block[0] + i) * 4 + (a_block[1] + j)
                    if U2[i * 2 + j, r] != 0:
                        U[a_idx, col + r] += scale_a * U2[i * 2 + j, r]
            for j in range(2):
                for k in range(2):
                    b_idx = (b_block[0] + j) * 4 + (b_block[1] + k)
                    if V2[j * 2 + k, r] != 0:
                        V[b_idx, col + r] += scale_b * V2[j * 2 + k, r]
            for k in range(2):
                for i in range(2):
                    c_idx = (c_block[0] + k) * 4 + (c_block[1] + i)
                    if W2[k * 2 + i, r] != 0:
                        W[c_idx, col + r] += scale_c * W2[k * 2 + i, r]
    
    col = 0
    # Define the 7 Strassen products with their block structure
    # Each product is a list of (a_block, b_block, c_block, scale_a, scale_b, scale_c)
    
    # M1 = (A11+A22)(B11+B22)
    # A11: (0,0), A22: (2,2)
    # B11: (0,0), B22: (2,2)
    # C11: (0,0), C22: (2,2)
    m1_blocks = [
        ((0,0), (0,0), (0,0), 1, 1, 1),  # A11*B11 -> C11
        ((0,0), (2,2), (0,0), 1, 1, 1),  # A11*B22 -> C11
        ((2,2), (0,0), (0,0), 1, 1, 1),  # A22*B11 -> C11
        ((2,2), (2,2), (0,0), 1, 1, 1),  # A22*B22 -> C11
        ((0,0), (0,0), (2,2), 1, 1, 1),  # A11*B11 -> C22
        ((0,0), (2,2), (2,2), 1, 1, 1),  # A11*B22 -> C22
        ((2,2), (0,0), (2,2), 1, 1, 1),  # A22*B11 -> C22
        ((2,2), (2,2), (2,2), 1, 1, 1),  # A22*B22 -> C22
    ]
    
    # Wait, this is getting complicated. Each Mi is a product of two 2x2 matrix sums.
    # (A11+A22)(B11+B22) = A11*B11 + A11*B22 + A22*B11 + A22*B22
    # Each term is a 2x2 matrix product, contributing to both C11 and C22 with the same value.
    # So M1 contributes to C11 with coefficient 1 and C22 with coefficient 1.
    # But each of the 4 sub-products needs its own 2x2 Strassen decomposition.
    # So M1 uses 4*7 = 28 rank terms? No, that's not right either.
    
    # Actually, the Strassen algorithm at the 4x4 level uses 7 multiplications of 2x2 matrices.
    # Each 2x2 multiplication has rank 7. So total rank = 7*7 = 49.
    # The key is that each of the 7 Strassen products at the 4x4 level is a single
    # 2x2 matrix multiplication (not a sum of multiple 2x2 multiplications).
    # The linear forms at the 4x4 level are sums of 2x2 blocks, but the product
    # is treated as a single 2x2 matrix product.
    
    # So for M1 = (A11+A22)(B11+B22), the A-part is the 2x2 matrix (A11+A22) and
    # the B-part is the 2x2 matrix (B11+B22). The product is a 2x2 matrix that
    # contributes to C11 and C22.
    
    # To decompose this, we need to express (A11+A22) as a linear combination of
    # the 4 elements of a 2x2 matrix, and similarly for (B11+B22).
    # The elements of (A11+A22) are:
    #   [0,0]: A[0,0] + A[2,2]
    #   [0,1]: A[0,1] + A[2,3]
    #   [1,0]: A[1,0] + A[3,2]
    #   [1,1]: A[1,1] + A[3,3]
    
    # So for M1, the U vector (for the 2x2 A-part) has:
    #   U[0,0] = 1 (A[0,0]), U[0,1] = 1 (A[0,1]), U[1,0] = 1 (A[1,0]), U[1,1] = 1 (A[1,1])
    #   U[2,2] = 1 (A[2,2]), U[2,3] = 1 (A[2,3]), U[3,2] = 1 (A[3,2]), U[3,3] = 1 (A[3,3])
    # In the global U matrix (shape 16x49), for column col+r (r=0..6):
    #   U[0, col+r] += U2[0,r]  (A[0,0])
    #   U[1, col+r] += U2[1,r]  (A[0,1])
    #   U[2, col+r] += U2[2,r]  (A[1,0])
    #   U[3, col+r] += U2[3,r]  (A[1,1])
    #   U[8, col+r] += U2[0,r]  (A[2,2])
    #   U[9, col+r] += U2[1,r]  (A[2,3])
    #   U[10, col+r] += U2[2,r] (A[3,2])
    #   U[11, col+r] += U2[3,r] (A[3,3])
    
    # Similarly for V and W.
    
    # Let me define a helper that, given the A-block offsets, B-block offsets,
    # and C-block offsets (with signs), adds the 2x2 Strassen terms.
    
    def add_strassen_term(U, V, W, col, a_offsets, b_offsets, c_offsets, sign_c):
        """
        a_offsets: list of (row, col) pairs for A elements with their signs
        b_offsets: list of (row, col) pairs for B elements with their signs  
        c_offsets: list of (row, col) pairs for C elements with their signs
        """
        for r in range(7):
            for (ar, ac) in a_offsets:
                a_idx = ar * 4 + ac
                U[a_idx, col + r] += U2[ar % 2 * 2 + ac % 2, r] if False else 0
            # This is getting too complex. Let me use a different approach.
        pass
    
    # Let me just directly construct the 49 terms by iterating over the 7 Strassen
    # products at the 4x4 level and for each, the 7 Strassen products at the 2x2 level.
    
    # The 7 Strassen products at 4x4 level, with their A and B block contributions:
    # Each product has A as a sum of 2x2 blocks and B as a sum of 2x2 blocks.
    # The A-blocks are (block_row, block_col) where block_row in {0,1}, block_col in {0,1}
    # A[0,0] = A11, A[0,1] = A12, A[1,0] = A21, A[1,1] = A22
    
    strassen_4x4_products = [
        # (A_blocks_with_signs, B_blocks_with_signs, C_blocks_with_signs)
        # A_blocks: list of (block_row, block_col, sign)
        # B_blocks: list of (block_row, block_col, sign)
        # C_blocks: list of (block_row, block_col, sign)
        # M1 = (A11+A22)(B11+B22), C11 += M1, C22 += M1
        ([((0,0,1),(1,1,1))], [((0,0,1),(1,1,1))], [((0,0,1),(1,1,1))]),
        # M2 = (A21+A22)B11, C21 += M2
        ([((1,0,1),(1,1,1))], [((0,0,1))], [((1,0,1))]),
        # M3 = A11(B21-B22), C12 += M3, C22 -= M3
        ([((0,0,1))], [((1,0,1),(1,1,-1))], [((0,1,1),(1,1,-1))]),
        # M4 = A22(B12-B11), C11 -= M4, C21 += M4
        ([((1,1,1))], [((0,1,1),(0,0,-1))], [((0,0,-1),(1,0,1))]),
        # M5 = (A11+A12)B22, C12 += M5, C22 += M5
        ([((0,0,1),(0,1,1))], [((1,1,1))], [((0,1,1),(1,1,1))]),
        # M6 = (A21-A11)(B11+B12), C11 += M6, C22 -= M6
        ([((1,0,1),(0,0,-1))], [((0,0,1),(0,1,1))], [((0,0,1),(1,1,-1))]),
        # M7 = (A12-A22)(B21+B22), C11 -= M7
        ([((0,1,1),(1,1,-1))], [((1,0,1),(1,1,1))], [((0,0,-1))]),
    ]
    
    col = 0
    for a_blocks, b_blocks, c_blocks in strassen_4x4_products:
        for r in range(7):
            # For each Strassen 2x2 term r, add contributions
            # A-part: sum over a_blocks of sign * U2[element, r]
            # The A-block (br, bc) corresponds to A elements at rows br*2..br*2+1, cols bc*2..bc*2+1
            for (abr, abc, asgn) in a_blocks:
                for i in range(2):
                    for j in range(2):
                        a_row = abr * 2 + i
                        a_col = abc * 2 + j
                        a_idx = a_row * 4 + a_col
                        U[a_idx, col + r] += asgn * U2[i * 2 + j, r]
            
            for (bbr, bbc, bsgn) in b_blocks:
                for i in range(2):
                    for j in range(2):
                        b_row = bbr * 2 + i
                        b_col = bbc * 2 + j
                        b_idx = b_row * 4 + b_col
                        V[b_idx, col + r] += bsgn * V2[i * 2 + j, r]
            
            for (cbr, cbc, csgn) in c_blocks:
                for i in range(2):
                    for j in range(2):
                        c_row = cbr * 2 + i
                        c_col = cbc * 2 + j
                        c_idx = c_row * 4 + c_col
                        W[c_idx, col + r] += csgn * W2[i * 2 + j, r]
        
        col += 7
    
    return U, V, W


def construct():
    n, m, p = DIMENSIONS
    # 5 = 4 + 1
    # Decompose 5x5x5 into:
    # 1. 4x4x4 core (indices 0-3) -> Strassen rank 49
    # 2. Remaining parts involving index 4 -> naive
    
    # Total rank = 49 + (number of naive terms)
    # Naive terms:
    # A[0:4,4] * B[4,0:4] -> 4x1x4 = 16 terms
    # A[0:4,4] * B[4,4] -> 4x1x1 = 4 terms  
    # A[4,0:4] * B[0:4,0:4] -> 1x4x4 = 16 terms
    # A[4,0:4] * B[0:4,4] -> 1x4x1 = 4 terms
    # A[4,4] * B[4,0:4] -> 1x1x4 = 4 terms
    # A[4,4] * B[4,4] -> 1x1x1 = 1 term
    # A[4,0:4] * B[4,0:4] -> 1x4x4 = 16 terms (wait, this is A[4,j]*B[j,k] for j,k in 0..3)
    
    # Let me be more careful. The 5x5 multiplication:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # Split i, j, k into {0,1,2,3} and {4}
    
    # Case 1: i in 0..3, j in 0..3, k in 0..3 -> 4x4x4 core, rank 49
    # Case 2: i in 0..3, j = 4, k in 0..3 -> A[i,4]*B[4,k], 4*1*4 = 16 terms
    # Case 3: i in 0..3, j = 4, k = 4 -> A[i,4]*B[4,4], 4*1*1 = 4 terms
    # Case 4: i = 4, j in 0..3, k in 0..3 -> A[4,j]*B[j,k], 1*4*4 = 16 terms
    # Case 5: i = 4, j in 0..3, k = 4 -> A[4,j]*B[j,4], 1*4*1 = 4 terms
    # Case 6: i = 4, j = 4, k in 0..3 -> A[4,4]*B[4,k], 1*1*4 = 4 terms
    # Case 7: i = 4, j = 4, k = 4 -> A[4,4]*B[4,4], 1*1*1 = 1 term
    
    # Total naive = 16+4+16+4+4+1 = 45
    # Wait, I double-counted. Let me list all (i,j,k) where at least one is 4:
    # Total = 125 - 64 = 61
    # So naive rank = 61
    
    # Total rank = 49 + 61 = 110
    
    rank = 110
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Add Strassen 4x4x4 core
    U_core, V_core, W_core = _strassen_4x4()
    U[:, :49] = U_core
    V[:, :49] = V_core
    W[:, :49] = W_core
    
    # Add naive terms for the remaining 61 (i,j,k) where at least one index is 4
    col = 49
    for i in range(5):
        for j in range(5):
            for k in range(5):
                if i < 4 and j < 4 and k < 4:
                    continue  # Already handled by Strassen
                U[i * 5 + j, col] = 1
                V[j * 5 + k, col] = 1
                W[k * 5 + i, col] = 1
                col += 1
    
    # Convert to lists for the required interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}


if __name__ == "__main__":
    result = construct()
    U = np.array(result["U"])
    V = np.array(result["V"])
    W = np.array(result["W"])
    print(f"U shape: {U.shape}")
    print(f"V shape: {V.shape}")
    print(f"W shape: {W.shape}")
    print(f"Rank: {U.shape[1]}")
    
    # Verify by checking that sum_r U[:,r] outer V[:,r] outer W[:,r] equals the multiplication tensor
    n, m, p = DIMENSIONS
    # The multiplication tensor T[i,j,k,r] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Sum over r should give 1 if this is a valid multiplication term, else 0
    # Actually, the tensor decomposition means:
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # And T[i,j,k] = 1 for all i,j,k (the multiplication tensor has 1s everywhere)
    # Wait no, the multiplication tensor for matrix multiplication is:
    # T[i,j,k,l] = delta[i,l] * delta[j,k] ... no.
    # The matrix multiplication tensor is T[i,j,k] where C[i,k] = sum_j A[i,j] B[j,k]
    # So T[i,j,k] = 1 for all valid i,j,k, and the tensor is sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    # But W is indexed as W[k*n+i, r], so W[k,i,r].
    
    # Let's verify:
    error = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(U.shape[1]):
                    s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                expected = 1
                if abs(s - expected) > 1e-9:
                    error += 1
    print(f"Verification errors: {error}")
