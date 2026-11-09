import numpy as np

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We split the 3x6 matrix B into two 3x3 matrices B1 and B2.
    # C = A @ B = [A @ B1, A @ B2]
    # We use a rank-20 decomposition for 3x3 matrix multiplication.
    
    # 1. Define the Rank-20 decomposition for 3x3x3
    # Inputs A (3x3), B (3x3).
    # A = [[A11 (2x2), A12 (2x1)], [A21 (1x2), A22 (1x1)]]
    # B = [[B11 (2x2), B12 (2x1)], [B21 (1x2), B22 (1x1)]]
    
    # Sub-products:
    # 1. A11 @ B11 (2x2) -> Rank 7 (Strassen)
    # 2. A22 @ B22 (1x1) -> Rank 1
    # 3. A12 @ B21 (2x1 * 1x2 = 2x2) -> Rank 4 (Outer product)
    # 4. A12 @ B22 (2x1 * 1x1 = 2x1) -> Rank 2
    # 5. A21 @ B12 (1x2 * 2x1 = 1x1) -> Rank 1? 
    #    Wait, A21 is 1x2, B12 is 2x1. Product is scalar. Rank 2?
    #    C = sum A21_i B12_i. This is inner product.
    #    Rank of inner product of two vectors of length 2 is 2?
    #    $v \cdot w = v_1 w_1 + v_2 w_2$. Rank 2.
    # 6. A21 @ B11 (1x2 * 2x2 = 1x2) -> Rank 2
    # 7. A11 @ B12 (2x2 * 2x1 = 2x1) -> Rank 2
    
    # Total Rank = 7 + 1 + 4 + 2 + 2 + 2 + 2 = 20.
    
    # We construct the U, V, W matrices for this 3x3x3 decomposition.
    # Then we duplicate it for the two halves of B.
    
    rank_333 = 20
    rank_total = 2 * rank_333
    
    U = [[0] * rank_total for _ in range(n * m)]
    V = [[0] * rank_total for _ in range(m * p)]
    W = [[0] * rank_total for _ in range(p * n)]
    
    # Helper to add a decomposition for a sub-block
    def add_decomposition(offset_u, offset_v, offset_w, sub_rank, 
                          u_indices, v_indices, w_indices, 
                          a_indices, b_indices, c_indices):
        """
        u_indices: list of (row_in_U, coeff)
        v_indices: list of (row_in_V, coeff)
        w_indices: list of (row_in_W, coeff)
        a_indices: list of (row_in_A, col_in_A)
        b_indices: list of (row_in_B, col_in_B)
        c_indices: list of (row_in_C, col_in_C)
        """
        # This helper is too abstract. I will explicitly build the factors.
        pass

    # Explicit Construction for 3x3x3 Rank 20
    # We will generate the factors for one 3x3x3 multiplication and then shift them.
    
    # Strassen for 2x2 (A11, B11)
    # A11 indices in A: (0,0), (0,1), (1,0), (1,1)
    # B11 indices in B: (0,0), (0,1), (1,0), (1,1)
    
    # We will build the U, V, W for the first 3x3x3 part (B1)
    # U rows: 0..8 (9 rows for 3x3 A)
    # V rows: 0..17 (18 rows for 3x6 B, but we only use first 9 for B1)
    # W rows: 0..17 (18 rows for 3x6 C, but we only use first 9 for C1)
    
    col_offset = 0
    
    # 1. A11 @ B11 (Strassen)
    # Strassen factors for 2x2:
    # M1 = (a+e)(g+k)
    # M2 = (d+f)(g+k)
    # M3 = (a+b)(l)
    # M4 = (e+f)(m)
    # M5 = (a+d)(k-l)
    # M6 = (b+c)(g+d)
    # M7 = (c+f)(m-h)
    # A11 = [[a, b], [d, e]] (indices 0,1,3,4 in flat 3x3? No, 0,1,3,4 are 0,0; 0,1; 1,0; 1,1)
    # B11 = [[g, h], [j, k]] (indices 0,1,3,4 in flat 3x3)
    
    # Flat indices for 3x3 A:
    # 0: (0,0) a, 1: (0,1) b, 2: (0,2) c2
    # 3: (1,0) d, 4: (1,1) e, 5: (1,2) f2
    # 6: (2,0) g2, 7: (2,1) h2, 8: (2,2) i2
    
    # A11: [[a, b], [d, e]] -> indices 0, 1, 3, 4
    # B11: [[g, h], [j, k]] -> indices 0, 1, 3, 4
    
    # Strassen U,V,W for 2x2 (relative to A11, B11, C11)
    # C11 is 2x2 at indices 0,1,3,4 of C.
    
    # I will use a known rank-7 set of coefficients.
    # To be safe, I will use the standard Strassen.
    
    strassen_u = [
        {0: 1, 3: 1}, # a+d
        {0: 1, 1: 1}, # a+b
        {1: 1, 4: 1}, # b+e
        {0: 1},       # a
        {4: 1, 3: -1},# e-d
        {3: 1, 4: 1}, # d+e
        {0: 1, 1: 1, 3: 1, 4: 1} # a+b+d+e
    ]
    strassen_v = [
        {0: 1, 3: 1}, # g+j
        {3: 1},       # j
        {0: 1},       # g
        {1: 1, 3: 1}, # h+j
        {3: 1, 0: -1},# j-g
        {0: 1, 3: 1}, # g+j (Wait, standard Strassen varies)
        {1: 1, 0: -1} # h-g
    ]
    # This is getting error-prone.
    
    # Let's use a simpler Rank 9 decomposition for 2x2? No, Rank 7 is standard.
    # I will use the Schoolbook for 2x2 (Rank 4) to ensure correctness, then the total rank is:
    # 1. A11 @ B11: Rank 4
    # 2. A22 @ B22: Rank 1
    # 3. A12 @ B21: Rank 4
    # 4. A12 @ B22: Rank 2
    # 5. A21 @ B12: Rank 2
    # 6. A21 @ B11: Rank 2
    # 7. A11 @ B12: Rank 2
    # Total = 4 + 1 + 4 + 2 + 2 + 2 + 2 = 17.
    # 17 < 20.
    # 2 * 17 = 34.
    # Score = 54 / 34 = 1.58.
    # This is very safe and simple.

    # I will implement the Rank 17 decomposition for 3x3x3.
    
    def add_schoolbook_block(A_idx, B_idx, C_idx, rank_offset):
        # A_idx: list of 2 (row, col) in A
        # B_idx: list of 2 (row, col) in B
        # C_idx: list of 2 (row, col) in C
        # Assumes 1x1 blocks? No, this is for specific linear forms.
        pass

    # I will explicitly write out the 17 multiplications.
    
    # Multiplication 1-4: A11 @ B11 (2x2 schoolbook)
    # A11: [[a,b],[d,e]] (0,0; 0,1; 1,0; 1,1)
    # B11: [[g,h],[j,k]] (0,0; 0,1; 1,0; 1,1)
    # C11: [[c11, c12], [c21, c22]] (0,0; 0,1; 1,0; 1,1)
    
    # 1: a*g -> C11[0,0]
    U[0][col_offset] = 1; V[0][col_offset] = 1; W[0][col_offset] = 1; col_offset += 1
    # 2: b*h -> C11[0,1]
    U[1][col_offset] = 1; V[1][col_offset] = 1; W[1][col_offset] = 1; col_offset += 1
    # 3: d*j -> C11[1,0]
    U[3][col_offset] = 1; V[3][col_offset] = 1; W[3][col_offset] = 1; col_offset += 1
    # 4: e*k -> C11[1,1]
    U[4][col_offset] = 1; V[4][col_offset] = 1; W[4][col_offset] = 1; col_offset += 1
    
    # 5: A22 @ B22 (1x1)
    # A22: i2 (2,2) -> U[8]
    # B22: k2 (2,2) -> V[8]
    # C22: c22 (2,2) -> W[8]
    U[8][col_offset] = 1; V[8][col_offset] = 1; W[8][col_offset] = 1; col_offset += 1
    
    # 6-9: A12 @ B21 (2x2 outer)
    # A12: [[b2],[f2]] (0,2; 1,2) -> U[2], U[5]
    # B21: [[g2, h2]] (2,0; 2,1) -> V[6], V[7]
    # C12 (part): (0,0; 0,1; 1,0; 1,1)
    # C[0,0] += b2*g2
    U[2][col_offset] = 1; V[6][col_offset] = 1; W[0][col_offset] = 1; col_offset += 1
    # C[0,1] += b2*h2
    U[2][col_offset] = 1; V[7][col_offset] = 1; W[1][col_offset] = 1; col_offset += 1
    # C[1,0] += f2*g2
    U[5][col_offset] = 1; V[6][col_offset] = 1; W[3][col_offset] = 1; col_offset += 1
    # C[1,1] += f2*h2
    U[5][col_offset] = 1; V[7][col_offset] = 1; W[4][col_offset] = 1; col_offset += 1
    
    # 10-11: A12 @ B22 (2x1)
    # A12: b2, f2 (U[2], U[5])
    # B22: k2 (V[8])
    # C[0,2] += b2*k2
    U[2][col_offset] = 1; V[8][col_offset] = 1; W[2][col_offset] = 1; col_offset += 1
    # C[1,2] += f2*k2
    U[5][col_offset] = 1; V[8][col_offset] = 1; W[5][col_offset] = 1; col_offset += 1
    
    # 12-13: A21 @ B12 (1x1 inner)
    # A21: g2, h2 (U[6], U[7])
    # B12: h, f? No, B12 is (0,2; 1,2) -> V[2], V[5]
    # C[2,0] += g2*B[0,2] + h2*B[1,2]
    # This is rank 2.
    # 12: g2 * B[0,2]
    U[6][col_offset] = 1; V[2][col_offset] = 1; W[6][col_offset] = 1; col_offset += 1
    # 13: h2 * B[1,2]
    U[7][col_offset] = 1; V[5][col_offset] = 1; W[6][col_offset] = 1; col_offset += 1
    
    # 14-15: A21 @ B11 (1x2)
    # A21: g2, h2 (U[6], U[7])
    # B11: g, h (V[0], V[1])
    # C[2,0] += g2*g
    U[6][col_offset] = 1; V[0][col_offset] = 1; W[6][col_offset] = 1; col_offset += 1
    # C[2,1] += h2*h
    U[7][col_offset] = 1; V[1][col_offset] = 1; W[7][col_offset] = 1; col_offset += 1
    # Wait, C[2,0] also has g2*h?
    # C[2,0] = A[2,0]B[0,0] + A[2,1]B[1,0] = g2*g + h2*j
    # I missed h2*j.
    # Let's restart the A21 @ B11 part.
    # A21: g2, h2. B11: g, h, j, k.
    # C[2,0] = g2*g + h2*j
    # C[2,1] = g2*h + h2*k
    
    # 14: g2*g
    U[6][col_offset] = 1; V[0][col_offset] = 1; W[6][col_offset] = 1; col_offset += 1
    # 15: h2*j
    U[7][col_offset] = 1; V[3][col_offset] = 1; W[6][col_offset] = 1; col_offset += 1
    # 16: g2*h
    U[6][col_offset] = 1; V[1][col_offset] = 1; W[7][col_offset] = 1; col_offset += 1
    # 17: h2*k
    U[7][col_offset] = 1; V[4][col_offset] = 1; W[7][col_offset] = 1; col_offset += 1
    
    # 18-19: A11 @ B12 (2x1)
    # A11: a,b,d,e (U[0,1,3,4])
    # B12: h,f? B[0,2], B[1,2] (V[2], V[5])
    # C[0,2] = a*B[0,2] + b*B[1,2]
    # C[1,2] = d*B[0,2] + e*B[1,2]
    
    # 18: a*B[0,2]
    U[0][col_offset] = 1; V[2][col_offset] = 1; W[2][col_offset] = 1; col_offset += 1
    # 19: b*B[1,2]
    U[1][col_offset] = 1; V[5][col_offset] = 1; W[2][col_offset] = 1; col_offset += 1
    # 20: d*B[0,2]
    U[3][col_offset] = 1; V[2][col_offset] = 1; W[5][col_offset] = 1; col_offset += 1
    # 21: e*B[1,2]
    U[4][col_offset] = 1; V[5][col_offset] = 1; W[5][col_offset] = 1; col_offset += 1
    
    # Total Rank 21?
    # 4 (A11B11) + 1 (A22B22) + 4 (A12B21) + 2 (A12B22) + 2 (A21B12) + 4 (A21B11) + 4 (A11B12) = 21.
    # 2 * 21 = 42.
    # Score = 54 / 42 = 1.28.
    
    # Now duplicate for the second half of B (columns 3-5 of B, columns 3-5 of C).
    # B indices shift by 3 in column?
    # B is 3x6. B1 is cols 0-2. B2 is cols 3-5.
    # V indices for B2:
    # B2[0,0] is B[0,3] -> V[3]
    # B2[0,1] is B[0,4] -> V[4]
    # B2[0,2] is B[0,5] -> V[5]
    # ...
    
    # C indices shift by 3 in column?
    # C is 3x6. C1 is cols 0-2. C2 is cols 3-5.
    # W indices for C2:
    # C2[0,0] is C[0,3] -> W[3]
    
    # I will write a helper to add the second block with shifted V and W indices.
    
    def add_second_block():
        nonlocal col_offset
        # Repeat the 21 multiplications
        # 1-4: A11 @ B21 (2x2)
        # B21 indices: 3,4,6,7
        U[0][col_offset] = 1; V[3][col_offset] = 1; W[3][col_offset] = 1; col_offset += 1
        U[1][col_offset] = 1; V[4][col_offset] = 1; W[4][col_offset] = 1; col_offset += 1
        U[3][col_offset] = 1; V[6][col_offset] = 1; W[6][col_offset] = 1; col_offset += 1
        U[4][col_offset] = 1; V[7][col_offset] = 1; W[7][col_offset] = 1; col_offset += 1
        
        # 5: A22 @ B22 (1x1)
        # B22 is B[2,5] -> V[11]
        # C22 is C[2,5] -> W[11]
        U[8][col_offset] = 1; V[11][col_offset] = 1; W[11][col_offset] = 1; col_offset += 1
        
        # 6-9: A12 @ B21 (2x2)
        # B21 is 1x2 at B[2,3], B[2,4] -> V[9], V[10]
        U[2][col_offset] = 1; V[9][col_offset] = 1; W[3][col_offset] = 1; col_offset += 1
        U[2][col_offset] = 1; V[10][col_offset] = 1; W[4][col_offset] = 1; col_offset += 1
        U[5][col_offset] = 1; V[9][col_offset] = 1; W[6][col_offset] = 1; col_offset += 1
        U[5][col_offset] = 1; V[10][col_offset] = 1; W[7][col_offset] = 1; col_offset += 1
        
        # 10-11: A12 @ B22 (2x1)
        U[2][col_offset] = 1; V[11][col_offset] = 1; W[5][col_offset] = 1; col_offset += 1
        U[5][col_offset] = 1; V[11][col_offset] = 1; W[8][col_offset] = 1; col_offset += 1
        
        # 12-13: A21 @ B12 (1x1)
        # B12 for B2 is B[0,5], B[1,5] -> V[5], V[8]
        U[6][col_offset] = 1; V[5][col_offset] = 1; W[9][col_offset] = 1; col_offset += 1
        U[7][col_offset] = 1; V[8][col_offset] = 1; W[9][col_offset] = 1; col_offset += 1
        
        # 14-17: A21 @ B11 (1x2)
        # B11 for B2 is B[0,3], B[0,4], B[1,3], B[1,4] -> V[3], V[4], V[6], V[7]
        U[6][col_offset] = 1; V[3][col_offset] = 1; W[9][col_offset] = 1; col_offset += 1
        U[7][col_offset] = 1; V[6][col_offset] = 1; W[9][col_offset] = 1; col_offset += 1
        U[6][col_offset] = 1; V[4][col_offset] = 1; W[10][col_offset] = 1; col_offset += 1
        U[7][col_offset] = 1; V[7][col_offset] = 1; W[10][col_offset] = 1; col_offset += 1
        
        # 18-21: A11 @ B12 (2x1)
        # B12 for B2 is B[0,5], B[1,5] -> V[5], V[8]
        U[0][col_offset] = 1; V[5][col_offset] = 1; W[5][col_offset] = 1; col_offset += 1
        U[1][col_offset] = 1; V[8][col_offset] = 1; W[5][col_offset] = 1; col_offset += 1
        U[3][col_offset] = 1; V[5][col_offset] = 1; W[8][col_offset] = 1; col_offset += 1
        U[4][col_offset] = 1; V[8][col_offset] = 1; W[8][col_offset] = 1; col_offset += 1

    add_second_block()
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
