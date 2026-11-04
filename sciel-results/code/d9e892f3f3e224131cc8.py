"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We decompose the 4x4 multiplication into two 2x4 multiplications
    # by splitting the middle dimension (m=4) into 2+2.
    # C = A[:, 0:2] B[0:2, :] + A[:, 2:4] B[2:4, :]
    # Each term is a 3x2 by 2x4 multiplication.
    # Rank of 3x2 by 2x4 is 22 (14 for 2x2x2 part + 8 for 1x2x2 part).
    # Total rank = 22 + 22 = 44.
    
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a Strassen 2x2x2 decomposition
    # Inputs: 
    #   U_base: index offset for U (n*m)
    #   V_base: index offset for V (m*p)
    #   W_base: index offset for W (p*n)
    #   i_off, j_off, k_off: offsets for the 2x2 block in the global matrices
    #   U is indexed [i*m + j]
    #   V is indexed [j*p + k]
    #   W is indexed [k*n + i]
    # Block is A[i_off:i_off+2, j_off:j_off+2] * B[j_off:j_off+2, k_off:k_off+2]
    # Result added to C[i_off:i_off+2, k_off:k_off+2]
    
    def add_strassen_2x2x2(U, V, W, i_off, j_off, k_off, col_start, n, m, p):
        # Strassen 7 products for 2x2 * 2x2
        # M1 = (A11 + A22)(B11 + B22)
        # M2 = (A21 + A22)B11
        # M3 = A11(B21 - B22)
        # M4 = A22(B12 - B11)
        # M5 = (A11 + A12)B22
        # M6 = (A21 - A11)(B11 + B12)
        # M7 = (A12 - A22)(B21 + B22)
        
        # U indices for A block
        # A11: (i_off, j_off) -> idx (i_off)*m + j_off
        # A12: (i_off, j_off+1)
        # A21: (i_off+1, j_off)
        # A22: (i_off+1, j_off+1)
        
        u11 = i_off * m + j_off
        u12 = i_off * m + (j_off + 1)
        u21 = (i_off + 1) * m + j_off
        u22 = (i_off + 1) * m + (j_off + 1)
        
        # V indices for B block
        # B11: (j_off, k_off) -> idx j_off*p + k_off
        # B12: (j_off, k_off+1)
        # B21: (j_off+1, k_off)
        # B22: (j_off+1, k_off+1)
        
        v11 = j_off * p + k_off
        v12 = j_off * p + (k_off + 1)
        v21 = (j_off + 1) * p + k_off
        v22 = (j_off + 1) * p + (k_off + 1)
        
        # W indices for C block
        # C11: (k_off, i_off) -> idx k_off*n + i_off
        # C12: (k_off+1, i_off)
        # C21: (k_off, i_off+1)
        # C22: (k_off+1, i_off+1)
        
        w11 = k_off * n + i_off
        w12 = k_off * n + (i_off + 1)
        w21 = (k_off + 1) * n + i_off
        w22 = (k_off + 1) * n + (i_off + 1)
        
        # M1
        c = col_start
        U[u11][c] += 1; V[v11][c] += 1; W[w11][c] += 1
        U[u11][c] += 1; V[v22][c] += 1; W[w11][c] += 1
        U[u22][c] += 1; V[v11][c] += 1; W[w22][c] += 1
        U[u22][c] += 1; V[v22][c] += 1; W[w22][c] += 1
        # Wait, Strassen M1 = (A11+A22)(B11+B22).
        # Expansion: A11B11 + A11B22 + A22B11 + A22B22.
        # We need to add these to the correct C positions.
        # C11 gets A11B11. C12 gets A11B22? No.
        # C = [[A11B11+A12B21, A11B12+A12B22],
        #      [A21B11+A22B21, A21B12+A22B22]]
        
        # Standard Strassen:
        # C11 = M1 - M4 + M5 - M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # Let's implement the 7 products directly into the tensors.
        # Product 1: (A11 + A22) * (B11 + B22)
        # This contributes to:
        # A11B11 -> C11
        # A11B22 -> C12 (if B22 is at col 1? No, B22 is row 1 col 1 of block)
        # Let's map indices carefully.
        # B11 is (0,0), B12 is (0,1), B21 is (1,0), B22 is (1,1).
        # A11B11 -> C(0,0)
        # A11B22 -> C(0,1)
        # A22B11 -> C(1,0)
        # A22B22 -> C(1,1)
        
        # So M1 adds:
        # 1 to U[A11], V[B11], W[C11]
        # 1 to U[A11], V[B22], W[C12]
        # 1 to U[A22], V[B11], W[C21]
        # 1 to U[A22], V[B22], W[C22]
        
        # But we are adding 4 separate terms to the rank-1 decomposition.
        # This is not efficient. Strassen combines them.
        # The rank-1 terms are the 7 products.
        # We need to define the 7 products as rank-1 tensors.
        
        # P1 = (A11 + A22)(B11 + B22)
        # U: A11(1), A22(1)
        # V: B11(1), B22(1)
        # W: C11(1), C12(1), C21(1), C22(1) -- NO.
        # The product P1 is a scalar. It is used in linear combinations to form C.
        # The decomposition is sum of rank-1 tensors.
        # Each rank-1 tensor corresponds to one multiplication of a linear combo of A and a linear combo of B.
        # So for P1, we have one column in U, V, W.
        # U[A11] += 1, U[A22] += 1
        # V[B11] += 1, V[B22] += 1
        # W[?] += ?
        # The result of P1 is added to C11, C12, C21, C22 with coefficients.
        # C11 += 1*P1
        # C12 += 1*P1
        # C21 += 1*P1
        # C22 += 1*P1
        # So W[C11] += 1, W[C12] += 1, W[C21] += 1, W[C22] += 1.
        
        # This is correct for the tensor decomposition.
        
        # Let's define the 7 products.
        
        # 1. (A11 + A22)(B11 + B22)
        U[u11][c] += 1; U[u22][c] += 1
        V[v11][c] += 1; V[v22][c] += 1
        W[w11][c] += 1; W[w12][c] += 1; W[w21][c] += 1; W[w22][c] += 1
        c += 1
        
        # 2. (A21 + A22)B11
        U[u21][c] += 1; U[u22][c] += 1
        V[v11][c] += 1
        # C21 += 1*P2
        W[w21][c] += 1
        c += 1
        
        # 3. A11(B21 - B22)
        U[u11][c] += 1
        V[v21][c] += 1; V[v22][c] -= 1
        # C12 += 1*P3
        W[w12][c] += 1
        c += 1
        
        # 4. A22(B12 - B11)
        U[u22][c] += 1
        V[v12][c] += 1; V[v11][c] -= 1
        # C21 -= 1*P4
        W[w21][c] -= 1
        c += 1
        
        # 5. (A11 + A12)B22
        U[u11][c] += 1; U[u12][c] += 1
        V[v22][c] += 1
        # C11 -= 1*P5, C12 += 1*P5
        W[w11][c] -= 1; W[w12][c] += 1
        c += 1
        
        # 6. (A21 - A11)(B11 + B12)
        U[u21][c] += 1; U[u11][c] -= 1
        V[v11][c] += 1; V[v12][c] += 1
        # C22 += 1*P6
        W[w22][c] += 1
        c += 1
        
        # 7. (A12 - A22)(B21 + B22)
        U[u12][c] += 1; U[u22][c] -= 1
        V[v21][c] += 1; V[v22][c] += 1
        # C11 += 1*P7
        W[w11][c] += 1
        c += 1
        
        return c

    # Helper to add a 1x2x2 decomposition (Rank 4)
    # A is 1x2, B is 2x2. C is 1x2.
    # C1 = A1B11 + A2B21
    # C2 = A1B12 + A2B22
    # Schoolbook: 4 mults.
    def add_1x2x2(U, V, W, i_off, j_off, k_off, col_start, n, m, p):
        # A indices: (i_off, j_off), (i_off, j_off+1)
        u1 = i_off * m + j_off
        u2 = i_off * m + (j_off + 1)
        
        # B indices: (j_off, k_off), (j_off, k_off+1), (j_off+1, k_off), (j_off+1, k_off+1)
        v11 = j_off * p + k_off
        v12 = j_off * p + (k_off + 1)
        v21 = (j_off + 1) * p + k_off
        v22 = (j_off + 1) * p + (k_off + 1)
        
        # C indices: (k_off, i_off), (k_off+1, i_off)
        w1 = k_off * n + i_off
        w2 = (k_off + 1) * n + i_off
        
        c = col_start
        
        # M1 = A1 B11 -> C1
        U[u1][c] += 1; V[v11][c] += 1; W[w1][c] += 1
        c += 1
        
        # M2 = A2 B21 -> C1
        U[u2][c] += 1; V[v21][c] += 1; W[w1][c] += 1
        c += 1
        
        # M3 = A1 B12 -> C2
        U[u1][c] += 1; V[v12][c] += 1; W[w2][c] += 1
        c += 1
        
        # M4 = A2 B22 -> C2
        U[u2][c] += 1; V[v22][c] += 1; W[w2][c] += 1
        c += 1
        
        return c

    # Block 1: A[:, 0:2] * B[0:2, :]
    # i_off=0, j_off=0, k_off=0
    # A is 3x2. B is 2x4.
    # Split A into 2x2 and 1x2.
    # Part 1a: A[0:2, 0:2] * B[0:2, 0:4]
    # B[0:2, 0:4] is 2x4. Split B into B[0:2, 0:2] and B[0:2, 2:4].
    # This is 2x2x2 and 2x2x2.
    
    # 1a-1: A[0:2, 0:2] * B[0:2, 0:2]
    col = add_strassen_2x2x2(U, V, W, 0, 0, 0, col, n, m, p)
    
    # 1a-2: A[0:2, 0:2] * B[0:2, 2:4]
    # B block is rows 0:2, cols 2:4.
    # j_off=0, k_off=2.
    col = add_strassen_2x2x2(U, V, W, 0, 0, 2, col, n, m, p)
    
    # Part 1b: A[2:3, 0:2] * B[0:2, 0:4]
    # A is 1x2. B is 2x4.
    # Split B into B[0:2, 0:2] and B[0:2, 2:4].
    # 1b-1: A[2:3, 0:2] * B[0:2, 0:2]
    col = add_1x2x2(U, V, W, 2, 0, 0, col, n, m, p)
    
    # 1b-2: A[2:3, 0:2] * B[0:2, 2:4]
    col = add_1x2x2(U, V, W, 2, 0, 2, col, n, m, p)
    
    # Block 2: A[:, 2:4] * B[2:4, :]
    # i_off=0, j_off=2, k_off=0
    # A is 3x2. B is 2x4.
    
    # 2a-1: A[0:2, 2:4] * B[2:4, 0:2]
    col = add_strassen_2x2x2(U, V, W, 0, 2, 0, col, n, m, p)
    
    # 2a-2: A[0:2, 2:4] * B[2:4, 2:4]
    col = add_strassen_2x2x2(U, V, W, 0, 2, 2, col, n, m, p)
    
    # 2b-1: A[2:3, 2:4] * B[2:4, 0:2]
    col = add_1x2x2(U, V, W, 2, 2, 0, col, n, m, p)
    
    # 2b-2: A[2:3, 2:4] * B[2:4, 2:4]
    col = add_1x2x2(U, V, W, 2, 2, 2, col, n, m, p)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
