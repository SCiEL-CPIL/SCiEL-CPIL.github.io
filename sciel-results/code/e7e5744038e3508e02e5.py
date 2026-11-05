"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Strassen-based decomposition for 2x4 * 4x8 matrix multiplication.
    
    Split A (2x4) into two 2x2 blocks: A = [A1 A2]
    Split B (4x8) into four 2x2 blocks: B = [B11 B12; B21 B22]
    
    Result C (2x8) has blocks: C = [C1 C2 C3 C4] where:
    C1 = A1*B11 + A2*B21
    C2 = A1*B12 + A2*B22
    C3 = A1*B13 + A2*B23  (if we had 8 cols as 4 blocks of 2)
    C4 = A1*B14 + A2*B24
    
    Actually, let's be more careful. A is 2x4, B is 4x8.
    Split A into left 2x2 (A1) and right 2x2 (A2): A = [A1 | A2]
    Split B into top 2x8 (B1) and bottom 2x8 (B2): B = [B1; B2]
    
    Then C = A1*B1 + A2*B2, where C is 2x8.
    
    Now A1*B1 is 2x2 * 2x8. We can split B1 into four 2x2 blocks: B1 = [B11 B12 B13 B14]
    Similarly A2*B2: B2 = [B21 B22 B23 B24]
    
    So C = [A1*B11 + A2*B21, A1*B12 + A2*B22, A1*B13 + A2*B23, A1*B14 + A2*B24]
    
    Each term like A1*B11 is 2x2 * 2x2. Using Strassen, each needs 7 multiplications.
    There are 8 such 2x2*2x2 multiplications (2 A-blocks * 4 B-column-blocks).
    Total rank = 8 * 7 = 56.
    """
    n, m, p = DIMENSIONS  # n=2, m=4, p=8
    
    # We'll build U, V, W for the tensor decomposition.
    # U has shape (n*m, rank) = (8, 56)
    # V has shape (m*p, rank) = (32, 56)  
    # W has shape (p*n, rank) = (16, 56)
    
    # Strassen's algorithm for 2x2 * 2x2:
    # Given X = [[a, b], [c, d]], Y = [[e, f], [g, h]]
    # M1 = (a+d)*(e+h)
    # M2 = (c+d)*e
    # M3 = a*(f-h)
    # M4 = d*(g-e)
    # M5 = (a+b)*h
    # M6 = (c-a)*(e+f)
    # M7 = (b-d)*(g+h)
    # Result: [[M1+M4-M5+M7, M3+M5], [M2+M4, M1-M2+M3+M6]]
    
    # For each of the 8 block multiplications, we need 7 rank-1 terms.
    # Block (i, j) where i in {0,1} (A row block), j in {0,1,2,3} (B col block)
    # A block i: rows i*2 to i*2+1 of A, cols 0..1 (if i=0) or 2..3 (if i=1)
    # Wait, A is 2x4. Let me index A as A[row, col] where row in {0,1}, col in {0,1,2,3}
    # A1 = A[:, 0:2] (2x2), A2 = A[:, 2:4] (2x2)
    
    # B is 4x8. B[row, col] where row in {0,1,2,3}, col in {0..7}
    # B1 = B[0:2, :] (2x8), B2 = B[2:4, :] (2x8)
    # B1j = B[0:2, 2*j:2*j+2] for j in {0,1,2,3}
    # B2j = B[2:4, 2*j:2*j+2] for j in {0,1,2,3}
    
    # C[row, col] = A1[:, 0:2] @ B1[0:2, col:col+2] + A2[:, 0:2] @ B2[0:2, col:col+2]
    # More precisely, for output col c:
    # C[0, c] = A[0,0]*B[0,c] + A[0,1]*B[1,c] + A[0,2]*B[2,c] + A[0,3]*B[3,c]
    # C[1, c] = A[1,0]*B[0,c] + A[1,1]*B[1,c] + A[1,2]*B[2,c] + A[1,3]*B[3,c]
    
    # Using Strassen on each 2x2*2x2 block:
    # For block (i, j): X = A block i (2x2), Y = B block (row i, col j) (2x2)
    # X = [[a, b], [c, d]] where a=A[i*0+0, i*2+0], b=A[i*0+0, i*2+1], c=A[i*0+1, i*2+0], d=A[i*0+1, i*2+1]
    # Wait, A is 2x4. A1 = A[:, 0:2] = [[A[0,0], A[0,1]], [A[1,0], A[1,1]]]
    # A2 = A[:, 2:4] = [[A[0,2], A[0,3]], [A[1,2], A[1,3]]]
    
    # For B, block (i, j) where i in {0,1} (row block), j in {0,1,2,3} (col block):
    # B[i, j] = B[2*i:2*i+2, 2*j:2*j+2] = [[B[2i, 2j], B[2i, 2j+1]], [B[2i+1, 2j], B[2i+1, 2j+1]]]
    
    # The contribution to C from block (i, j) goes to C[:, 2*j:2*j+2]
    # Specifically, the 2x2 result Z = X @ Y adds to C[0:2, 2*j:2*j+2]
    
    # Strassen for X = [[a,b],[c,d]], Y = [[e,f],[g,h]]:
    # M1 = (a+d)*(e+h) -> contributes to Z[0,0] and Z[1,1]
    # M2 = (c+d)*e -> contributes to Z[1,0] and Z[1,1]  
    # M3 = a*(f-h) -> contributes to Z[0,1] and Z[1,1]
    # M4 = d*(g-e) -> contributes to Z[0,0] and Z[1,1]
    # M5 = (a+b)*h -> contributes to Z[0,0] and Z[0,1]
    # M6 = (c-a)*(e+f) -> contributes to Z[1,0]
    # M7 = (b-d)*(g+h) -> contributes to Z[0,0]
    
    # Z[0,0] = M1 + M4 - M5 + M7
    # Z[0,1] = M3 + M5
    # Z[1,0] = M2 + M4
    # Z[1,1] = M1 - M2 + M3 + M6
    
    # Now, for the tensor decomposition:
    # U has entries U[A_index, rank_index]
    # V has entries V[B_index, rank_index]
    # W has entries W[C_index, rank_index]
    # where C_index = k*n + i (trace convention: W[k*n+i, r])
    
    # For each Strassen multiplication M_t in block (i, j):
    # M_t = (linear combo of A entries) * (linear combo of B entries)
    # This contributes to specific C entries.
    
    rank = 56
    U = [[0] * rank for _ in range(n * m)]  # 8 x 56
    V = [[0] * rank for _ in range(m * p)]  # 32 x 56
    W = [[0] * rank for _ in range(p * n)]  # 16 x 56
    
    col = 0
    
    # Strassen multipliers for 2x2 * 2x2:
    # For X = [[a,b],[c,d]], Y = [[e,f],[g,h]]:
    # M1: (a+d) * (e+h)
    # M2: (c+d) * e
    # M3: a * (f-h)
    # M4: d * (g-e)
    # M5: (a+b) * h
    # M6: (c-a) * (e+f)
    # M7: (b-d) * (g+h)
    
    # A block i (i in {0,1}):
    # a = A[0, 2*i], b = A[0, 2*i+1], c = A[1, 2*i], d = A[1, 2*i+1]
    # In U indexing: A[row, col] -> U[row * m + col, :] = U[row * 4 + col, :]
    
    # B block (i, j) (i in {0,1}, j in {0,1,2,3}):
    # e = B[2*i, 2*j], f = B[2*i, 2*j+1], g = B[2*i+1, 2*j], h = B[2*i+1, 2*j+1]
    # In V indexing: B[row, col] -> V[row * p + col, :] = V[row * 8 + col, :]
    
    # C output for block (i, j) goes to C[:, 2*j:2*j+2]:
    # Z[0,0] -> C[0, 2*j], Z[0,1] -> C[0, 2*j+1]
    # Z[1,0] -> C[1, 2*j], Z[1,1] -> C[1, 2*j+1]
    # In W indexing: C[row, col] -> W[col * n + row, :] = W[col * 2 + row, :]
    
    # Strassen coefficient table:
    # multiplier: (A_coeffs, B_coeffs, C_contributions)
    # A_coeffs: dict from (row, col) to coeff for A[row, col]
    # B_coeffs: dict from (row, col) to coeff for B[row, col]
    # C_contributions: dict from (row, col) to coeff for C[row, col]
    
    strassen_multipliers = [
        # M1: (a+d) * (e+h)
        ({(0, 0): 1, (1, 0): 1},  # a + c... wait, a=X[0,0], c=X[1,0]
         {(0, 0): 1, (1, 0): 1},  # e + g... wait, e=Y[0,0], g=Y[1,0]
         {(0, 0): 1, (1, 1): 1}), # Z[0,0] += 1, Z[1,1] += 1
        # M2: (c+d) * e
        ({(1, 0): 1, (1, 1): 1},  # c + d
         {(0, 0): 1},              # e
         {(1, 0): 1, (1, 1): -1}), # Z[1,0] += 1, Z[1,1] -= 1
        # M3: a * (f-h)
        ({(0, 0): 1},              # a
         {(0, 1): 1, (1, 1): -1}, # f - h
         {(0, 1): 1, (1, 1): 1}), # Z[0,1] += 1, Z[1,1] += 1
        # M4: d * (g-e)
        ({(1, 1): 1},              # d
         {(1, 0): 1, (0, 0): -1}, # g - e
         {(0, 0): 1, (1, 1): 1}), # Z[0,0] += 1, Z[1,1] += 1
        # M5: (a+b) * h
        ({(0, 0): 1, (0, 1): 1},  # a + b
         {(1, 1): 1},              # h
         {(0, 0): -1, (0, 1): 1}), # Z[0,0] -= 1, Z[0,1] += 1
        # M6: (c-a) * (e+f)
        ({(1, 0): 1, (0, 0): -1}, # c - a
         {(0, 0): 1, (0, 1): 1}, # e + f
         {(1, 1): 1}),            # Z[1,1] += 1
        # M7: (b-d) * (g+h)
        ({(0, 1): 1, (1, 1): -1}, # b - d
         {(1, 0): 1, (1, 1): 1}, # g + h
         {(0, 0): 1}),            # Z[0,0] += 1
    ]
    
    for i in range(2):  # A row block (0 or 1)
        for j in range(4):  # B col block (0,1,2,3)
            for a_coeffs, b_coeffs, c_coeffs in strassen_multipliers:
                # A part: for each (ar, ac) in a_coeffs, A[ar, 2*i + ac]
                for (ar, ac), coeff in a_coeffs.items():
                    U[ar * m + (2 * i + ac)][col] += coeff
                
                # B part: for each (br, bc) in b_coeffs, B[2*i + br, 2*j + bc]
                for (br, bc), coeff in b_coeffs.items():
                    V[(2 * i + br) * p + (2 * j + bc)][col] += coeff
                
                # C part: for each (cr, cc) in c_coeffs, C[cr, 2*j + cc]
                # W index: C[row, col] -> W[col * n + row]
                for (cr, cc), coeff in c_coeffs.items():
                    W[(2 * j + cc) * n + cr][col] += coeff
                
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
