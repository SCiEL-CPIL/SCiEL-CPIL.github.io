"""Matrix multiplication 2x2x8 rank-minimal tensor decomposition."""

def construct():
    n, m, p = 2, 2, 8
    # U shape: (n*m, rank) = (4, rank)
    # V shape: (m*p, rank) = (16, rank)
    # W shape: (p*n, rank) = (16, rank)
    #
    # We use Strassen's algorithm for the 2x2x2 block structure.
    # A is 2x2, B is 2x8 (viewed as 2 blocks of 2x4? No, B is 2 rows, 8 cols)
    # Actually, B is 2x8. We can view B as a block matrix with 1 row of blocks,
    # each block being 2x8? No, that doesn't help.
    #
    # Let's reconsider the structure.
    # A: 2x2
    # B: 2x8
    # C = A*B: 2x8
    #
    # The standard Strassen algorithm works for multiplying two 2x2 matrices.
    # Here, B is 2x8. We can decompose B into four 2x2 blocks? No, B is 2 rows, 8 cols.
    # We can decompose B into four 2x2 blocks: B = [B11 B12 B13 B14] where each Bjk is 2x2.
    # Then C = [C11 C12 C13 C14] where each Cjk is 2x2.
    # Each Cjk = A * Bjk.
    #
    # So we have 4 independent 2x2x2 matrix multiplications.
    # Each can be done with rank 7 (Strassen).
    # Total rank = 4 * 7 = 28.
    #
    # Let's verify this.
    # A is shared across all 4 multiplications.
    # B is split into 4 2x2 blocks.
    # C is split into 4 2x2 blocks.
    #
    # The tensor decomposition has:
    # U: (2*2, 28) = (4, 28)
    # V: (2*8, 28) = (16, 28)
    # W: (8*2, 28) = (16, 28)
    #
    # Wait, the task says:
    # U shape: (n*m, rank) = (4, rank)
    # V shape: (m*p, rank) = (16, rank)
    # W shape: (p*n, rank) = (16, rank)
    #
    # The output factor follows the official trace convention W[k*n+i,r].
    # W[k*n+i] corresponds to output C[i,k] (i-th row, k-th column of C).
    #
    # Let's set up the Strassen decomposition for a single 2x2x2 multiplication.
    # A = [[a11, a12], [a21, a22]]
    # B = [[b11, b12], [b21, b22]]
    # C = A*B = [[c11, c12], [c21, c22]]
    #
    # Strassen's 7 products:
    # M1 = (a11 + a22) * (b11 + b22)
    # M2 = (a21 + a22) * b11
    # M3 = (a11 - a21) * (b21 + b22)
    # M4 = (a12 - a22) * (b11 + b12)
    # M5 = (a11 + a12) * b22
    # M6 = (a21 - a11) * (b11 + b12)
    # M7 = (a12 + a22) * (b21 - b11)
    #
    # Then:
    # c11 = M1 + M4 - M5 + M7
    # c12 = M3 + M5
    # c21 = M2 + M4
    # c22 = M1 - M2 + M3 + M6
    #
    # For the 4 blocks of B (each 2x2), we use the same A but different B blocks.
    # Let B_jk be the block at row block j, col block k. But B is 2x8, so it's 1 row of blocks, 4 col blocks.
    # B = [B1, B2, B3, B4] where each Bj is 2x2.
    # C = [C1, C2, C3, C4] where each Cj is 2x2.
    #
    # For each j = 1,2,3,4:
    # Cj = A * Bj
    # Using Strassen:
    # Mj1 = (a11 + a22) * (Bj[0,0] + Bj[1,1])
    # Mj2 = (a21 + a22) * Bj[0,0]
    # Mj3 = (a11 - a21) * (Bj[1,0] + Bj[1,1])
    # Mj4 = (a12 - a22) * (Bj[0,0] + Bj[0,1])
    # Mj5 = (a11 + a12) * Bj[1,1]
    # Mj6 = (a21 - a11) * (Bj[0,0] + Bj[0,1])
    # Mj7 = (a12 + a22) * (Bj[1,0] - Bj[0,0])
    #
    # Cj[0,0] = Mj1 + Mj4 - Mj5 + Mj7
    # Cj[0,1] = Mj3 + Mj5
    # Cj[1,0] = Mj2 + Mj4
    # Cj[1,1] = Mj1 - Mj2 + Mj3 + Mj6
    #
    # Total rank = 4 * 7 = 28.
    
    rank = 28
    U = [[0] * rank for _ in range(4)]  # (n*m, rank)
    V = [[0] * rank for _ in range(16)]  # (m*p, rank)
    W = [[0] * rank for _ in range(16)]  # (p*n, rank)
    
    # U index: i*m + j = i*2 + j, where i=0,1 (row of A), j=0,1 (col of A)
    # So U[0]=a11, U[1]=a12, U[2]=a21, U[3]=a22
    
    # V index: j*p + k, where j=0,1 (row of B), k=0..7 (col of B)
    # For block b (0-indexed), cols are 2*b, 2*b+1
    # V[j*p + k] corresponds to B[j, k]
    
    # W index: k*n + i, where k=0..7 (col of C), i=0,1 (row of C)
    # For block b, cols are 2*b, 2*b+1
    # W[k*n + i] corresponds to C[i, k]
    
    # Strassen coefficients for each block
    # For block b, products M1..M7
    
    # Let's define the linear forms for A
    # M1: (a11 + a22) -> U contributions: [1, 0, 0, 1]
    # M2: (a21 + a22) -> U contributions: [0, 0, 1, 1]
    # M3: (a11 - a21) -> U contributions: [1, 0, -1, 0]
    # M4: (a12 - a22) -> U contributions: [0, 1, 0, -1]
    # M5: (a11 + a12) -> U contributions: [1, 1, 0, 0]
    # M6: (a21 - a11) -> U contributions: [-1, 0, 1, 0]
    # M7: (a12 + a22) -> U contributions: [0, 1, 0, 1]
    
    A_forms = [
        [1, 0, 0, 1],   # M1
        [0, 0, 1, 1],   # M2
        [1, 0, -1, 0],  # M3
        [0, 1, 0, -1],  # M4
        [1, 1, 0, 0],   # M5
        [-1, 0, 1, 0],  # M6
        [0, 1, 0, 1],   # M7
    ]
    
    # For each block b (0..3), and each product s (0..6)
    # Product index within block: s
    # Global column: b * 7 + s
    
    for b in range(4):
        for s in range(7):
            col = b * 7 + s
            
            # U entries
            for i in range(4):
                U[i][col] = A_forms[s][i]
            
            # V entries (linear forms in B block b)
            # Block b uses B cols 2*b, 2*b+1
            # V index: j*p + k = j*8 + k
            # B[j, 2*b] and B[j, 2*b+1]
            
            # M1: (B[0, 2*b] + B[1, 2*b+1]) -> V[0*8 + 2*b] += 1, V[1*8 + 2*b+1] += 1
            # M2: B[0, 2*b] -> V[0*8 + 2*b] += 1
            # M3: (B[1, 2*b] + B[1, 2*b+1]) -> V[1*8 + 2*b] += 1, V[1*8 + 2*b+1] += 1
            # M4: (B[0, 2*b] + B[0, 2*b+1]) -> V[0*8 + 2*b] += 1, V[0*8 + 2*b+1] += 1
            # M5: B[1, 2*b+1] -> V[1*8 + 2*b+1] += 1
            # M6: (B[0, 2*b] + B[0, 2*b+1]) -> V[0*8 + 2*b] += 1, V[0*8 + 2*b+1] += 1
            # M7: (B[1, 2*b] - B[0, 2*b]) -> V[1*8 + 2*b] += 1, V[0*8 + 2*b] += -1
            
            if s == 0:  # M1
                V[0 * 8 + 2 * b][col] += 1
                V[1 * 8 + 2 * b + 1][col] += 1
            elif s == 1:  # M2
                V[0 * 8 + 2 * b][col] += 1
            elif s == 2:  # M3
                V[1 * 8 + 2 * b][col] += 1
                V[1 * 8 + 2 * b + 1][col] += 1
            elif s == 3:  # M4
                V[0 * 8 + 2 * b][col] += 1
                V[0 * 8 + 2 * b + 1][col] += 1
            elif s == 4:  # M5
                V[1 * 8 + 2 * b + 1][col] += 1
            elif s == 5:  # M6
                V[0 * 8 + 2 * b][col] += 1
                V[0 * 8 + 2 * b + 1][col] += 1
            elif s == 6:  # M7
                V[1 * 8 + 2 * b][col] += 1
                V[0 * 8 + 2 * b][col] += -1
            
            # W entries (output C block b)
            # W index: k*n + i = k*2 + i
            # Block b: cols 2*b, 2*b+1
            # C[0, 2*b] = W[2*b*2 + 0] = W[4*b]
            # C[0, 2*b+1] = W[4*b+1]
            # C[1, 2*b] = W[4*b+2]
            # C[1, 2*b+1] = W[4*b+3]
            
            # c11 = M1 + M4 - M5 + M7
            # c12 = M3 + M5
            # c21 = M2 + M4
            # c22 = M1 - M2 + M3 + M6
            
            if s == 0:  # M1
                W[4 * b][col] += 1       # c11
                W[4 * b + 3][col] += 1   # c22
            elif s == 1:  # M2
                W[4 * b + 2][col] += 1   # c21
                W[4 * b + 3][col] += -1  # c22
            elif s == 2:  # M3
                W[4 * b + 1][col] += 1   # c12
                W[4 * b + 3][col] += 1   # c22
            elif s == 3:  # M4
                W[4 * b][col] += 1       # c11
                W[4 * b + 2][col] += 1   # c21
            elif s == 4:  # M5
                W[4 * b][col] += -1      # c11
                W[4 * b + 1][col] += 1   # c12
            elif s == 5:  # M6
                W[4 * b + 3][col] += 1   # c22
            elif s == 6:  # M7
                W[4 * b][col] += 1       # c11
    
    return {"U": U, "V": V, "W": W}
