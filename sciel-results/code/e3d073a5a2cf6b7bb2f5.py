"""Matrix multiplication 5x5x5 rank-minimal decomposition."""
DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Strassen 2x2 decomposition (rank 7)
    # For 2x2 matrices: A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # C = AB
    # Strassen uses 7 multiplications:
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-c)
    # m5 = (a+b)h
    # m6 = (c-a)(e+f)
    # m7 = (b-d)(g+h)
    # C11 = m1 + m4 - m5 + m7
    # C12 = m3 + m5
    # C21 = m2 + m4
    # C22 = m1 - m2 + m3 + m6
    
    # Build U, V, W for 2x2x2 Strassen
    # U has shape (n*m, rank) = (4, 7)
    # V has shape (m*p, rank) = (4, 7)  
    # W has shape (p*n, rank) = (4, 7)
    # Index convention: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    
    def strassen_2x2():
        rank = 7
        U = [[0]*rank for _ in range(4)]
        V = [[0]*rank for _ in range(4)]
        W = [[0]*rank for _ in range(4)]
        
        # m1 = (a+d)(e+h)
        # a = A[0,0], d = A[1,1] -> U[0*2+0]=U[0], U[1*2+1]=U[3]
        # e = B[0,0], h = B[1,1] -> V[0*2+0]=V[0], V[1*2+1]=V[3]
        # Output: contributes to C[0,0] and C[1,1]
        # C[0,0] -> W[0*2+0]=W[0], C[1,1] -> W[1*2+1]=W[3]
        U[0][0] = 1  # a
        U[3][0] = 1  # d
        V[0][0] = 1  # e
        V[3][0] = 1  # h
        W[0][0] = 1  # C[0,0]
        W[3][0] = 1  # C[1,1]
        
        # m2 = (c+d)e
        # c = A[1,0], d = A[1,1] -> U[1*2+0]=U[2], U[3]
        # e = B[0,0] -> V[0]
        # Output: contributes to C[1,0]
        # C[1,0] -> W[1*2+0]=W[2]
        U[2][1] = 1  # c
        U[3][1] = 1  # d
        V[0][1] = 1  # e
        W[2][1] = 1  # C[1,0]
        
        # m3 = a(f-h)
        # a = A[0,0] -> U[0]
        # f = B[0,1], h = B[1,1] -> V[0*2+1]=V[1], V[3]
        # Output: contributes to C[0,1] and C[1,1]
        # C[0,1] -> W[0*2+1]=W[1], C[1,1] -> W[3]
        U[0][2] = 1  # a
        V[1][2] = 1  # f
        V[3][2] = -1  # -h
        W[1][2] = 1  # C[0,1]
        W[3][2] = 1  # C[1,1]
        
        # m4 = d(g-c)
        # d = A[1,1] -> U[3]
        # g = B[1,0], c = A[1,0]... wait, this is (g-c) where g is from B, c is from A?
        # Actually m4 = d*(g-c), so:
        # d = A[1,1] -> U[3]
        # g = B[1,0] -> V[1*2+0]=V[2]
        # c = A[1,0] -> U[2]
        # Output: contributes to C[1,0] and C[0,0]
        # C[1,0] -> W[2], C[0,0] -> W[0]
        U[3][3] = 1  # d
        V[2][3] = 1  # g
        U[2][3] = -1  # -c
        W[2][3] = 1  # C[1,0]
        W[0][3] = 1  # C[0,0]
        
        # m5 = (a+b)h
        # a = A[0,0], b = A[0,1] -> U[0], U[0*2+1]=U[1]
        # h = B[1,1] -> V[3]
        # Output: contributes to C[0,0] and C[0,1]
        # C[0,0] -> W[0], C[0,1] -> W[1]
        U[0][4] = 1  # a
        U[1][4] = 1  # b
        V[3][4] = 1  # h
        W[0][4] = -1  # -C[0,0] (because C11 = m1 + m4 - m5 + m7)
        W[1][4] = 1  # C[0,1]
        
        # m6 = (c-a)(e+f)
        # c = A[1,0], a = A[0,0] -> U[2], U[0]
        # e = B[0,0], f = B[0,1] -> V[0], V[1]
        # Output: contributes to C[1,1]
        # C[1,1] -> W[3]
        U[2][5] = 1  # c
        U[0][5] = -1  # -a
        V[0][5] = 1  # e
        V[1][5] = 1  # f
        W[3][5] = 1  # C[1,1]
        
        # m7 = (b-d)(g+h)
        # b = A[0,1], d = A[1,1] -> U[1], U[3]
        # g = B[1,0], h = B[1,1] -> V[2], V[3]
        # Output: contributes to C[0,0]
        # C[0,0] -> W[0]
        U[1][6] = 1  # b
        U[3][6] = -1  # -d
        V[2][6] = 1  # g
        V[3][6] = 1  # h
        W[0][6] = 1  # C[0,0]
        
        return U, V, W
    
    # Laderman 3x3 decomposition (rank 7)
    # This is more complex. Let me use a known rank-7 decomposition for 3x3.
    # Actually, the minimum rank for 3x3 matrix multiplication is known to be 19,
    # not 7. Laderman's algorithm is for 3x3 with rank 19? No wait.
    
    # Let me reconsider. The minimum rank for 3x3x3 is 19 (Bini et al.).
    # For 2x2x2, minimum rank is 7 (Strassen).
    
    # So the tensor product approach: 5 = 2 + 3
    # Block decomposition:
    # A = [[A11, A12], [A21, A22]] where A11 is 2x2, A12 is 2x3, A21 is 3x2, A22 is 3x3
    # B = [[B11, B12], [B21, B22]] similarly
    # C = AB = [[A11*B11 + A12*B21, A11*B12 + A12*B22], [A21*B11 + A22*B21, A21*B12 + A22*B22]]
    
    # Rank of full product:
    # - (2,2) block: rank 7 (Strassen)
    # - (3,3) block: rank 19 (best known for 3x3)
    # - (2,3) block: A12*B21, this is 2x3 * 3x2 = 2x2, rank 4 (schoolbook for 2x2*3... wait)
    
    # Actually for rectangular: multiplying m×n by n×p has rank m*n*p at most.
    # 2x3 times 3x2: rank at most 2*3*2=12, but can be done with 6? No.
    # The minimum rank for 2x3 * 3x2 is 12 (schoolbook) since no known improvement.
    
    # This is getting complicated. Let me try a different approach.
    
    # Alternative: Use the fact that 5x5 can be decomposed as 2x2 + 3x3 blocks.
    # Total rank = rank(2x2 mult) + rank(3x3 mult) + rank(cross terms)
    
    # Cross terms:
    # A12*B21: 2x3 * 3x2 -> 2x2 output, rank <= 2*3*2 = 12
    # A11*B12: 2x2 * 2x3 -> 2x3 output, rank <= 2*2*3 = 12
    # A21*B11: 3x2 * 2x2 -> 3x2 output, rank <= 3*2*2 = 12
    # A22*B21: 3x3 * 3x2 -> 3x2 output, rank <= 3*3*2 = 18
    # A21*B12: 3x2 * 2x3 -> 3x3 output, rank <= 3*2*3 = 18
    # A12*B22: 2x3 * 3x3 -> 2x3 output, rank <= 2*3*3 = 18
    
    # Total: 7 + 19 + 12 + 12 + 12 + 18 + 18 + 18 = 116
    
    # That's worse than 110 from the 1+4 approach.
    
    # Let me go back to the 1+4 approach with Strassen for 4x4 (which is 2x2 Kronecker 2x2).
    
    # For 4x4x4 using Kronecker product of two 2x2x2 Strassen:
    # Rank = 7 * 7 = 49
    
    # 5 = 1 + 4
    # Blocks:
    # (1,1): 1x1 * 1x1 = 1x1, rank 1
    # (1,4): 1x4 * 4x1 -> 1x1, rank 4
    # (4,1): 4x1 * 1x1 -> 4x1, rank 4
    # (4,4): 4x4 * 4x4 -> 4x4, rank 49
    # (1,4)*other, etc.
    
    # Full C = A*B where A,B are 5x5:
    # A = [[a11, a12_1x4], [a21_4x1, a22_4x4]]
    # B = [[b11, b12_1x4], [b21_4x1, b22_4x4]]
    
    # C = [[a11*b11 + a12*b21, a11*b12 + a12*b22], [a21*b11 + a22*b21, a21*b12 + a22*b22]]
    
    # Ranks:
    # (1,1) block of C: a11*b11 (rank 1) + a12*b21 (1x4 * 4x1 = 1x1, rank 4)
    # (1,2) block of C: a11*b12 (1x1 * 1x4 = 1x4, rank 4) + a12*b22 (1x4 * 4x4 = 1x4, rank 16)
    # (2,1) block of C: a21*b11 (4x1 * 1x1 = 4x1, rank 4) + a22*b21 (4x4 * 4x1 = 4x1, rank 16)
    # (2,2) block of C: a21*b12 (4x1 * 1x4 = 4x4, rank 16) + a22*b22 (4x4 * 4x4 = 4x4, rank 49)
    
    # Total rank = 1 + 4 + 4 + 16 + 4 + 16 + 16 + 49 = 110
    
    # Now I need to construct this carefully.
    
    # First, get the Strassen 2x2 decomposition.
    # Then Kronecker product for 4x4.
    
    def kron_decomp(U1, V1, W1, U2, V2, W2):
        """Kronecker product of two decompositions."""
        r1 = len(U1[0])
        r2 = len(U2[0])
        rank = r1 * r2
        
        # Dimensions
        n1, m1 = len(U1) // 2, 2  # For 2x2
        # Actually, U has shape (n*m, rank), so n*m = len(U)
        nm1 = len(U1)
        mp1 = len(V1)
        pn1 = len(W1)
        nm2 = len(U2)
        mp2 = len(V2)
        pn2 = len(W2)
        
        # For Kronecker: if we're composing two matrix multiplications,
        # the new dimension is n1*n2, m1*m2, p1*p2
        # U_new has shape (n1*n2 * m1*m2, r1*r2)
        # Index mapping: for 2x2, i is 0 or 1. For 4x4, i is 0..3 = 2*i1 + i2
        
        n_total = int(nm1 ** 0.5)  # This doesn't work in general
        # Better: pass dimensions explicitly
        
        # Let me redo this with explicit dimensions
        return None  # Placeholder
    
    # Let me build the 4x4 decomposition from two 2x2 Strassen decompositions.
    # For 2x2: U shape (4, 7), indices: U[i*2+j] for i,j in {0,1}
    # For 4x4: U shape (16, 49), indices: U[i*4+j] for i,j in {0,1,2,3}
    # Where i = 2*i1 + i2, j = 2*j1 + j2
    
    def build_4x4_from_2x2(U2, V2, W2):
        """Build 4x4x4 decomposition from 2x2x2 via Kronecker product."""
        r = len(U2[0])  # rank of 2x2
        rank4 = r * r  # 49
        
        # 4x4: U has 16 rows, V has 16 rows, W has 16 rows
        U4 = [[0]*rank4 for _ in range(16)]
        V4 = [[0]*rank4 for _ in range(16)]
        W4 = [[0]*rank4 for _ in range(16)]
        
        for c1 in range(r):
            for c2 in range(r):
                c = c1 * r + c2
                for i1 in range(2):
                    for i2 in range(2):
                        i = 2 * i1 + i2
                        for j1 in range(2):
                            for j2 in range(2):
                                j = 2 * j1 + j2
                                # U4[i*4+j][c] += U2[i1*2+j1][c1] * U2[i2*2+j2][c2]
                                U4[i*4+j][c] += U2[i1*2+j1][c1] * U2[i2*2+j2][c2]
        
        for c1 in range(r):
            for c2 in range(r):
                c = c1 * r + c2
                for j1 in range(2):
                    for j2 in range(2):
                        j = 2 * j1 + j2
                        for k1 in range(2):
                            for k2 in range(2):
                                k = 2 * k1 + k2
                                V4[j*4+k][c] += V2[j1*2+k1][c1] * V2[j2*2+k2][c2]
        
        for c1 in range(r):
            for c2 in range(r):
                c = c1 * r + c2
                for k1 in range(2):
                    for k2 in range(2):
                        k = 2 * k1 + k2
                        for i1 in range(2):
                            for i2 in range(2):
                                i = 2 * i1 + i2
                                W4[k*4+i][c] += W2[k1*2+i1][c1] * W2[k2*2+i2][c2]
        
        return U4, V4, W4
    
    U2, V2, W2 = strassen_2x2()
    U4, V4, W4 = build_4x4_from_2x2(U2, V2, W2)
    
    # Now build the 5x5 decomposition using 1+4 block structure.
    # 5 = 1 + 4
    # A indices: i in 0..4. i=0 is the 1-block, i=1..4 is the 4-block.
    # Similarly for j, k.
    
    # Block structure:
    # i=0: first block (size 1)
    # i=1,2,3,4: second block (size 4)
    # Within second block: i' = i-1, so i' in 0..3
    
    # U shape: (5*5, total_rank) = (25, total_rank)
    # V shape: (5*5, total_rank) = (25, total_rank)
    # W shape: (5*5, total_rank) = (25, total_rank)
    
    # Total rank = 1 + 4 + 4 + 16 + 4 + 16 + 16 + 49 = 110
    
    total_rank = 110
    U5 = [[0]*total_rank for _ in range(25)]
    V5 = [[0]*total_rank for _ in range(25)]
    W5 = [[0]*total_rank for _ in range(25)]
    
    col_offset = 0
    
    # Block (1,1): A[0,0] * B[0,0] -> C[0,0]
    # Rank 1
    # U[0*5+0][col] = 1, V[0*5+0][col] = 1, W[0*5+0][col] = 1
    U5[0][col_offset] = 1
    V5[0][col_offset] = 1
    W5[0][col_offset] = 1
    col_offset += 1
    
    # Block (1,4) for C[0,*]: A[0, j] for j in 1..4 times B[j, 0]
    # A[0, j] where j in 1..4 -> U index: 0*5 + j = j
    # B[j, 0] where j in 1..4 -> V index: j*5 + 0 = 5*j
    # C[0, 0] -> W index: 0*5 + 0 = 0
    # This is a 1x4 * 4x1 = 1x1 multiplication, rank 4
    for j in range(1, 5):
        U5[0*5 + j][col_offset] = 1
        V5[j*5 + 0][col_offset] = 1
        W5[0*5 + 0][col_offset] = 1
        col_offset += 1
    
    # Block (1,4) for C[0, *]: A[0,0] * B[0, j] for j in 1..4
    # A[0,0] -> U[0]
    # B[0, j] for j in 1..4 -> V[0*5 + j] = V[j]
    # C[0, j] for j in 1..4 -> W[0*5 + j] = W[j]
    # Rank 4
    for j in range(1, 5):
        U5[0][col_offset] = 1
        V5[0*5 + j][col_offset] = 1
        W5[0*5 + j][col_offset] = 1
        col_offset += 1
    
    # Block (1,4) for C[0, *]: A[0, j] for j in 1..4 * B[j, k] for k in 1..4
    # This is 1x4 * 4x4 -> 1x4
    # A[0, j] where j in 1..4 -> U[j]
    # B[j, k] where j in 1..4, k in 1..4 -> V[j*5 + k]
    # C[0, k] where k in 1..4 -> W[k]
    # Rank 4*4 = 16
    for j in range(1, 5):
        for k in range(1, 5):
            U5[0*5 + j][col_offset] = 1
            V5[j*5 + k][col_offset] = 1
            W5[0*5 + k][col_offset] = 1
            col_offset += 1
    
    # Block (4,1) for C[*, 0]: A[i, 0] for i in 1..4 * B[0,0]
    # A[i, 0] where i in 1..4 -> U[i*5 + 0] = U[5*i]
    # B[0,0] -> V[0]
    # C[i, 0] where i in 1..4 -> W[i*5 + 0] = W[5*i]
    # Rank 4
    for i in range(1, 5):
        U5[i*5 + 0][col_offset] = 1
        V5[0][col_offset] = 1
        W5[i*5 + 0][col_offset] = 1
        col_offset += 1
    
    # Block (4,4) for C[*, 0]: A[i, j] for i in 1..4, j in 1..4 * B[j, 0]
    # This is 4x4 * 4x1 -> 4x1
    # A[i, j] where i in 1..4, j in 1..4 -> U[i*5 + j]
    # B[j, 0] where j in 1..4 -> V[j*5 + 0] = V[5*j]
    # C[i, 0] where i in 1..4 -> W[i*5 + 0] = W[5*i]
    # Rank 4*4 = 16
    for i in range(1, 5):
        for j in range(1, 5):
            U5[i*5 + j][col_offset] = 1
            V5[j*5 + 0][col_offset] = 1
            W5[i*5 + 0][col_offset] = 1
            col_offset += 1
    
    # Block (4,1) for C[*, *]: A[i, 0] for i in 1..4 * B[0, k] for k in 1..4
    # This is 4x1 * 1x4 -> 4x4
    # A[i, 0] where i in 1..4 -> U[5*i]
    # B[0, k] where k in 1..4 -> V[k]
    # C[i, k] where i in 1..4, k in 1..4 -> W[i*5 + k]
    # Rank 4*4 = 16
    for i in range(1, 5):
        for k in range(1, 5):
            U5[i*5 + 0][col_offset] = 1
            V5[0*5 + k][col_offset] = 1
            W5[i*5 + k][col_offset] = 1
            col_offset += 1
    
    # Block (4,4) for C[*, *]: A[i, j] for i in 1..4, j in 1..4 * B[j, k] for j in 1..4, k in 1..4
    # This is 4x4 * 4x4 -> 4x4, using Strassen-Kronecker rank 49
    # Map: i in 1..4 -> i' = i-1 in 0..3
    #       j in 1..4 -> j' = j-1 in 0..3
    #       k in 1..4 -> k' = k-1 in 0..3
    # U4[i'*4 + j'][c] -> U5[i*5 + j][col_offset + c]
    # V4[j'*4 + k'][c] -> V5[j*5 + k][col_offset + c]
    # W4[k'*4 + i'][c] -> W5[k*5 + i][col_offset + c]
    
    for c in range(49):
        for i in range(4):
            for j in range(4):
                val = U4[i*4 + j][c]
                if val != 0:
                    U5[(i+1)*5 + (j+1)][col_offset + c] += val
        for j in range(4):
            for k in range(4):
                val = V4[j*4 + k][c]
                if val != 0:
                    V5[(j+1)*5 + (k+1)][col_offset + c] += val
        for k in range(4):
            for i in range(4):
                val = W4[k*4 + i][c]
                if val != 0:
                    W5[(k+1)*5 + (i+1)][col_offset + c] += val
    
    col_offset += 49
    
    # Verify col_offset == total_rank
    assert col_offset == total_rank, f"col_offset {col_offset} != total_rank {total_rank}"
    
    return {"U": U5, "V": V5, "W": W5}
# EVOLVE-BLOCK-END
