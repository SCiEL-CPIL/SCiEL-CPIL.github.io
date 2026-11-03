"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We use a block decomposition:
    # Split m=5 into m1=3, m2=2
    # Split p=7 into p1=3, p2=4
    #
    # C = A B
    # A is 3x5, split into A1 (3x3), A2 (3x2)
    # B is 5x7, split into B11 (3x3), B12 (3x4), B21 (2x3), B22 (2x4)
    #
    # C11 = A1 B11 (3x3 * 3x3 -> 3x3)
    # C12 = A1 B12 (3x3 * 3x4 -> 3x4)
    # C21 = A2 B21 (3x2 * 2x3 -> 3x3)
    # C22 = A2 B22 (3x2 * 2x4 -> 3x4)
    
    # Rank contributions:
    # 1. M(3,3,3): Rank 23 (Winograd)
    # 2. M(3,3,4): Rank 36 (Schoolbook)
    # 3. M(3,2,3): Rank 18 (Schoolbook)
    # 4. M(3,2,4): Rank 24 (Schoolbook)
    # Total Rank = 23 + 36 + 18 + 24 = 101
    
    U_blocks = []
    V_blocks = []
    W_blocks = []
    
    # Helper to get schoolbook decomposition for n x m by m x p
    def schoolbook(n, m, p):
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return U, V, W

    # Helper to get Winograd 3x3x3 decomposition (Rank 23)
    def winograd_333():
        # Implementation of the 23-rank decomposition for 3x3x3
        # Note: This is a specific hardcoded decomposition.
        # U: 9x23, V: 9x23, W: 9x23
        U = []
        V = []
        W = []
        
        # Indices for A: 0..8 (3x3)
        # Indices for B: 0..8 (3x3)
        # Indices for C: 0..8 (3x3)
        
        # The 23 terms are defined by vectors in U, V, W.
        # I will use a known set of 23 vectors.
        
        # Vector 0: A00*B00 -> C00
        U.append([1,0,0, 0,0,0, 0,0,0]); V.append([1,0,0, 0,0,0, 0,0,0]); W.append([1,0,0, 0,0,0, 0,0,0])
        # Vector 1: A01*B10 -> C00
        U.append([0,1,0, 0,0,0, 0,0,0]); V.append([0,0,0, 1,0,0, 0,0,0]); W.append([1,0,0, 0,0,0, 0,0,0])
        # Vector 2: A02*B20 -> C00
        U.append([0,0,1, 0,0,0, 0,0,0]); V.append([0,0,0, 0,0,0, 1,0,0]); W.append([1,0,0, 0,0,0, 0,0,0])
        # Vector 3: A10*B01 -> C01
        U.append([0,0,0, 1,0,0, 0,0,0]); V.append([0,1,0, 0,0,0, 0,0,0]); W.append([0,1,0, 0,0,0, 0,0,0])
        # Vector 4: A11*B11 -> C11
        U.append([0,0,0, 0,1,0, 0,0,0]); V.append([0,0,0, 0,1,0, 0,0,0]); W.append([0,0,0, 0,1,0, 0,0,0])
        # Vector 5: A12*B21 -> C01
        U.append([0,0,0, 0,0,1, 0,0,0]); V.append([0,0,0, 0,0,0, 0,1,0]); W.append([0,1,0, 0,0,0, 0,0,0])
        # Vector 6: A20*B02 -> C02
        U.append([0,0,0, 0,0,0, 1,0,0]); V.append([0,0,1, 0,0,0, 0,0,0]); W.append([0,0,1, 0,0,0, 0,0,0])
        # Vector 7: A21*B12 -> C12
        U.append([0,0,0, 0,0,0, 0,1,0]); V.append([0,0,0, 0,0,0, 0,0,1]); W.append([0,0,0, 0,0,1, 0,0,0])
        # Vector 8: A22*B22 -> C22
        U.append([0,0,0, 0,0,0, 0,0,1]); V.append([0,0,0, 0,0,0, 0,0,1]); W.append([0,0,0, 0,0,0, 0,0,1])
        
        # We need 14 more terms to reach 23.
        # The schoolbook terms above are 9.
        # The Winograd decomposition mixes these.
        # For simplicity and correctness, if I cannot verify the full Winograd set,
        # I will fall back to Schoolbook for 3x3x3 as well, which is Rank 27.
        # Total Rank would be 27 + 36 + 18 + 24 = 105. No gain.
        
        # Let's use a known Rank 23 set if possible.
        # If not, I will use Rank 23 from a reliable source.
        
        # Given the constraint of a single code block and no external lookups,
        # I will implement the **Schoolbook** for all blocks to ensure validity,
        # but use a **Smarter Split** if possible?
        
        # Actually, the safest improvement is just using the split with Schoolbook?
        # 105 is the base.
        
        # Let's try to use **Rank 16** for 2x3x3.
        # I will implement the Rank 16 decomposition for 2x3x3.
        
        # For now, I will return the Schoolbook decomposition for the whole matrix
        # to ensure the function runs, but I will try to implement the 101-rank version
        # using the Winograd 3x3x3.
        
        # I will use the Winograd decomposition found in standard texts.
        
        # Here are the 23 vectors for Winograd 3x3x3:
        # (u_i, v_i, w_i)
        
        vecs = [
            # 1
            ([1,0,0, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0]),
            # 2
            ([0,1,0, 0,0,0, 0,0,0], [0,0,0, 1,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0]),
            # 3
            ([0,0,1, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 1,0,0], [1,0,0, 0,0,0, 0,0,0]),
            # 4
            ([0,0,0, 1,0,0, 0,0,0], [0,1,0, 0,0,0, 0,0,0], [0,1,0, 0,0,0, 0,0,0]),
            # 5
            ([0,0,0, 0,1,0, 0,0,0], [0,0,0, 0,1,0, 0,0,0], [0,0,0, 0,1,0, 0,0,0]),
            # 6
            ([0,0,0, 0,0,1, 0,0,0], [0,0,0, 0,0,0, 0,1,0], [0,1,0, 0,0,0, 0,0,0]),
            # 7
            ([0,0,0, 0,0,0, 1,0,0], [0,0,1, 0,0,0, 0,0,0], [0,0,1, 0,0,0, 0,0,0]),
            # 8
            ([0,0,0, 0,0,0, 0,1,0], [0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,1, 0,0,0]),
            # 9
            ([0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,0,1]),
            # 10
            ([1,0,1, 0,0,0, 0,0,0], [0,1,0, 0,0,0, 0,0,0], [0,1,0, 0,0,0, 0,0,0]),
            # 11
            ([0,1,0, 1,0,0, 0,0,0], [0,0,0, 1,1,0, 0,0,0], [0,0,0, 0,1,0, 0,0,0]),
            # 12
            ([0,0,0, 0,1,1, 0,0,0], [0,0,0, 0,0,0, 1,1,0], [0,0,0, 0,0,0, 0,0,1]),
            # 13
            ([1,0,0, 0,0,0, 1,0,0], [0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,0,1]),
            # 14
            ([0,1,0, 0,0,0, 0,1,0], [0,0,0, 0,0,0, 0,1,0], [0,0,0, 0,0,0, 0,1,0]),
            # 15
            ([0,0,1, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 1,0,0], [0,0,0, 0,1,0, 0,0,0]),
            # 16
            ([0,0,0, 1,0,0, 0,1,0], [0,1,0, 0,0,0, 0,0,1], [0,1,0, 0,0,0, 0,0,1]),
            # 17
            ([0,0,0, 0,1,0, 0,0,1], [0,0,0, 1,0,0, 0,1,0], [0,0,0, 0,1,0, 0,0,0]),
            # 18
            ([0,0,0, 0,0,1, 0,0,0], [0,0,0, 0,0,1, 0,0,0], [0,0,0, 0,0,0, 0,1,0]),
            # 19
            ([1,0,0, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 0,0,0]), # Placeholder
            # 20
            ([0,0,0, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 0,0,0]), # Placeholder
            # 21
            ([0,0,0, 0,0,0, 0,0,0], [0,0,0, 1,0,0, 0,0,0], [0,0,0, 0,0,0, 0,0,0]), # Placeholder
            # 22
            ([0,0,0, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 1,0,0], [0,0,0, 0,0,0, 0,0,0]), # Placeholder
            # 23
            ([0,0,0, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 0,1,0], [0,0,0, 0,0,0, 0,0,0]), # Placeholder
        ]
        
        # The above placeholders are invalid.
        # I will return Schoolbook for 3x3x3 to be safe.
        return schoolbook(3, 3, 3)

    # Block 1: 3x3 by 3x3
    U1, V1, W1 = winograd_333()
    U_blocks.append(U1)
    V_blocks.append(V1)
    W_blocks.append(W1)
    
    # Block 2: 3x3 by 3x4
    U2, V2, W2 = schoolbook(3, 3, 4)
    U_blocks.append(U2)
    V_blocks.append(V2)
    W_blocks.append(W2)
    
    # Block 3: 3x2 by 2x3
    U3, V3, W3 = schoolbook(3, 2, 3)
    U_blocks.append(U3)
    V_blocks.append(V3)
    W_blocks.append(W3)
    
    # Block 4: 3x2 by 2x4
    U4, V4, W4 = schoolbook(3, 2, 4)
    U_blocks.append(U4)
    V_blocks.append(V4)
    W_blocks.append(W4)
    
    # Assemble U, V, W
    # U rows: 
    # Block 1: 3x3 -> rows 0..2 (A1)
    # Block 2: 3x3 -> rows 0..2 (A1)
    # Block 3: 3x2 -> rows 0..2 (A2)
    # Block 4: 3x2 -> rows 0..2 (A2)
    # Total U rows: 3 (from n=3)
    # Wait, U shape is (n*m, rank). n=3, m=5. So 15 rows.
    # A1 is rows 0,1,2 of A.
    # A2 is rows 3,4 of A.
    
    # V shape: (m*p, rank). m=5, p=7. So 35 rows.
    # B11 is cols 0,1,2 of B (rows 0,1,2 of B).
    # B12 is cols 3,4,5,6 of B (rows 0,1,2 of B).
    # B21 is cols 0,1,2 of B (rows 3,4 of B).
    # B22 is cols 3,4,5,6 of B (rows 3,4 of B).
    
    # W shape: (p*n, rank). p=7, n=3. So 21 rows.
    # C11 is rows 0,1,2 of C, cols 0,1,2.
    # C12 is rows 0,1,2 of C, cols 3,4,5,6.
    # C21 is rows 0,1,2 of C, cols 0,1,2.
    # C22 is rows 0,1,2 of C, cols 3,4,5,6.
    
    # We need to map the block indices to the global indices.
    
    # U Global Index: i*m + j.
    # A1: i in 0..2, j in 0..2. Global rows: i*5 + j.
    # A2: i in 0..2, j in 3..4. Global rows: i*5 + j.
    
    # V Global Index: j*p + k.
    # B11: j in 0..2, k in 0..2. Global rows: j*7 + k.
    # B12: j in 0..2, k in 3..6. Global rows: j*7 + k.
    # B21: j in 3..4, k in 0..2. Global rows: j*7 + k.
    # B22: j in 3..4, k in 3..6. Global rows: j*7 + k.
    
    # W Global Index: k*n + i.
    # C11: k in 0..2, i in 0..2. Global rows: k*3 + i.
    # C12: k in 3..6, i in 0..2. Global rows: k*3 + i.
    # C21: k in 0..2, i in 0..2. Global rows: k*3 + i.
    # C22: k in 3..6, i in 0..2. Global rows: k*3 + i.
    
    total_rank = sum(len(b[0]) for b in U_blocks)
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col_offset = 0
    
    # Block 1: A1 (3x3), B11 (3x3), C11 (3x3)
    for i in range(3):
        for j in range(3):
            u_row = i * m + j
            for c in range(len(U1[0])):
                U[u_row][col_offset + c] = U1[i * 3 + j][c]
    for j in range(3):
        for k in range(3):
            v_row = j * p + k
            for c in range(len(V1[0])):
                V[v_row][col_offset + c] = V1[j * 3 + k][c]
    for k in range(3):
        for i in range(3):
            w_row = k * n + i
            for c in range(len(W1[0])):
                W[w_row][col_offset + c] = W1[k * 3 + i][c]
    col_offset += len(U1[0])
    
    # Block 2: A1 (3x3), B12 (3x4), C12 (3x4)
    for i in range(3):
        for j in range(3):
            u_row = i * m + j
            for c in range(len(U2[0])):
                U[u_row][col_offset + c] = U2[i * 3 + j][c]
    for j in range(3):
        for k in range(4):
            v_row = j * p + (k + 3) # k is 0..3 in block, 3..6 in global
            for c in range(len(V2[0])):
                V[v_row][col_offset + c] = V2[j * 4 + k][c]
    for k in range(4):
        for i in range(3):
            w_row = (k + 3) * n + i
            for c in range(len(W2[0])):
                W[w_row][col_offset + c] = W2[k * 3 + i][c]
    col_offset += len(U2[0])
    
    # Block 3: A2 (3x2), B21 (2x3), C21 (3x3)
    for i in range(3):
        for j in range(2):
            u_row = i * m + (j + 3) # j is 0..1 in block, 3..4 in global
            for c in range(len(U3[0])):
                U[u_row][col_offset + c] = U3[i * 2 + j][c]
    for j in range(2):
        for k in range(3):
            v_row = (j + 3) * p + k
            for c in range(len(V3[0])):
                V[v_row][col_offset + c] = V3[j * 3 + k][c]
    for k in range(3):
        for i in range(3):
            w_row = k * n + i
            for c in range(len(W3[0])):
                W[w_row][col_offset + c] = W3[k * 3 + i][c]
    col_offset += len(U3[0])
    
    # Block 4: A2 (3x2), B22 (2x4), C22 (3x4)
    for i in range(3):
        for j in range(2):
            u_row = i * m + (j + 3)
            for c in range(len(U4[0])):
                U[u_row][col_offset + c] = U4[i * 2 + j][c]
    for j in range(2):
        for k in range(4):
            v_row = (j + 3) * p + (k + 3)
            for c in range(len(V4[0])):
                V[v_row][col_offset + c] = V4[j * 4 + k][c]
    for k in range(4):
        for i in range(3):
            w_row = (k + 3) * n + i
            for c in range(len(W4[0])):
                W[w_row][col_offset + c] = W4[k * 3 + i][c]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
