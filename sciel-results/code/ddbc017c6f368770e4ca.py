"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 19 decomposition
    rank = 19
    U = [[0] * rank for _ in range(n * m)] # 6 x 19
    V = [[0] * rank for _ in range(m * p)] # 15 x 19
    W = [[0] * rank for _ in range(p * n)] # 10 x 19

    # Part 1: j=0 (M_2,1,5) Rank 1
    # U[0,0]=1, U[1,0]=1 (i=0,1; j=0) -> indices 0, 1 in U
    # V[0,k]=1 (j=0) -> indices 0..4 in V
    # W[k,i]=1 (k=0..4, i=0,1) -> indices k*2+i in W
    c = 0
    for i in range(n):
        U[i * m][c] = 1
    for k in range(p):
        V[0 * p + k][c] = 1
    for k in range(p):
        for i in range(n):
            W[k * n + i][c] = 1
    c += 1

    # Part 2: j in {1,2} (M_2,2,5)
    # Split B cols into 2, 2, 1
    # Col 0,1: Strassen
    # Col 2,3: Strassen
    # Col 4: Schoolbook Rank 4
    
    # Helper to add Strassen block
    def add_strassen(j_offset, k_offset):
        nonlocal c
        # A indices: i in {0,1}, j in {1,2} (offset 1,2)
        # B indices: j in {1,2}, k in {k_offset, k_offset+1}
        # 7 mults
        # S1 = a11*a21? No.
        # A = [[a1, a2], [b1, b2]]
        # B = [[c1, c2], [d1, d2]]
        # P1 = a1 * (c1 - d1)
        # P2 = (a1 + b1) * d1
        # P3 = b2 * (c2 - d2)
        # P4 = b2 * (c2 + d2)
        # P5 = (a1 + b2) * c2
        # P6 = (a1 - b1) * (c1 + d1)
        # P7 = (a2 + b2) * (c2 - d2) -- wait, standard Strassen:
        
        # Standard Strassen for 2x2:
        # M1 = (A11 + A22)(B11 + B22)
        # M2 = (A21 + A22)B11
        # M3 = A11(B21 - B22)
        # M4 = A22(B12 - B11)
        # M5 = (A11 + A12)B22
        # M6 = (A21 - A11)(B11 + B12)
        # M7 = (A12 - A22)(B21 + B22)
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6

        # Indices in U:
        # A11: i=0, j=1+j_offset (idx 1) -> U[1*1] = U[1]
        # A12: i=0, j=2+j_offset (idx 2) -> U[2*1] = U[2]
        # A21: i=1, j=1+j_offset (idx 1) -> U[1*2] = U[3]
        # A22: i=1, j=2+j_offset (idx 2) -> U[4*1] = U[4]
        
        # Indices in V:
        # B11: j=1+j_offset, k=k_offset (idx 0) -> V[(1+j_offset)*p + k_offset]
        # B12: j=1+j_offset, k=k_offset+1 (idx 1) -> V[(1+j_offset)*p + k_offset+1]
        # B21: j=2+j_offset, k=k_offset (idx 0) -> V[(2+j_offset)*p + k_offset]
        # B22: j=2+j_offset, k=k_offset+1 (idx 1) -> V[(2+j_offset)*p + k_offset+1]

        # Indices in W:
        # C11: k=k_offset, i=0 -> W[k_offset*n]
        # C12: k=k_offset+1, i=0 -> W[(k_offset+1)*n]
        # C21: k=k_offset, i=1 -> W[k_offset*n + 1]
        # C22: k=k_offset+1, i=1 -> W[(k_offset+1)*n + 1]

        # 7 steps
        steps = [
            # M1
            {
                'U': {1: 1, 4: 1}, 
                'V': {(1+j_offset)*p + k_offset: 1, (2+j_offset)*p + k_offset+1: 1},
                'W': {k_offset*n: 1, (k_offset+1)*n: 1, k_offset*n+1: -1, (k_offset+1)*n+1: 1} 
                # C11 += 1, C12 += 0? No.
                # M1 adds to C11, C22.
                # C11 = M1... C22 = M1...
                # So W[k_off, 0] += 1, W[k_off+1, 1] += 1.
            },
        ]
        # I will just hardcode the 7 terms for speed and correctness
        
        # Term 1: M1 = (A11+A22)(B11+B22) -> C11 += 1, C22 += 1
        U[1][c] += 1; U[4][c] += 1
        V[(1+j_offset)*p + k_offset][c] += 1; V[(2+j_offset)*p + k_offset+1][c] += 1
        W[k_offset*n][c] += 1; W[(k_offset+1)*n + 1][c] += 1
        c += 1
        
        # Term 2: M2 = (A21+A22)B11 -> C21 += 1, C22 -= 1
        U[3][c] += 1; U[4][c] += 1
        V[(1+j_offset)*p + k_offset][c] += 1
        W[k_offset*n + 1][c] += 1; W[(k_offset+1)*n + 1][c] -= 1
        c += 1
        
        # Term 3: M3 = A11(B21-B22) -> C12 += 1, C22 += 1
        U[1][c] += 1
        V[(2+j_offset)*p + k_offset][c] += 1; V[(2+j_offset)*p + k_offset+1][c] -= 1
        W[(k_offset+1)*n][c] += 1; W[(k_offset+1)*n + 1][c] += 1
        c += 1
        
        # Term 4: M4 = A22(B12-B11) -> C11 += 1, C21 += 1
        U[4][c] += 1
        V[(1+j_offset)*p + k_offset+1][c] += 1; V[(1+j_offset)*p + k_offset][c] -= 1
        W[k_offset*n][c] += 1; W[k_offset*n + 1][c] += 1
        c += 1
        
        # Term 5: M5 = (A11+A12)B22 -> C11 -= 1, C12 += 1
        U[1][c] += 1; U[2][c] += 1
        V[(2+j_offset)*p + k_offset+1][c] += 1
        W[k_offset*n][c] -= 1; W[(k_offset+1)*n][c] += 1
        c += 1
        
        # Term 6: M6 = (A21-A11)(B11+B12) -> C22 += 1
        U[3][c] += 1; U[1][c] -= 1
        V[(1+j_offset)*p + k_offset][c] += 1; V[(1+j_offset)*p + k_offset+1][c] += 1
        W[(k_offset+1)*n + 1][c] += 1
        c += 1
        
        # Term 7: M7 = (A12-A22)(B21+B22) -> C11 += 1, C12 -= 1
        U[2][c] += 1; U[4][c] -= 1
        V[(2+j_offset)*p + k_offset][c] += 1; V[(2+j_offset)*p + k_offset+1][c] += 1
        W[k_offset*n][c] += 1; W[(k_offset+1)*n][c] -= 1
        c += 1

    # Apply Strassen for cols 0,1 (j_offset=1, k_offset=0)
    add_strassen(1, 0)
    
    # Apply Strassen for cols 2,3 (j_offset=1, k_offset=2)
    add_strassen(1, 2)
    
    # Part 3: Col 4 (M_2,2,1) Rank 4
    # A' * B_3 where B_3 = [[i], [n]]
    # C_3 = [a i + b n, c i + d n]
    # U: A11, A12, A21, A22
    # V: B21 (j=1, k=4), B22 (j=2, k=4)
    # W: C21 (k=4, i=0), C22 (k=4, i=1)
    
    # Term 8: A11 * B21 -> C21 += 1
    U[1][c] = 1
    V[(1+1)*p + 4][c] = 1 # j=1, k=4
    W[4*n + 0][c] = 1
    c += 1
    
    # Term 9: A12 * B22 -> C21 += 1
    U[2][c] = 1
    V[(2+1)*p + 4][c] = 1 # j=2, k=4
    W[4*n + 0][c] = 1
    c += 1
    
    # Term 10: A21 * B21 -> C22 += 1
    U[3][c] = 1
    V[(1+1)*p + 4][c] = 1
    W[4*n + 1][c] = 1
    c += 1
    
    # Term 11: A22 * B22 -> C22 += 1
    U[4][c] = 1
    V[(2+1)*p + 4][c] = 1
    W[4*n + 1][c] = 1
    c += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
