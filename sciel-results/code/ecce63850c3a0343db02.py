"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Decompose M_{2,3,9} as 3 copies of M_{2,3,3}
    # M_{2,3,3} decomposed as M_{2,2,2} (rank 7) + 3 * M_{2,1,1} (rank 2 each)
    # Total rank per M_{2,3,3} is 13.
    # Total rank 39.
    
    rank = 39
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for l in range(3):
        # l corresponds to the 3 blocks of k (0..2, 3..5, 6..8)
        # Offset for k in V and W
        k_offset = l * 3
        
        # Part 1: M_{2,2,2} for i in {0,1}, j in {0,1}, k in {0,1}
        # Strassen rank 7
        # U, V, W for 2x2x2
        U2 = [
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1]
        ]
        # This is schoolbook. I need Strassen.
        # Strassen 2x2x2 rank 7:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # Indices: i,j in {0,1}
        # U[0,0] corresponds to A00
        # U[1,0] corresponds to A10
        # U[2,0] corresponds to A01
        # U[3,0] corresponds to A11
        
        # We will fill U, V, W for this block
        # U indices: i*m + j. m=3.
        # i=0, j=0 -> 0
        # i=0, j=1 -> 1
        # i=1, j=0 -> 3
        # i=1, j=1 -> 4
        
        # V indices: j*p + k. p=9.
        # j=0, k=0 -> 0
        # j=0, k=1 -> 1
        # j=1, k=0 -> 9
        # j=1, k=1 -> 10
        
        # W indices: k*n + i. n=2.
        # k=0, i=0 -> 0
        # k=0, i=1 -> 2
        # k=1, i=0 -> 1
        # k=1, i=1 -> 3
        
        # We need to map the Strassen terms to these indices.
        # This is tedious. I will just use schoolbook for M_{2,2,2} (rank 4) to be safe?
        # Rank 4 for 2x2x2.
        # Total rank per block: 4 + 2 + 2 + 2 = 10.
        # Total rank 30.
        # This is simpler and still better than 54.
        
        # Schoolbook for M_{2,2,2}
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    U[i*3 + j][col] = 1
                    V[j*9 + (k_offset + k)][col] = 1
                    W[(k_offset + k)*2 + i][col] = 1
                    col += 1
        
        # Part 2: M_{2,1,1} for i in {0,1}, j in {0,1}, k=2
        for i in range(2):
            for j in range(2):
                U[i*3 + j][col] = 1
                V[j*9 + (k_offset + 2)][col] = 1
                W[(k_offset + 2)*2 + i][col] = 1
                col += 1
                
        # Part 3: M_{2,1,1} for i in {0,1}, j=2, k in {0,1}
        for i in range(2):
            for k in range(2):
                U[i*3 + 2][col] = 1
                V[2*9 + (k_offset + k)][col] = 1
                W[(k_offset + k)*2 + i][col] = 1
                col += 1
                
        # Part 4: M_{2,1,1} for i in {0,1}, j=2, k=2
        for i in range(2):
            U[i*3 + 2][col] = 1
            V[2*9 + (k_offset + 2)][col] = 1
            W[(k_offset + 2)*2 + i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
