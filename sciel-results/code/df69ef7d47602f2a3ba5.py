"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Decompose 4x4x5 matrix multiplication using:
    - Strassen (rank 7) recursively for the 4x4x4 block: rank 49
    - Schoolbook for the 4x4x1 block: rank 16
    Total rank: 65
    """
    n, m, p = DIMENSIONS  # n=4, m=4, p=5
    
    def strassen_2x2():
        """Return (U, V, W) for 2x2x2 Strassen decomposition.
        U shape: (4, 7) - indices i*2+j
        V shape: (4, 7) - indices j*2+k  
        W shape: (4, 7) - indices k*2+i
        """
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # Term 0: M1 = (A00+A11)(B00+B11)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1
        
        # Term 1: M2 = (A10+A11)B00
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[1][1] = 1
        
        # Term 2: M3 = A00(B10-B11)
        U[0][2] = 1
        V[2][2] = 1; V[3][2] = -1
        W[2][2] = 1
        
        # Term 3: M4 = A11(B01-B00)
        U[3][3] = 1
        V[1][3] = 1; V[0][3] = -1
        W[3][3] = 1
        
        # Term 4: M5 = (A00+A01)B11
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = -1
        
        # Term 5: M6 = (A10-A00)(B00+B01)
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1
        
        # Term 6: M7 = (A01-A11)(B10+B11)
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][6] = 1
        
        return U, V, W
    
    def strassen_4x4():
        """Return (U, V, W) for 4x4x4 Strassen decomposition.
        U shape: (16, 49) - indices i*4+j
        V shape: (16, 49) - indices j*4+k
        W shape: (16, 49) - indices k*4+i
        """
        U = [[0]*49 for _ in range(16)]
        V = [[0]*49 for _ in range(16)]
        W = [[0]*49 for _ in range(16)]
        
        block_offsets = {0: (0,0), 1: (0,2), 2: (2,0), 3: (2,2)}
        
        terms = [
            ([ (0,1), (3,1) ], [ (0,1), (3,1) ], [ (0,1), (3,1) ]),
            ([ (2,1), (3,1) ], [ (0,1) ], [ (2,1) ]),
            ([ (0,1) ], [ (2,1), (3,-1) ], [ (1,1) ]),
            ([ (3,1) ], [ (1,1), (0,-1) ], [ (3,1) ]),
            ([ (0,1), (1,1) ], [ (3,1) ], [ (0,-1) ]),
            ([ (2,1), (0,-1) ], [ (0,1), (1,1) ], [ (3,1) ]),
            ([ (1,1), (3,-1) ], [ (2,1), (3,1) ], [ (0,1) ]),
        ]
        
        U2, V2, W2 = strassen_2x2()
        
        for t in range(7):
            A_blocks, B_blocks, C_blocks = terms[t]
            for s in range(7):
                col = t * 7 + s
                for (ablock, asgn) in A_blocks:
                    bro, bco = block_offsets[ablock]
                    for r in range(4):
                        if U2[r][s] != 0:
                            i_local = r // 2
                            j_local = r % 2
                            u_idx = (bro + i_local) * 4 + (bco + j_local)
                            U[u_idx][col] += asgn * U2[r][s]
                
                for (bblock, bsng) in B_blocks:
                    bro, bco = block_offsets[bblock]
                    for r in range(4):
                        if V2[r][s] != 0:
                            j_local = r // 2
                            k_local = r % 2
                            v_idx = (bro + j_local) * 4 + (bco + k_local)
                            V[v_idx][col] += bsng * V2[r][s]
                
                for (cblock, csng) in C_blocks:
                    bro, bco = block_offsets[cblock]
                    for r in range(4):
                        if W2[r][s] != 0:
                            k_local = r // 2
                            i_local = r % 2
                            w_idx = (bro + k_local) * 4 + (bco + i_local)
                            W[w_idx][col] += csng * W2[r][s]
        
        return U, V, W
    
    U4, V4, W4 = strassen_4x4()
    
    total_rank = 49 + 16
    U = [[0] * total_rank for _ in range(16)]
    V = [[0] * total_rank for _ in range(20)]
    W = [[0] * total_rank for _ in range(20)]
    
    for i in range(16):
        for c in range(49):
            U[i][c] = U4[i][c]
    
    for j in range(16):
        for c in range(49):
            V[j][c] = V4[j][c]
    
    for k in range(16):
        for c in range(49):
            W[k][c] = W4[k][c]
    
    col_offset = 49
    for i in range(4):
        for j in range(4):
            col = col_offset + i * 4 + j
            U[i * 4 + j][col] = 1
            V[j * 5 + 4][col] = 1
            W[4 * 4 + i][col] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
