"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 6, 6, 6
    # 3x3 blocks of 2x2 each
    # Strassen decomposition for 2x2: 7 terms
    # Outer: 3x3x3 = 27 block positions
    # Total rank: 27 * 7 = 189
    
    # Strassen factors for 2x2 matrices
    # A indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # B indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # C indices (W[k*2+i]): (i=0,k=0)=0, (i=0,k=1)=1, (i=1,k=0)=2, (i=1,k=1)=3
    
    strassen_U = [
        [1, 0, 0, 1],   # M1: (A00+A11)
        [0, 0, 1, 1],   # M2: (A10+A11)
        [1, 0, 0, 0],   # M3: A00
        [0, 0, 0, 1],   # M4: A11
        [1, 1, 0, 0],   # M5: (A00+A01)
        [-1, 0, 1, 0],  # M6: (A10-A00)
        [0, 1, 0, -1],  # M7: (A01-A11)
    ]
    strassen_V = [
        [1, 0, 0, 1],   # M1: (B00+B11)
        [1, 0, 0, 0],   # M2: B00
        [0, 1, 0, -1],  # M3: (B01-B11)
        [0, -1, 1, 0],  # M4: (B10-B00)
        [0, 0, 0, 1],   # M5: B11
        [1, 1, 0, 0],   # M6: (B00+B01)
        [0, 0, 1, 1],   # M7: (B10+B11)
    ]
    strassen_W = [
        [1, 0, 0, 1],   # M1: C00+=1, C11+=1
        [0, 0, 1, 0],   # M2: C10+=1
        [0, 1, 0, 0],   # M3: C01+=1
        [1, 0, 1, 0],   # M4: C00+=1, C10+=1
        [-1, 1, 0, 0],  # M5: C00-=1, C01+=1
        [0, 0, 0, 1],   # M6: C11+=1
        [1, 0, 0, 0],   # M7: C00+=1
    ]
    
    num_blocks = 3  # 3x3 grid of 2x2 blocks
    block_size = 2
    rank = num_blocks ** 3 * 7  # 27 * 7 = 189
    
    U = [[0] * rank for _ in range(n * m)]  # 36 x 189
    V = [[0] * rank for _ in range(m * p)]  # 36 x 189
    W = [[0] * rank for _ in range(p * n)]  # 36 x 189
    
    col = 0
    for I in range(num_blocks):
        for J in range(num_blocks):
            for K in range(num_blocks):
                for t in range(7):
                    # U: block (I,J), Strassen term t
                    for r in range(block_size):
                        for s in range(block_size):
                            u_row = (I * block_size + r) * m + (J * block_size + s)
                            U[u_row][col] += strassen_U[t][r * block_size + s]
                    
                    # V: block (J,K), Strassen term t
                    for s in range(block_size):
                        for tt in range(block_size):
                            v_row = (J * block_size + s) * p + (K * block_size + tt)
                            V[v_row][col] += strassen_V[t][s * block_size + tt]
                    
                    # W: block (I,K), Strassen term t
                    # W[k*n+i] where i = I*block_size+r, k = K*block_size+tt
                    for r in range(block_size):
                        for tt in range(block_size):
                            w_row = (K * block_size + tt) * n + (I * block_size + r)
                            W[w_row][col] += strassen_W[t][r * block_size + tt]
                    
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
