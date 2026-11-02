"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 44 decomposition
    rank = 44
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen decomposition for 2x2x2
    # Local A indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # Local B indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # Local W indices: 0:(0,0), 1:(1,0), 2:(0,1), 3:(1,1)
    
    strassen_U = [
        [1, 0, 1, 0],  # P1
        [0, 0, 1, 1],  # P2
        [1, 0, 0, 0],  # P3
        [0, 0, 0, 1],  # P4
        [1, 1, 0, 0],  # P5
        [-1, 0, 1, 0], # P6
        [0, 1, -1, 0]  # P7
    ]
    
    strassen_V = [
        [1, 0, 0, 1],  # P1
        [1, 0, 0, 0],  # P2
        [0, 1, 0, -1], # P3
        [0, 0, 1, -1], # P4
        [0, 0, 0, 1],  # P5
        [1, 1, 0, 0],  # P6
        [0, 0, 1, 1]   # P7
    ]
    
    strassen_W = [
        [1, 0, 0, 1],  # P1
        [0, 1, 0, -1], # P2
        [0, 0, 1, 1],  # P3
        [1, 1, 0, 0],  # P4
        [-1, 0, 1, 0], # P5
        [0, 0, 0, 1],  # P6
        [1, 0, 0, 0]   # P7
    ]
    
    # Extra terms for A col 2, B row 2
    # A indices: 2:(0,2), 5:(1,2) in global 2x3 A
    # B indices: 4:(2,0), 5:(2,1) in local 3x2 B
    # W indices: 0,1,2,3 in local 2x2 C
    
    extra_U = [
        [1, 0], # A02 for C00
        [1, 0], # A02 for C01
        [0, 1], # A12 for C10
        [0, 1]  # A12 for C11
    ]
    
    extra_V = [
        [1, 0], # B20 for C00
        [0, 1], # B21 for C01
        [1, 0], # B20 for C10
        [0, 1]  # B21 for C11
    ]
    
    extra_W = [
        [1, 0, 0, 0], # C00
        [0, 0, 1, 0], # C01
        [0, 1, 0, 0], # C10
        [0, 0, 0, 1]  # C11
    ]
    
    col = 0
    for block in range(4):
        # Strassen part
        for c in range(7):
            # U
            # Global A indices: 0,1,3,4
            U[0][col] = strassen_U[c][0]
            U[1][col] = strassen_U[c][1]
            U[3][col] = strassen_U[c][2]
            U[4][col] = strassen_U[c][3]
            
            # V
            # Global B indices: 0*8+2*block, 0*8+2*block+1, 1*8+2*block, 1*8+2*block+1
            v_base = block * 2
            V[v_base][col] = strassen_V[c][0]
            V[v_base + 1][col] = strassen_V[c][1]
            V[8 + v_base][col] = strassen_V[c][2]
            V[8 + v_base + 1][col] = strassen_V[c][3]
            
            # W
            # Global C indices: 2*block*2+0, 2*block*2+1, (2*block+1)*2+0, (2*block+1)*2+1
            w_base = block * 4
            W[w_base][col] = strassen_W[c][0]
            W[w_base + 1][col] = strassen_W[c][1]
            W[w_base + 2][col] = strassen_W[c][2]
            W[w_base + 3][col] = strassen_W[c][3]
            
            col += 1
            
        # Extra part
        for c in range(4):
            # U
            # Global A indices: 2, 5
            U[2][col] = extra_U[c][0]
            U[5][col] = extra_U[c][1]
            
            # V
            # Global B indices: 2*8+2*block, 2*8+2*block+1
            v_base = 16 + block * 2
            V[v_base][col] = extra_V[c][0]
            V[v_base + 1][col] = extra_V[c][1]
            
            # W
            # Global C indices: same as Strassen
            w_base = block * 4
            W[w_base][col] = extra_W[c][0]
            W[w_base + 1][col] = extra_W[c][1]
            W[w_base + 2][col] = extra_W[c][2]
            W[w_base + 3][col] = extra_W[c][3]
            
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
