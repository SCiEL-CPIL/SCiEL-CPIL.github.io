"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We decompose M_{2,3,4} into two M_{2,3,2} blocks.
    # Block 1: B columns 0, 1. Output C columns 0, 1.
    # Block 2: B columns 2, 3. Output C columns 2, 3.
    
    # Helper to generate schoolbook decomposition for n x m by m x p
    def get_schoolbook(n, m, p):
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

    # Get decomposition for 2 x 3 by 3 x 2
    U1, V1, W1 = get_schoolbook(n, m, 2)
    U2, V2, W2 = get_schoolbook(n, m, 2)
    
    rank = 24
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Fill U: Same A for both blocks
    for r in range(n * m):
        for c in range(12):
            U[r][c] = U1[r][c]
            U[r][c + 12] = U2[r][c]
            
    # Fill V: 
    # V1 corresponds to B cols 0,1. Global indices: j*4 + 0, j*4 + 1.
    # V2 corresponds to B cols 2,3. Global indices: j*4 + 2, j*4 + 3.
    # In local 3x2 matrix, indices are j*2 + 0, j*2 + 1.
    for j in range(m):
        # Local row 0 (k=0) -> Global row j*4 + 0
        # Local row 1 (k=1) -> Global row j*4 + 1
        # Local row 0 (k=0) -> Global row j*4 + 2
        # Local row 1 (k=1) -> Global row j*4 + 3
        
        for c in range(12):
            # V1 part
            V[j * 4 + 0][c] = V1[j * 2 + 0][c]
            V[j * 4 + 1][c] = V1[j * 2 + 1][c]
            # V2 part
            V[j * 4 + 2][c + 12] = V2[j * 2 + 0][c]
            V[j * 4 + 3][c + 12] = V2[j * 2 + 1][c]
            
    # Fill W:
    # W1 corresponds to C cols 0,1. Global indices: k*n + i.
    # In local 2x2 output, indices are k*2 + i.
    # W2 corresponds to C cols 2,3. Global indices: (k+2)*n + i.
    for i in range(n):
        for k in range(2): # k is 0,1 in local
            # Local row k*2 + i -> Global row k*n + i
            for c in range(12):
                W[k * n + i][c] = W1[k * 2 + i][c]
                W[(k + 2) * n + i][c + 12] = W2[k * 2 + i][c]
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
