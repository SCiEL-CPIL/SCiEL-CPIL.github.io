"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 22 decomposition
    rank = 22
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Helper to add a Strassen 2x2x2 block
    def add_strassen_block(offset_u, offset_v, offset_w, k_offset):
        # offset_u, offset_v, offset_w are column indices in U, V, W
        # k_offset is the offset for the k index in V and W
        
        # 1. A00 * (B01 - B11)
        U[offset_u + 0][offset_u] = 1.0
        V[offset_v + 1][offset_u] = 1.0  # j=0, k=1
        V[offset_v + 3][offset_u] = -1.0 # j=1, k=1
        W[offset_w + 1][offset_u] = 1.0  # k=1, i=0
        W[offset_w + 3][offset_u] = -1.0 # k=1, i=1
        
        # 2. (A00 + A10) * B11
        U[offset_u + 0][offset_u + 1] = 1.0
        U[offset_u + 2][offset_u + 1] = 1.0
        V[offset_v + 3][offset_u + 1] = 1.0 # j=1, k=1
        W[offset_w + 0][offset_u + 1] = -1.0 # k=0, i=0
        W[offset_w + 2][offset_u + 1] = 1.0  # k=0, i=1
        
        # 3. A10 * (B00 - B10)
        U[offset_u + 2][offset_u + 2] = 1.0
        V[offset_v + 0][offset_u + 2] = 1.0 # j=0, k=0
        V[offset_v + 2][offset_u + 2] = -1.0# j=1, k=0
        W[offset_w + 2][offset_u + 2] = 1.0  # k=0, i=1
        
        # 4. A11 * B10
        U[offset_u + 3][offset_u + 3] = 1.0
        V[offset_v + 2][offset_u + 3] = 1.0 # j=1, k=0
        W[offset_w + 0][offset_u + 3] = 1.0  # k=0, i=0
        W[offset_w + 2][offset_u + 3] = 1.0  # k=0, i=1
        
        # 5. (A00 + A11) * (B00 + B11)
        U[offset_u + 0][offset_u + 4] = 1.0
        U[offset_u + 3][offset_u + 4] = 1.0
        V[offset_v + 0][offset_u + 4] = 1.0 # j=0, k=0
        V[offset_v + 3][offset_u + 4] = 1.0 # j=1, k=1
        W[offset_w + 0][offset_u + 4] = 1.0  # k=0, i=0
        W[offset_w + 2][offset_u + 4] = 1.0  # k=0, i=1
        
        # 6. (A01 - A11) * (B10 + B11)
        U[offset_u + 1][offset_u + 5] = 1.0
        U[offset_u + 3][offset_u + 5] = -1.0
        V[offset_v + 2][offset_u + 5] = 1.0 # j=1, k=0
        V[offset_v + 3][offset_u + 5] = 1.0 # j=1, k=1
        W[offset_w + 0][offset_u + 5] = 1.0  # k=0, i=0
        W[offset_w + 2][offset_u + 5] = 1.0  # k=0, i=1
        
        # 7. (A00 - A10) * (B00 + B01)
        U[offset_u + 0][offset_u + 6] = 1.0
        U[offset_u + 2][offset_u + 6] = -1.0
        V[offset_v + 0][offset_u + 6] = 1.0 # j=0, k=0
        V[offset_v + 1][offset_u + 6] = 1.0 # j=0, k=1
        W[offset_w + 2][offset_u + 6] = 1.0  # k=0, i=1
        W[offset_w + 3][offset_u + 6] = 1.0  # k=0, i=1? No, k=1, i=1 is index 3.
        # Wait, W index is k*n + i.
        # k=0, i=0 -> 0
        # k=0, i=1 -> 1
        # k=1, i=0 -> 2
        # k=1, i=1 -> 3
        
        # Let's re-verify W indices for Strassen.
        # C00 = M5 + M4 - M2 + M6
        # C01 = M1 + M5
        # C10 = M2 + M3
        # C11 = M3 + M4 - M1 + M7
        
        # My previous mapping:
        # W[0, 5]=1, W[0, 4]=1, W[0, 2]=-1, W[0, 6]=1. (C00)
        # W[2, 1]=1, W[2, 5]=1. (C01)
        # W[1, 2]=1, W[1, 3]=1. (C10)
        # W[3, 3]=1, W[3, 4]=1, W[3, 1]=-1, W[3, 7]=1. (C11)
        
        # Let's rewrite the helper with explicit W assignments to avoid confusion.
        
    # We will just hardcode the 22 columns.
    
    # Block 1: Strassen for j=0,1 and k=0,1
    # Columns 0-6
    
    # 1. A00 * (B01 - B11)
    U[0][0] = 1.0
    V[1][0] = 1.0
    V[5][0] = -1.0
    W[2][0] = 1.0
    W[3][0] = -1.0
    
    # 2. (A00 + A10) * B11
    U[0][1] = 1.0
    U[2][1] = 1.0
    V[5][1] = 1.0
    W[0][1] = -1.0
    W[1][1] = 1.0
    
    # 3. A10 * (B00 - B10)
    U[2][2] = 1.0
    V[0][2] = 1.0
    V[4][2] = -1.0
    W[1][2] = 1.0
    
    # 4. A11 * B10
    U[3][3] = 1.0
    V[4][3] = 1.0
    W[0][3] = 1.0
    W[1][3] = 1.0
    
    # 5. (A00 + A11) * (B00 + B11)
    U[0][4] = 1.0
    U[3][4] = 1.0
    V[0][4] = 1.0
    V[5][4] = 1.0
    W[0][4] = 1.0
    W[1][4] = 1.0
    
    # 6. (A01 - A11) * (B10 + B11)
    U[1][5] = 1.0
    U[3][5] = -1.0
    V[4][5] = 1.0
    V[5][5] = 1.0
    W[0][5] = 1.0
    W[1][5] = 1.0
    
    # 7. (A00 - A10) * (B00 + B01)
    U[0][6] = 1.0
    U[2][6] = -1.0
    V[0][6] = 1.0
    V[1][6] = 1.0
    W[1][6] = 1.0
    W[3][6] = 1.0
    
    # Block 2: Strassen for j=0,1 and k=2,3
    # Columns 7-13
    # k_offset = 2.
    # V indices: j*p + k.
    # j=0, k=2 -> 2. j=0, k=3 -> 3.
    # j=1, k=2 -> 6. j=1, k=3 -> 7.
    # W indices: k*n + i.
    # k=2, i=0 -> 4. k=2, i=1 -> 5.
    # k=3, i=0 -> 6. k=3, i=1 -> 7.
    
    # 1. A00 * (B03 - B13)
    U[0][7] = 1.0
    V[3][7] = 1.0
    V[7][7] = -1.0
    W[6][7] = 1.0
    W[7][7] = -1.0
    
    # 2. (A00 + A10) * B13
    U[0][8] = 1.0
    U[2][8] = 1.0
    V[7][8] = 1.0
    W[4][8] = -1.0
    W[5][8] = 1.0
    
    # 3. A10 * (B02 - B12)
    U[2][9] = 1.0
    V[2][9] = 1.0
    V[6][9] = -1.0
    W[5][9] = 1.0
    
    # 4. A11 * B12
    U[3][10] = 1.0
    V[6][10] = 1.0
    W[4][10] = 1.0
    W[5][10] = 1.0
    
    # 5. (A00 + A11) * (B02 + B13)
    U[0][11] = 1.0
    U[3][11] = 1.0
    V[2][11] = 1.0
    V[7][11] = 1.0
    W[4][11] = 1.0
    W[5][11] = 1.0
    
    # 6. (A01 - A11) * (B12 + B13)
    U[1][12] = 1.0
    U[3][12] = -1.0
    V[6][12] = 1.0
    V[7][12] = 1.0
    W[4][12] = 1.0
    W[5][12] = 1.0
    
    # 7. (A00 - A10) * (B02 + B03)
    U[0][13] = 1.0
    U[2][13] = -1.0
    V[2][13] = 1.0
    V[3][13] = 1.0
    W[5][13] = 1.0
    W[7][13] = 1.0
    
    # Block 3: Schoolbook for j=2
    # Columns 14-21
    # i in 0,1. k in 0,1,2,3.
    # U[i*3+2, c] = 1
    # V[2*4+k, c] = 1
    # W[k*2+i, c] = 1
    
    for i in range(2):
        for k in range(4):
            c = 14 + i * 4 + k
            U[i * 3 + 2][c] = 1.0
            V[2 * 4 + k][c] = 1.0
            W[k * 2 + i][c] = 1.0
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
