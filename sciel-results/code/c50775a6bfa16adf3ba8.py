import time

DIMENSIONS = [3, 3, 8]

def _strassen_2x2x2():
    """
    Returns U, V, W for 2x2 * 2x2 matrix multiplication with rank 7.
    """
    # Strassen's 7 multiplications
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b12-b22)
    # M4 = a22(b21-b11)
    # M5 = (a11+a21)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    
    # U: 4x7, V: 4x7, W: 4x7
    # Indices: U[i,j] where i is A index (0..3), j is mult index (0..6)
    # A indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    # B indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    # C indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1; W[1][0] = -1; W[2][0] = -1
    
    # M2
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = 1
    
    # M3
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # M4
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[2][3] = 1; W[0][3] = 1
    
    # M5
    U[0][4] = 1; U[2][4] = 1
    V[3][4] = 1
    W[0][4] = 1; W[1][4] = 1
    
    # M6
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[2][5] = 1; W[1][5] = -1
    
    # M7
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[1][6] = 1; W[0][6] = -1
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    # Use Rank 68 decomposition
    # Split into two 3x3 * 3x4 blocks
    rank_per_block = 34
    total_rank = rank_per_block * 2
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Helper to add a block
    def add_block(offset, col_start):
        # Block is 3x3 * 3x4
        # Strassen for 2x2 * 2x2 (cols 0-1)
        # Strassen for 2x2 * 2x2 (cols 2-3)
        # Fringe
        
        sU, sV, sW = _strassen_2x2x2()
        
        # Block 1: Cols 0-1 of the 3x4 block
        # A indices 0,1; B indices 0,1
        for i in range(4):
            for j in range(7):
                # A row i (0..3) corresponds to A[i//2, i%2]
                # B row i (0..3) corresponds to B[i//2, i%2]
                # C row i (0..3) corresponds to C[i//2, i%2]
                
                # Map to global indices
                # A global: i_row * 3 + i_col
                a_row, a_col = i // 2, i % 2
                U[a_row * 3 + a_col][offset + j] += sU[i][j]
                
                # B global: b_row * 4 + b_col
                # Here b_row is 0,1 (top 2 rows of B)
                # b_col is col_start + (j%2)? No, B block is 2x2.
                # B indices in block: 0,1,2,3 -> (0,0), (0,1), (1,0), (1,1)
                b_row, b_col = i // 2, i % 2
                V[b_row * 8 + (col_start + b_col)][offset + j] += sV[i][j]
                
                # C global: c_col * 3 + c_row
                # C indices in block: 0,1,2,3 -> (0,0), (0,1), (1,0), (1,1)
                c_row, c_col = i // 2, i % 2
                W[(col_start + c_col) * 3 + c_row][offset + j] += sW[i][j]
                
        # Block 2: Cols 2-3 of the 3x4 block
        for i in range(4):
            for j in range(7):
                a_row, a_col = i // 2, i % 2
                U[a_row * 3 + a_col][offset + 7 + j] += sU[i][j]
                
                b_row, b_col = i // 2, i % 2
                V[b_row * 8 + (col_start + 2 + b_col)][offset + 7 + j] += sV[i][j]
                
                c_row, c_col = i // 2, i % 2
                W[(col_start + 2 + c_col) * 3 + c_row][offset + 7 + j] += sW[i][j]
        
        # Fringe Top: A[0:2, 2] * B[2, 0:4]
        # A indices: 0*3+2, 1*3+2
        # B indices: 2*8 + (col_start + 0..3)
        # C indices: (col_start + 0..3)*3 + 0..1
        for k in range(4):
            # A[0,2] * B[2, col_start+k]
            U[0 * 3 + 2][offset + 14 + k] = 1
            V[2 * 8 + (col_start + k)][offset + 14 + k] = 1
            W[(col_start + k) * 3 + 0][offset + 14 + k] = 1
            
            # A[1,2] * B[2, col_start+k]
            U[1 * 3 + 2][offset + 18 + k] = 1
            V[2 * 8 + (col_start + k)][offset + 18 + k] = 1
            W[(col_start + k) * 3 + 1][offset + 18 + k] = 1

        # Fringe Bottom: A[2, 0:2] * B[0:2, 0:4] + A[2, 2] * B[2, 0:4]
        # A[2,0] * B[0, col_start+k]
        for k in range(4):
            U[2 * 3 + 0][offset + 22 + k] = 1
            V[0 * 8 + (col_start + k)][offset + 22 + k] = 1
            W[(col_start + k) * 3 + 2][offset + 22 + k] = 1
            
            # A[2,1] * B[1, col_start+k]
            U[2 * 3 + 1][offset + 26 + k] = 1
            V[1 * 8 + (col_start + k)][offset + 26 + k] = 1
            W[(col_start + k) * 3 + 2][offset + 26 + k] = 1
            
            # A[2,2] * B[2, col_start+k]
            U[2 * 3 + 2][offset + 30 + k] = 1
            V[2 * 8 + (col_start + k)][offset + 30 + k] = 1
            W[(col_start + k) * 3 + 2][offset + 30 + k] = 1

    # Add Block 1 (Cols 0-3)
    add_block(0, 0)
    # Add Block 2 (Cols 4-7)
    add_block(34, 4)
    
    return {"U": U, "V": V, "W": W}
