import time

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct an exact tensor decomposition for 3x3 * 3x6 matrix multiplication.
    Uses a rank-52 decomposition derived from two rank-26 3x3x3 decompositions.
    Each 3x3x3 decomposition uses Strassen for the 2x2 block and schoolbook for the rest.
    """
    start_time = time.time()
    deadline = start_time + 120
    
    n, m, p = DIMENSIONS
    
    # We will build the rank-26 decomposition for 3x3x3 first.
    # Then we will duplicate it for the two halves of the 3x3x6 problem.
    
    # Helper to get Strassen 2x2x2 decomposition
    # Returns U, V, W for 2x2x2, rank 7
    def get_strassen_2x2x2():
        # Indices: 0,1 for rows/cols
        # Rank 7
        U = [[0]*7 for _ in range(4)] # 2x2 flattened
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # M1 = (A00+A11)(B00+B11)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1; W[1][0] = -1; W[2][0] = 1 # C00=M1, C11=M1, C01=-M5+M7? No, let's use standard mapping
        
        # Standard Strassen mapping to C:
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # Let's assign columns 0..6 to M1..M7
        
        # M1
        U[0][0]=1; U[3][0]=1
        V[0][0]=1; V[3][0]=1
        W[0][0]=1; W[3][0]=1
        
        # M2
        U[2][1]=1; U[3][1]=1
        V[0][1]=1
        W[2][1]=1; W[3][1]=-1
        
        # M3
        U[0][2]=1
        V[1][2]=1; V[3][2]=-1
        W[1][2]=1; W[3][2]=1
        
        # M4
        U[3][3]=1
        V[2][3]=1; V[0][3]=-1
        W[0][3]=1; W[2][3]=1
        
        # M5
        U[0][4]=1; U[1][4]=1
        V[3][4]=1
        W[0][4]=-1; W[1][4]=1
        
        # M6
        U[2][5]=1; U[0][5]=-1
        V[0][5]=1; V[1][5]=1
        W[3][5]=1
        
        # M7
        U[1][6]=1; U[3][6]=-1
        V[2][6]=1; V[3][6]=1
        W[0][6]=1; W[1][6]=-1
        
        return U, V, W

    # Build 3x3x3 rank 26 decomposition
    # U, V, W will be 9x26
    U_33 = [[0]*26 for _ in range(9)]
    V_33 = [[0]*26 for _ in range(9)]
    W_33 = [[0]*26 for _ in range(9)]
    
    col_offset = 0
    
    # 1. A11 B11 (2x2 x 2x2) -> Rank 7
    U_s, V_s, W_s = get_strassen_2x2x2()
    for i in range(4):
        for c in range(7):
            # Map 2x2 index i to 3x3 index
            # i=0->0, i=1->1, i=2->6, i=3->7 ?
            # Flattening: 00, 01, 10, 11 -> 0, 1, 3, 4 in 3x3?
            # 3x3 indices:
            # 0 1 2
            # 3 4 5
            # 6 7 8
            # 2x2 block A11 is at 0,1,3,4.
            # So i=0 (00) -> 0
            # i=1 (01) -> 1
            # i=2 (10) -> 3
            # i=3 (11) -> 4
            
            idx_map = [0, 1, 3, 4]
            U_33[idx_map[i]][col_offset + c] = U_s[i][c]
            V_33[idx_map[i]][col_offset + c] = V_s[i][c]
            W_33[idx_map[i]][col_offset + c] = W_s[i][c]
    col_offset += 7
    
    # 2. A12 B21 (2x1 x 1x2) -> Rank 4
    # A12 is col 2 of A (indices 2, 5)
    # B21 is row 2 of B (indices 6, 7)
    # Product is 2x2.
    # C00 += A02 B20, C01 += A02 B21, C10 += A12 B20, C11 += A12 B21
    # A indices: 2, 5. B indices: 6, 7.
    # U: 2, 5. V: 6, 7. W: 0, 1, 3, 4.
    
    # Term 1: A02 * B20 -> C00
    U_33[2][col_offset] = 1
    V_33[6][col_offset] = 1
    W_33[0][col_offset] = 1
    col_offset += 1
    
    # Term 2: A02 * B21 -> C01
    U_33[2][col_offset] = 1
    V_33[7][col_offset] = 1
    W_33[1][col_offset] = 1
    col_offset += 1
    
    # Term 3: A12 * B20 -> C10
    U_33[5][col_offset] = 1
    V_33[6][col_offset] = 1
    W_33[3][col_offset] = 1
    col_offset += 1
    
    # Term 4: A12 * B21 -> C11
    U_33[5][col_offset] = 1
    V_33[7][col_offset] = 1
    W_33[4][col_offset] = 1
    col_offset += 1
    
    # 3. A11 B12 (2x2 x 2x1) -> Rank 4
    # B12 is col 2 of B top part? No, B is 3x3.
    # A11 is 2x2 (0,1,3,4). B12 is 2x1 (1,4)? No.
    # B indices:
    # 0 1 2
    # 3 4 5
    # 6 7 8
    # B12 (rows 0,1; col 2) is indices 2, 5.
    # A11 (rows 0,1; cols 0,1) is indices 0,1,3,4.
    # Product is 2x1.
    # C02 += A00 B02 + A01 B12
    # C12 += A10 B02 + A11 B12
    
    # Term 1: A00 * B02 -> C02
    U_33[0][col_offset] = 1
    V_33[2][col_offset] = 1
    W_33[2][col_offset] = 1
    col_offset += 1
    
    # Term 2: A01 * B12 -> C02
    U_33[1][col_offset] = 1
    V_33[5][col_offset] = 1
    W_33[2][col_offset] = 1
    col_offset += 1
    
    # Term 3: A10 * B02 -> C12
    U_33[3][col_offset] = 1
    V_33[2][col_offset] = 1
    W_33[5][col_offset] = 1
    col_offset += 1
    
    # Term 4: A11 * B12 -> C12
    U_33[4][col_offset] = 1
    V_33[5][col_offset] = 1
    W_33[5][col_offset] = 1
    col_offset += 1
    
    # 4. A12 B22 (2x1 x 1x1) -> Rank 2
    # A12 (2,5). B22 (8).
    # C02 += A02 B22
    # C12 += A12 B22
    
    # Term 1
    U_33[2][col_offset] = 1
    V_33[8][col_offset] = 1
    W_33[2][col_offset] = 1
    col_offset += 1
    
    # Term 2
    U_33[5][col_offset] = 1
    V_33[8][col_offset] = 1
    W_33[5][col_offset] = 1
    col_offset += 1
    
    # 5. A21 B11 (1x2 x 2x2) -> Rank 4
    # A21 (6,7). B11 (0,1,3,4).
    # C20 += A20 B00 + A21 B10
    # C21 += A20 B01 + A21 B11
    
    # Term 1: A20 * B00 -> C20
    U_33[6][col_offset] = 1
    V_33[0][col_offset] = 1
    W_33[6][col_offset] = 1
    col_offset += 1
    
    # Term 2: A21 * B10 -> C20
    U_33[7][col_offset] = 1
    V_33[3][col_offset] = 1
    W_33[6][col_offset] = 1
    col_offset += 1
    
    # Term 3: A20 * B01 -> C21
    U_33[6][col_offset] = 1
    V_33[1][col_offset] = 1
    W_33[7][col_offset] = 1
    col_offset += 1
    
    # Term 4: A21 * B11 -> C21
    U_33[7][col_offset] = 1
    V_33[4][col_offset] = 1
    W_33[7][col_offset] = 1
    col_offset += 1
    
    # 6. A22 B21 (1x1 x 1x2) -> Rank 2
    # A22 (8). B21 (6,7).
    # C20 += A22 B20
    # C21 += A22 B21
    
    # Term 1
    U_33[8][col_offset] = 1
    V_33[6][col_offset] = 1
    W_33[6][col_offset] = 1
    col_offset += 1
    
    # Term 2
    U_33[8][col_offset] = 1
    V_33[7][col_offset] = 1
    W_33[7][col_offset] = 1
    col_offset += 1
    
    # 7. A21 B12 (1x2 x 2x1) -> Rank 2
    # A21 (6,7). B12 (2,5).
    # C22 += A20 B02 + A21 B12
    
    # Term 1: A20 * B02 -> C22
    U_33[6][col_offset] = 1
    V_33[2][col_offset] = 1
    W_33[8][col_offset] = 1
    col_offset += 1
    
    # Term 2: A21 * B12 -> C22
    U_33[7][col_offset] = 1
    V_33[5][col_offset] = 1
    W_33[8][col_offset] = 1
    col_offset += 1
    
    # 8. A22 B22 (1x1 x 1x1) -> Rank 1
    # A22 (8). B22 (8).
    # C22 += A22 B22
    
    U_33[8][col_offset] = 1
    V_33[8][col_offset] = 1
    W_33[8][col_offset] = 1
    col_offset += 1
    
    # Total rank for 3x3x3 is 26.
    
    # Now build 3x3x6
    # Rank = 52
    U = [[0]*52 for _ in range(9)]
    V = [[0]*52 for _ in range(18)] # 3x6 flattened
    W = [[0]*52 for _ in range(18)] # 6x3 flattened
    
    # First half (k=0..2)
    for i in range(9):
        for c in range(26):
            U[i][c] = U_33[i][c]
            V[i][c] = V_33[i][c]
            W[i][c] = W_33[i][c]
            
    # Second half (k=3..5)
    # V indices shift by 3 in the second dimension?
    # V is m x p = 3 x 6. Flattened: j*6 + k.
    # First half k=0..2. Second half k=3..5.
    # W is p x n = 6 x 3. Flattened: k*3 + i.
    # First half k=0..2. Second half k=3..5.
    
    for i in range(9):
        for c in range(26):
            U[i][26 + c] = U_33[i][c]
            # V index: j*6 + k.
            # In V_33, index is j*3 + k (k=0..2).
            # We need to map j*3 + k to j*6 + (k+3).
            # V_33 row index i corresponds to j, k?
            # V_33 is 9x26. Row index is j*3+k.
            # So j = i // 3, k = i % 3.
            # New V row index: j*6 + (k+3).
            j = i // 3
            k = i % 3
            new_v_idx = j * 6 + (k + 3)
            V[new_v_idx][26 + c] = V_33[i][c]
            
            # W index: k*3 + i (i is row of C, which is 0..2).
            # In W_33, row index is k*3 + i_row.
            # So k = i // 3, i_row = i % 3.
            # New W row index: (k+3)*3 + i_row.
            k = i // 3
            i_row = i % 3
            new_w_idx = (k + 3) * 3 + i_row
            W[new_w_idx][26 + c] = W_33[i][c]
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
