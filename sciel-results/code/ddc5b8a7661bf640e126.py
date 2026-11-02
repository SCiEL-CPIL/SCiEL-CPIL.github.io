import numpy as np

def construct():
    n, m, p = 2, 2, 7
    
    # Helper to create a zero matrix
    def zeros(rows, cols):
        return [[0] * cols for _ in range(rows)]
    
    # Strassen decomposition for 2x2x2
    # U (2x7), V (2x7), W (2x7)
    # Indices: U[i, r], V[j, r], W[k, r]
    # i, j, k in 0..1
    
    U_s = zeros(2, 7)
    V_s = zeros(2, 7)
    W_s = zeros(2, 7)
    
    # P1 = (a+d)(e+h)
    U_s[0][0] = 1; U_s[1][0] = 1
    V_s[0][0] = 1; V_s[1][0] = 1
    W_s[0][0] = 1; W_s[1][0] = 1
    
    # P2 = (c+d)e
    U_s[0][1] = 0; U_s[1][1] = 1
    V_s[0][1] = 1; V_s[1][1] = 1
    W_s[0][1] = 1; W_s[1][1] = 0 # Wait, W index is k. C21 = P2+P4. C22 = P1-P2+P3+P7.
    # Let's use the standard Strassen formulas carefully.
    # C11 = P1 + P4 - P5 - P7
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P2 + P3 + P7
    
    # P1: (A11+A22)(B11+B22) -> i=0,1; j=0,1; k=0,1
    # U: [1, 1], V: [1, 1], W: [1, 1]
    
    # P2: (A21+A22)B11 -> i=1; j=0,1; k=0
    # U: [0, 1], V: [1, 1], W: [1, 0]
    
    # P3: A11(B12-B22) -> i=0; j=0; k=1,0
    # U: [1, 0], V: [1, 0], W: [1, -1]
    
    # P4: A22(B21-B11) -> i=1; j=1; k=1,0
    # U: [0, 1], V: [0, 1], W: [-1, 1]
    
    # P5: (A11+A12)B22 -> i=0,1; j=1; k=1
    # U: [1, 1], V: [0, 1], W: [0, 1]
    
    # P6: (A21-A11)(B11+B12) -> i=1,0; j=0; k=0,1
    # U: [1, -1], V: [1, 0], W: [1, 1]
    
    # P7: (A12-A22)(B21+B22) -> i=0,1; j=1; k=1,0
    # U: [1, -1], V: [0, 1], W: [1, 1]
    
    # Let's fill the arrays based on this.
    # Note: The W index k corresponds to the column of C.
    # For 2x2x2, k is 0 or 1.
    
    # Reset
    U_s = [[0]*7 for _ in range(2)]
    V_s = [[0]*7 for _ in range(2)]
    W_s = [[0]*7 for _ in range(2)]
    
    # r=0: P1
    U_s[0][0] = 1; U_s[1][0] = 1
    V_s[0][0] = 1; V_s[1][0] = 1
    W_s[0][0] = 1; W_s[1][0] = 1
    
    # r=1: P2
    U_s[0][1] = 0; U_s[1][1] = 1
    V_s[0][1] = 1; V_s[1][1] = 1
    W_s[0][1] = 1; W_s[1][1] = 0
    
    # r=2: P3
    U_s[0][2] = 1; U_s[1][2] = 0
    V_s[0][2] = 1; V_s[1][2] = 0
    W_s[0][2] = 1; W_s[1][2] = -1
    
    # r=3: P4
    U_s[0][3] = 0; U_s[1][3] = 1
    V_s[0][3] = 0; V_s[1][3] = 1
    W_s[0][3] = -1; W_s[1][3] = 1
    
    # r=4: P5
    U_s[0][4] = 1; U_s[1][4] = 1
    V_s[0][4] = 0; V_s[1][4] = 1
    W_s[0][4] = 0; W_s[1][4] = 1
    
    # r=5: P6
    U_s[0][5] = 1; U_s[1][5] = -1
    V_s[0][5] = 1; V_s[1][5] = 0
    W_s[0][5] = 1; W_s[1][5] = 1
    
    # r=6: P7
    U_s[0][6] = 1; U_s[1][6] = -1
    V_s[0][6] = 0; V_s[1][6] = 1
    W_s[0][6] = 1; W_s[1][6] = 1
    
    # Schoolbook for 2x2x3
    # Rank 12
    U_sc = [[0]*12 for _ in range(2)]
    V_sc = [[0]*12 for _ in range(2)]
    W_sc = [[0]*12 for _ in range(3)]
    
    col = 0
    for i in range(2):
        for j in range(2):
            for k in range(3):
                U_sc[i][col] = 1
                V_sc[j][col] = 1
                W_sc[k][col] = 1
                col += 1
                
    # Combine
    # Total rank = 7 + 7 + 12 = 26
    rank = 26
    U = [[0]*rank for _ in range(2)]
    V = [[0]*rank for _ in range(2)]
    W = [[0]*rank for _ in range(7)]
    
    # Block 1: 2x2x2 (cols 0,1 of B, cols 0,1 of C)
    for r in range(7):
        for i in range(2):
            U[i][r] += U_s[i][r]
        for j in range(2):
            V[j][r] += V_s[j][r]
        for k in range(2):
            W[k][r] += W_s[k][r]
            
    # Block 2: 2x2x2 (cols 2,3 of B, cols 2,3 of C)
    offset = 7
    for r in range(7):
        for i in range(2):
            U[i][offset+r] += U_s[i][r]
        for j in range(2):
            V[j][offset+r] += V_s[j][r]
        for k in range(2):
            W[2+k][offset+r] += W_s[k][r]
            
    # Block 3: 2x2x3 (cols 4,5,6 of B, cols 4,5,6 of C)
    offset = 14
    for r in range(12):
        for i in range(2):
            U[i][offset+r] += U_sc[i][r]
        for j in range(2):
            V[j][offset+r] += V_sc[j][r]
        for k in range(3):
            W[4+k][offset+r] += W_sc[k][r]
            
    return {"U": U, "V": V, "W": W}
