import numpy as np

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 15
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a term
    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val):
        U[u_idx][col] += u_val
        V[v_idx][col] += v_val
        W[w_idx][col] += w_val

    col = 0
    
    # Strassen terms for 2x2x2 block (k in 0,1)
    # Indices:
    # U: i in {0,1}, j in {0,1} -> rows 0,1,3,4
    # V: j in {0,1}, k in {0,1} -> rows 0,1,4,5
    # W: k in {0,1}, i in {0,1} -> rows 0,1,2,3
    
    # M1 = (A00+A11)(B00+B11)
    add_term(0, 1, 0, 1, 0, 1) # C00
    add_term(4, 1, 5, 1, 0, 1) # C00 (same term, different U/V/W parts? No, U/V/W are vectors)
    # Wait, U, V, W are matrices. U[i, r], V[j, r], W[k, r].
    # For a single rank-1 term, we define U[:, r], V[:, r], W[:, r].
    # M1:
    # U: i=0,j=0 (0) = 1; i=1,j=1 (4) = 1
    # V: j=0,k=0 (0) = 1; j=1,k=1 (5) = 1
    # W: 
    # C00 (k=0,i=0 -> W[0]) += 1
    # C01 (k=1,i=0 -> W[2]) += 1
    # C10 (k=0,i=1 -> W[1]) += 1
    # C11 (k=1,i=1 -> W[3]) += -1
    
    U[0][col] = 1; U[4][col] = 1
    V[0][col] = 1; V[5][col] = 1
    W[0][col] = 1; W[2][col] = 1; W[1][col] = 1; W[3][col] = -1
    col += 1
    
    # M2 = (A10+A11)B00
    # U: i=1,j=0 (3) = 1; i=1,j=1 (4) = 1
    # V: j=0,k=0 (0) = 1
    # W:
    # C10 (W[1]) += 1
    # C11 (W[3]) += -1
    U[3][col] = 1; U[4][col] = 1
    V[0][col] = 1
    W[1][col] = 1; W[3][col] = -1
    col += 1
    
    # M3 = A00 B10
    # U: i=0,j=0 (0) = 1
    # V: j=1,k=0 (4) = 1
    # W:
    # C01 (W[2]) += 1
    # C11 (W[3]) += 1
    U[0][col] = 1
    V[4][col] = 1
    W[2][col] = 1; W[3][col] = 1
    col += 1
    
    # M4 = A11 B11
    # U: i=1,j=1 (4) = 1
    # V: j=1,k=1 (5) = 1
    # W:
    # C00 (W[0]) += 1
    # C10 (W[1]) += 1
    U[4][col] = 1
    V[5][col] = 1
    W[0][col] = 1; W[1][col] = 1
    col += 1
    
    # M5 = (A00+A01)B11
    # U: i=0,j=0 (0) = 1; i=0,j=1 (1) = 1
    # V: j=1,k=1 (5) = 1
    # W:
    # C00 (W[0]) += -1
    # C01 (W[2]) += 1
    U[0][col] = 1; U[1][col] = 1
    V[5][col] = 1
    W[0][col] = -1; W[2][col] = 1
    col += 1
    
    # M6 = (A10-A00)(B00+B01)
    # U: i=1,j=0 (3) = 1; i=0,j=0 (0) = -1
    # V: j=0,k=0 (0) = 1; j=0,k=1 (1) = 1
    # W:
    # C11 (W[3]) += 1
    U[3][col] = 1; U[0][col] = -1
    V[0][col] = 1; V[1][col] = 1
    W[3][col] = 1
    col += 1
    
    # M7 = (A01-A11)(B10+B11)
    # U: i=0,j=1 (1) = 1; i=1,j=1 (4) = -1
    # V: j=1,k=0 (4) = 1; j=1,k=1 (5) = 1
    # W:
    # C00 (W[0]) += 1
    U[1][col] = 1; U[4][col] = -1
    V[4][col] = 1; V[5][col] = 1
    W[0][col] = 1
    col += 1
    
    # Second block k in {2,3}
    # V shift: +2 in k index -> +2 in row index (since p=4, row = 4j+k)
    # W shift: +2 in k index -> +4 in row index (since n=2, row = 2k+i)
    
    # M1'
    U[0][col] = 1; U[4][col] = 1
    V[2][col] = 1; V[7][col] = 1 # V[0+2], V[5+2]
    W[4][col] = 1; W[6][col] = 1; W[5][col] = 1; W[7][col] = -1 # W[0+4], W[2+4], W[1+4], W[3+4]
    col += 1
    
    # M2'
    U[3][col] = 1; U[4][col] = 1
    V[2][col] = 1
    W[5][col] = 1; W[7][col] = -1
    col += 1
    
    # M3'
    U[0][col] = 1
    V[6][col] = 1 # V[4+2]
    W[6][col] = 1; W[7][col] = 1
    col += 1
    
    # M4'
    U[4][col] = 1
    V[7][col] = 1
    W[4][col] = 1; W[5][col] = 1
    col += 1
    
    # M5'
    U[0][col] = 1; U[1][col] = 1
    V[7][col] = 1
    W[4][col] = -1; W[6][col] = 1
    col += 1
    
    # M6'
    U[3][col] = 1; U[0][col] = -1
    V[2][col] = 1; V[3][col] = 1 # V[0+2], V[1+2]
    W[7][col] = 1
    col += 1
    
    # M7'
    U[1][col] = 1; U[4][col] = -1
    V[6][col] = 1; V[7][col] = 1 # V[4+2], V[5+2]
    W[4][col] = 1
    col += 1
    
    # Term 3: A2 B2 (j=2)
    # U: i=0,j=2 (2) = 1; i=1,j=2 (5) = 1
    # V: j=2,k=0..3 (8,9,10,11) = 1
    # W: k=0..3, i=0..1 (0..7) = 1
    U[2][col] = 1; U[5][col] = 1
    V[8][col] = 1; V[9][col] = 1; V[10][col] = 1; V[11][col] = 1
    for w in range(8):
        W[w][col] = 1
    col += 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
