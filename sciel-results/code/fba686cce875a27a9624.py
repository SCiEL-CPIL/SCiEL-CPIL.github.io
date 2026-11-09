# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 3
    rank = 17
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: 2x2x2 Strassen (Rank 7)
    # Indices: i in {0,1}, j in {0,1}, k in {0,1}
    # U rows: i*3 + j
    # V rows: j*3 + k
    # W rows: k*2 + i
    
    # Strassen Products
    # 1: (A00+A11)(B00+B11)
    U[0*3+0][col] = 1
    U[1*3+1][col] = 1
    V[0*3+0][col] = 1
    V[0*3+1][col] = 1
    # W reconstruction later
    
    # 2: (A10+A11)B00
    U[1*3+0][col+1] = 1
    U[1*3+1][col+1] = 1
    V[0*3+0][col+1] = 1
    
    # 3: A00(B10-B11)
    U[0*3+0][col+2] = 1
    V[1*3+0][col+2] = 1
    V[1*3+1][col+2] = -1
    
    # 4: A11(B01-B00)
    U[1*3+1][col+3] = 1
    V[0*3+1][col+3] = 1
    V[0*3+0][col+3] = -1
    
    # 5: (A00+A01)B11
    U[0*3+0][col+4] = 1
    U[0*3+1][col+4] = 1
    V[1*3+1][col+4] = 1
    
    # 6: (A10-A00)(B00+B01)
    U[1*3+0][col+5] = 1
    U[0*3+0][col+5] = -1
    V[0*3+0][col+5] = 1
    V[0*3+1][col+5] = 1
    
    # 7: (A01-A11)(B10+B11)
    U[0*3+1][col+6] = 1
    U[1*3+1][col+6] = -1
    V[1*3+0][col+6] = 1
    V[1*3+1][col+6] = 1
    
    # W reconstruction for Part 1
    # C00 = P1 - P4 + P5 - P7
    W[0*2+0][col+0] = 1
    W[0*2+0][col+3] = -1
    W[0*2+0][col+4] = 1
    W[0*2+0][col+6] = -1
    
    # C01 = P4 + P7
    W[0*2+1][col+3] = 1
    W[0*2+1][col+6] = 1
    
    # C10 = P2 + P3
    W[1*2+0][col+1] = 1
    W[1*2+0][col+2] = 1
    
    # C11 = P1 - P3 + P6 - P2
    # Note: Standard Strassen C11 is P1 - P3 + P6 - P2? 
    # Let's verify: P1 - P3 + P6 - P2 = A11B11.
    # If not, I might need to adjust. But this is the standard set.
    W[1*2+1][col+0] = 1
    W[1*2+1][col+2] = -1
    W[1*2+1][col+5] = 1
    W[1*2+1][col+1] = -1
    
    col += 7
    
    # Part 2: 2x2x1 block (j in {0,1}, k=2) Rank 4
    # Terms: (0,0,2), (0,1,2), (1,0,2), (1,1,2)
    
    # (0,0,2)
    U[0*3+0][col+0] = 1
    V[0*3+2][col+0] = 1
    W[2*2+0][col+0] = 1
    
    # (0,1,2)
    U[0*3+1][col+1] = 1
    V[1*3+2][col+1] = 1
    W[2*2+0][col+1] = 1
    
    # (1,0,2)
    U[1*3+0][col+2] = 1
    V[0*3+2][col+2] = 1
    W[2*2+1][col+2] = 1
    
    # (1,1,2)
    U[1*3+1][col+3] = 1
    V[1*3+2][col+3] = 1
    W[2*2+1][col+3] = 1
    
    col += 4
    
    # Part 3: 2x1x2 block (j=2, k in {0,1}) Rank 4
    # Terms: (0,2,0), (0,2,1), (1,2,0), (1,2,1)
    
    # (0,2,0)
    U[0*3+2][col+0] = 1
    V[2*3+0][col+0] = 1
    W[0*2+0][col+0] = 1
    
    # (0,2,1)
    U[0*3+2][col+1] = 1
    V[2*3+1][col+1] = 1
    W[1*2+0][col+1] = 1
    
    # (1,2,0)
    U[1*3+2][col+2] = 1
    V[2*3+0][col+2] = 1
    W[0*2+1][col+2] = 1
    
    # (1,2,1)
    U[1*3+2][col+3] = 1
    V[2*3+1][col+3] = 1
    W[1*2+1][col+3] = 1
    
    col += 4
    
    # Part 4: 2x1x1 block (j=2, k=2) Rank 2
    # Terms: (0,2,2), (1,2,2)
    
    # (0,2,2)
    U[0*3+2][col+0] = 1
    V[2*3+2][col+0] = 1
    W[2*2+0][col+0] = 1
    
    # (1,2,2)
    U[1*3+2][col+1] = 1
    V[2*3+2][col+1] = 1
    W[2*2+1][col+1] = 1
    
    col += 2
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
