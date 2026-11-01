"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 26 decomposition using Strassen for the 2x2 core
    rank = 26
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Indices for A (n x m) and B (m x p)
    # A: 0 1 2
    #   3 4 5
    #   6 7 8
    # B: 0 1 2
    #   3 4 5
    #   6 7 8
    
    # 1. Strassen for top-left 2x2 (A[0:2, 0:2], B[0:2, 0:2])
    # A_sub = [[0, 1], [3, 4]]
    # B_sub = [[0, 1], [3, 4]]
    
    # M1 = (a00 + a11) * (b00 + b11)
    U[0][0] = 1; U[4][0] = 1
    V[0][0] = 1; V[4][0] = 1
    W[0][0] = 1; W[3][0] = 1; W[1][0] = 1; W[4][0] = 1 # C00, C01, C10, C11 contributions?
    # Actually, we need to map the Strassen results to the full 3x3 output.
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Let's assign columns 0-6 to Strassen terms.
    # Col 0: M1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[4][0] = 1
    V[0][0] = 1; V[4][0] = 1
    W[0][0] = 1 # C00
    W[3][0] = -1 # C10? No, C10 is index 3 in W? 
    # W index is k*n + i. 
    # C00: i=0, k=0 -> idx 0.
    # C01: i=0, k=1 -> idx 1.
    # C10: i=1, k=0 -> idx 3.
    # C11: i=1, k=1 -> idx 4.
    
    # M1 contributes to C00, C01, C10, C11?
    # Strassen M1 = (a+d)(e+h).
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # So:
    # M1: W[0]+=1, W[4]+=1
    # M2: W[3]+=1, W[4]-=1
    # M3: W[1]+=1, W[4]+=1
    # M4: W[0]+=1, W[3]+=1
    # M5: W[0]-=1, W[1]+=1
    # M6: W[4]+=1
    # M7: W[0]+=1
    
    # Let's fill U, V, W for cols 0-6.
    
    # Col 0: M1
    U[0][0] = 1; U[4][0] = 1
    V[0][0] = 1; V[4][0] = 1
    W[0][0] = 1; W[4][0] = 1
    
    # Col 1: M2 = (A10+A11) * B00
    U[3][1] = 1; U[4][1] = 1
    V[0][1] = 1
    W[3][1] = 1; W[4][1] = -1
    
    # Col 2: M3 = A00 * (B01 - B11)
    U[0][2] = 1
    V[1][2] = 1; V[4][2] = -1
    W[1][2] = 1; W[4][2] = 1
    
    # Col 3: M4 = A11 * (B10 - B00)
    U[4][3] = 1
    V[3][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[3][3] = 1
    
    # Col 4: M5 = (A00+A01) * B11
    U[0][4] = 1; U[1][4] = 1
    V[4][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # Col 5: M6 = (A10 - A00) * (B00 + B01)
    U[3][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[4][5] = 1
    
    # Col 6: M7 = (A01+A11) * (B10+B11)
    U[1][6] = 1; U[4][6] = 1
    V[3][6] = 1; V[4][6] = 1
    W[0][6] = 1
    
    # 2. Remaining terms for 3x3
    # C02 = A00*B02 + A01*B12 + A02*B22
    # C12 = A10*B02 + A11*B12 + A12*B22
    # C20 = A20*B00 + A21*B10 + A22*B20
    # C21 = A20*B01 + A21*B11 + A22*B21
    # C22 = A20*B02 + A21*B12 + A22*B22
    
    # We can use standard multiplication for these, but we can optimize.
    # Let's just use standard multiplication for the 10 remaining terms (C02, C12, C20, C21, C22).
    # That's 5 terms * 3 mults = 15 mults.
    # Total rank = 7 + 15 = 22?
    # Wait, C02, C12, C20, C21, C22 are 5 elements.
    # Each is a dot product of length 3.
    # So 5 * 3 = 15 multiplications.
    # Total rank = 7 + 15 = 22.
    
    # Let's add these 15 columns.
    col = 7
    # C02: i=0, k=2 -> W idx 2
    for j in range(3):
        U[0*m + j][col] = 1 # A0j
        V[j*p + 2][col] = 1 # B j2
        W[2*n + 0][col] = 1 # C02
        col += 1
        
    # C12: i=1, k=2 -> W idx 5
    for j in range(3):
        U[1*m + j][col] = 1 # A1j
        V[j*p + 2][col] = 1 # B j2
        W[2*n + 1][col] = 1 # C12
        col += 1
        
    # C20: i=2, k=0 -> W idx 6
    for j in range(3):
        U[2*m + j][col] = 1 # A2j
        V[j*p + 0][col] = 1 # B j0
        W[0*n + 2][col] = 1 # C20
        col += 1
        
    # C21: i=2, k=1 -> W idx 7
    for j in range(3):
        U[2*m + j][col] = 1 # A2j
        V[j*p + 1][col] = 1 # B j1
        W[1*n + 2][col] = 1 # C21
        col += 1
        
    # C22: i=2, k=2 -> W idx 8
    for j in range(3):
        U[2*m + j][col] = 1 # A2j
        V[j*p + 2][col] = 1 # B j2
        W[2*n + 2][col] = 1 # C22
        col += 1
        
    # Total rank used: 7 + 15 = 22.
    # We declared rank 26, but we only used 22.
    # We can trim the matrices to rank 22.
    
    U = [row[:22] for row in U]
    V = [row[:22] for row in V]
    W = [row[:22] for row in W]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
