"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=8
    # Use Strassen's algorithm for each pair of output columns
    # B is 2x8, group columns into 4 pairs
    # Each pair uses Strassen's 7 rank-1 terms
    # Total rank = 7 * 4 = 28
    
    rank = 28
    U = [[0] * rank for _ in range(n * m)]  # 4 x 28
    V = [[0] * rank for _ in range(m * p)]  # 16 x 28
    W = [[0] * rank for _ in range(p * n)]  # 16 x 28
    
    # Strassen's 7 combinations for 2x2 multiplication
    # A is 2x2: A[0,0]=A00, A[0,1]=A01, A[1,0]=A10, A[1,1]=A11
    # B pair is 2x2: B[j,k] for j in {0,1}, k in {0,1}
    # Output C is 2x2: C[i,k] for i in {0,1}, k in {0,1}
    
    # Strassen's 7 intermediate products:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    
    # U factors (for A): each of 7 products has coefficients for A00, A01, A10, A11
    # A indices: A00 -> 0, A01 -> 1, A10 -> 2, A11 -> 3
    U_factors = [
        [1, 0, 0, 1],   # M1: A00 + A11
        [0, 0, 1, 1],   # M2: A10 + A11
        [1, 0, 0, 0],   # M3: A00
        [0, 0, 0, 1],   # M4: A11
        [1, 1, 0, 0],   # M5: A00 + A01
        [-1, 0, 1, 0],  # M6: A10 - A00
        [0, 1, 0, -1],  # M7: A01 - A11
    ]
    
    # For each column pair s (0..3), columns 2s and 2s+1
    # V factors: for each Strassen product, coefficients for B[0,2s], B[0,2s+1], B[1,2s], B[1,2s+1]
    # B indices: B[j,k] -> j*p + k = j*8 + k
    # For pair s: B[0,2s], B[0,2s+1], B[1,2s], B[1,2s+1]
    
    # W factors: for each Strassen product, contributions to C[0,2s], C[0,2s+1], C[1,2s], C[1,2s+1]
    # C indices: C[i,k] -> k*n + i = k*2 + i
    
    # Strassen reconstruction:
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    W_factors = [
        [1, 0, 0, 0, 1, 0, 0, 0],   # M1: C00 += 1, C11 += 1
        [0, 0, 0, 0, 0, 0, 1, 0],   # M2: C10 += 1
        [0, 1, 0, 0, 0, 0, 1, 0],   # M3: C01 += 1, C11 += 1
        [0, 0, 1, 0, 0, 0, 0, 0],   # M4: C00 += 1
        [0, 1, 0, 0, -1, 0, 0, 0],  # M5: C01 += 1, C00 -= 1
        [0, 0, 0, 0, 0, 1, 0, 0],   # M6: C11 += 1
        [0, 0, 0, 1, -1, 0, 0, 0],  # M7: C00 -= 1, C01 += 1
    ]
    
    # Wait, let me re-derive Strassen carefully.
    # Standard Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    #
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # V factors for each M:
    # M1: (B00+B11) -> B[0,0]=1, B[1,1]=1
    # M2: B00 -> B[0,0]=1
    # M3: (B10-B11) -> B[1,0]=1, B[1,1]=-1
    # M4: (B01-B00) -> B[0,1]=1, B[0,0]=-1
    # M5: B11 -> B[1,1]=1
    # M6: (B00+B10) -> B[0,0]=1, B[1,0]=1
    # M7: (B01+B11) -> B[0,1]=1, B[1,1]=1
    
    # For column pair s (columns 2s, 2s+1):
    # B[0,2s] index = 0*8+2s = 2s
    # B[0,2s+1] index = 0*8+2s+1 = 2s+1
    # B[1,2s] index = 1*8+2s = 8+2s
    # B[1,2s+1] index = 1*8+2s+1 = 8+2s+1
    
    # V_factors[s][m][j] for j in {0,2s, 8+2s, 8+2s+1}
    # Let me define V contributions per Strassen product:
    # M1: B[0,2s]=1, B[1,2s+1]=1
    # M2: B[0,2s]=1
    # M3: B[1,2s]=1, B[1,2s+1]=-1
    # M4: B[0,2s+1]=1, B[0,2s]=-1
    # M5: B[1,2s+1]=1
    # M6: B[0,2s]=1, B[1,2s]=1
    # M7: B[0,2s+1]=1, B[1,2s+1]=1
    
    # W factors: C[i,k] index = k*2 + i
    # C[0,2s] = 2s, C[0,2s+1] = 2s+1, C[1,2s] = 2s+2, C[1,2s+1] = 2s+3
    # M1: C00+=1, C11+=1 -> C[0,2s]+=1, C[1,2s+1]+=1
    # M2: C10+=1 -> C[1,2s]+=1
    # M3: C01+=1, C11+=1 -> C[0,2s+1]+=1, C[1,2s+1]+=1
    # M4: C00+=1 -> C[0,2s]+=1
    # M5: C01+=1, C00-=1 -> C[0,2s+1]+=1, C[0,2s]-=1
    # M6: C11+=1 -> C[1,2s+1]+=1
    # M7: C00-=1, C01+=1 -> C[0,2s]-=1, C[0,2s+1]+=1
    
    for s in range(4):  # 4 column pairs
        col_base = s * 7
        # For each of 7 Strassen products
        for m_idx in range(7):
            col = col_base + m_idx
            
            # Set U factors
            for a_idx in range(4):
                U[a_idx][col] = U_factors[m_idx][a_idx]
            
            # Set V factors
            # B[0,2s] = index 2s
            # B[0,2s+1] = index 2s+1
            # B[1,2s] = index 8+2s
            # B[1,2s+1] = index 8+2s+1
            b00_idx = 2 * s
            b01_idx = 2 * s + 1
            b10_idx = 8 + 2 * s
            b11_idx = 8 + 2 * s + 1
            
            if m_idx == 0:  # M1
                V[b00_idx][col] = 1
                V[b11_idx][col] = 1
            elif m_idx == 1:  # M2
                V[b00_idx][col] = 1
            elif m_idx == 2:  # M3
                V[b10_idx][col] = 1
                V[b11_idx][col] = -1
            elif m_idx == 3:  # M4
                V[b01_idx][col] = 1
                V[b00_idx][col] = -1
            elif m_idx == 4:  # M5
                V[b11_idx][col] = 1
            elif m_idx == 5:  # M6
                V[b00_idx][col] = 1
                V[b10_idx][col] = 1
            elif m_idx == 6:  # M7
                V[b01_idx][col] = 1
                V[b11_idx][col] = 1
            
            # Set W factors
            # C[0,2s] = index 2s
            # C[0,2s+1] = index 2s+1
            # C[1,2s] = index 2s+2
            # C[1,2s+1] = index 2s+3
            c00_idx = 2 * s
            c01_idx = 2 * s + 1
            c10_idx = 2 * s + 2
            c11_idx = 2 * s + 3
            
            if m_idx == 0:  # M1
                W[c00_idx][col] = 1
                W[c11_idx][col] = 1
            elif m_idx == 1:  # M2
                W[c10_idx][col] = 1
            elif m_idx == 2:  # M3
                W[c01_idx][col] = 1
                W[c11_idx][col] = 1
            elif m_idx == 3:  # M4
                W[c00_idx][col] = 1
            elif m_idx == 4:  # M5
                W[c01_idx][col] = 1
                W[c00_idx][col] = -1
            elif m_idx == 5:  # M6
                W[c11_idx][col] = 1
            elif m_idx == 6:  # M7
                W[c00_idx][col] = -1
                W[c01_idx][col] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
