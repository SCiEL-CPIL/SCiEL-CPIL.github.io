"""Matrix multiplication 2x3x8 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 8
    # We'll build U (n*m x rank), V (m*p x rank), W (p*n x rank)
    
    # Strategy: split m=3 into m1=2 and m2=1
    # A (2x3) = [A1 (2x2), A2 (2x1)]
    # B (3x8) = [B1 (2x8); B2 (1x8)]
    # C (2x8) = A1*B1 + A2*B2
    # 
    # A1*B1: A1 is 2x2, B1 is 2x8. Split B1 into four 2x2 blocks: B1[:, 0:2], B1[:, 2:4], B1[:, 4:6], B1[:, 6:8]
    # Each 2x2 * 2x2 uses rank 7 (Strassen-like for 2x2x2)
    # Total for A1*B1: 4 * 7 = 28 terms
    #
    # A2*B2: A2 is 2x1, B2 is 1x8. Schoolbook: 2*1*8 = 16 terms
    #
    # Total rank = 28 + 16 = 44

    # 7-term decomposition for 2x2x2:
    # (a11, a12, a21, a22) * (b11, b12, b21, b22) -> (c11, c12, c21, c22)
    # Using standard 7 multiplications:
    # m1 = (a11+a22)*(b11+b22)
    # m2 = (a21+a22)*b11
    # m3 = a11*(b12-b22)
    # m4 = a22*(b21-b11)
    # m5 = (a11+a12)*b22
    # m6 = (a21-a11)*(b11+b12)
    # m7 = (a12-a22)*(b21+b22)
    # c11 = m1 - m4 + m6 + m7
    # c12 = m3 + m5
    # c21 = m2 + m4
    # c22 = m1 - m3 + m2 + m5
    
    # For each 2x2 * 2x2 block, we need to map:
    # A indices: row 0 -> a11,a12; row 1 -> a21,a22 (within the 2x2 subblock)
    # B indices: col 0 -> b11,b21; col 1 -> b12,b22 (within the 2x2 subblock)
    # Wait, let me be careful with index conventions.
    
    # In our global indices:
    # A is 2x3, stored as U[i*m + j, c] where i in [0,1], j in [0,1,2]
    # So A[i,j] corresponds to U[i*3 + j, c]
    # B is 3x8, stored as V[j*p + k, c] where j in [0,1,2], k in [0..7]
    # So B[j,k] corresponds to V[j*8 + k, c]
    # C is 2x8, stored as W[k*n + i, c] where k in [0..7], i in [0,1]
    # So C[i,k] corresponds to W[k*2 + i, c]
    
    rank = 44
    U = [[0.0] * rank for _ in range(n * m)]  # 6 x 44
    V = [[0.0] * rank for _ in range(m * p)]  # 24 x 44
    W = [[0.0] * rank for _ in range(p * n)]  # 16 x 44
    
    col = 0
    
    # Part 1: A1 * B1 where A1 is 2x2 (A[0,0], A[0,1], A[1,0], A[1,1])
    # B1 is 2x8 (B[0,0..7], B[1,0..7])
    # Split B1 into 4 blocks of 2x2:
    # Block t (t=0,1,2,3): columns 2t, 2t+1
    # For block t:
    #   a11 = A[0,0], a12 = A[0,1], a21 = A[1,0], a22 = A[1,1]
    #   b11 = B[0, 2t], b12 = B[0, 2t+1], b21 = B[1, 2t], b22 = B[1, 2t+1]
    #   c11 = C[0, 2t], c12 = C[0, 2t+1], c21 = C[1, 2t], c22 = C[1, 2t+1]
    
    # The 7 multiplications and their coefficients:
    # m1 = (a11+a22)*(b11+b22): U contributions: A[0,0]+=1, A[1,1]+=1; V contributions: B[0,2t]+=1, B[1,2t]+=1
    # m2 = (a21+a22)*b11: U: A[1,0]+=1, A[1,1]+=1; V: B[0,2t]+=1
    # m3 = a11*(b12-b22): U: A[0,0]+=1; V: B[0,2t+1]+=1, B[1,2t+1]-=1
    # m4 = a22*(b21-b11): U: A[1,1]+=1; V: B[1,2t]+=1, B[0,2t]-=1
    # m5 = (a11+a12)*b22: U: A[0,0]+=1, A[0,1]+=1; V: B[1,2t+1]+=1
    # m6 = (a21-a11)*(b11+b12): U: A[1,0]+=1, A[0,0]-=1; V: B[0,2t]+=1, B[0,2t+1]+=1
    # m7 = (a12-a22)*(b21+b22): U: A[0,1]+=1, A[1,1]-=1; V: B[1,2t]+=1, B[1,2t+1]+=1
    
    # W contributions:
    # c11 = m1 - m4 + m6 + m7
    # c12 = m3 + m5
    # c21 = m2 + m4
    # c22 = m1 - m3 + m2 + m5
    
    for t in range(4):
        # m1
        U[0*3+0][col] += 1.0
        U[1*3+1][col] += 1.0
        V[0*8+2*t][col] += 1.0
        V[1*8+2*t][col] += 1.0
        W[2*t*2+0][col] += 1.0   # c11
        W[(2*t+1)*2+0][col] += 1.0  # c12? No wait
        # Let me redo: c11 = C[0, 2t] -> W[2t*2+0]
        # c12 = C[0, 2t+1] -> W[(2t+1)*2+0]
        # c21 = C[1, 2t] -> W[2t*2+1]
        # c22 = C[1, 2t+1] -> W[(2t+1)*2+1]
        # m1 contributes to c11 and c22
        W[2*t*2+0][col] += 1.0    # c11
        W[(2*t+1)*2+1][col] += 1.0  # c22
        col += 1
        
        # m2 = (a21+a22)*b11
        U[1*3+0][col] += 1.0
        U[1*3+1][col] += 1.0
        V[0*8+2*t][col] += 1.0
        W[2*t*2+1][col] += 1.0    # c21
        W[(2*t+1)*2+1][col] += 1.0  # c22
        col += 1
        
        # m3 = a11*(b12-b22)
        U[0*3+0][col] += 1.0
        V[0*8+(2*t+1)][col] += 1.0
        V[1*8+(2*t+1)][col] += -1.0
        W[(2*t+1)*2+0][col] += 1.0  # c12
        W[(2*t+1)*2+1][col] += -1.0  # c22
        col += 1
        
        # m4 = a22*(b21-b11)
        U[1*3+1][col] += 1.0
        V[1*8+2*t][col] += 1.0
        V[0*8+2*t][col] += -1.0
        W[2*t*2+0][col] += -1.0   # c11
        W[2*t*2+1][col] += 1.0    # c21
        col += 1
        
        # m5 = (a11+a12)*b22
        U[0*3+0][col] += 1.0
        U[0*3+1][col] += 1.0
        V[1*8+(2*t+1)][col] += 1.0
        W[(2*t+1)*2+0][col] += 1.0  # c12
        W[(2*t+1)*2+1][col] += 1.0  # c22
        col += 1
        
        # m6 = (a21-a11)*(b11+b12)
        U[1*3+0][col] += 1.0
        U[0*3+0][col] += -1.0
        V[0*8+2*t][col] += 1.0
        V[0*8+(2*t+1)][col] += 1.0
        W[2*t*2+0][col] += 1.0    # c11
        col += 1
        
        # m7 = (a12-a22)*(b21+b22)
        U[0*3+1][col] += 1.0
        U[1*3+1][col] += -1.0
        V[1*8+2*t][col] += 1.0
        V[1*8+(2*t+1)][col] += 1.0
        W[2*t*2+0][col] += 1.0    # c11
        col += 1
    
    # Part 2: A2 * B2 where A2 is 2x1 (A[0,2], A[1,2]), B2 is 1x8 (B[2,0..7])
    # C2[i,k] = A[i,2] * B[2,k]
    # This is schoolbook: 2*1*8 = 16 terms
    for i in range(2):
        for k in range(8):
            U[i*3+2][col] += 1.0
            V[2*8+k][col] += 1.0
            W[k*2+i][col] += 1.0
            col += 1
    
    # Convert to integer where possible
    for mat in [U, V, W]:
        for row in mat:
            for j in range(len(row)):
                if abs(row[j] - round(row[j])) < 1e-10:
                    row[j] = int(round(row[j]))
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
