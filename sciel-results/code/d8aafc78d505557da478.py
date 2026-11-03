"""Matrix multiplication 2x3x8 with rank-44 decomposition using Strassen on 2x2x2 subblocks."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=8
    
    # Split middle dimension m=3 into m1=2 + m2=1
    # A is 2x3, B is 3x8
    # C = A[:, 0:2] @ B[0:2, :] + A[:, 2:3] @ B[2:3, :]
    # First term: 2x2x8, use Strassen on 2x2x2 blocks (4 blocks of 2 columns each)
    # Second term: 2x1x8, schoolbook rank 16
    
    rank = 44
    U = [[0] * rank for _ in range(n * m)]  # 6 x 44
    V = [[0] * rank for _ in range(m * p)]  # 24 x 44
    W = [[0] * rank for _ in range(p * n)]  # 16 x 44
    
    col = 0
    
    # Strassen's 7 multiplications for 2x2x2
    # For matrices A (2x2), B (2x2), C = A*B:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Strassen coefficients:
    # Each M_t is a product of linear forms in A and B
    # A indices: A00=(0,0), A01=(0,1), A10=(1,0), A11=(1,1) in the 2x2 sub-block
    # B indices: B00=(0,0), B01=(0,1), B10=(1,0), B11=(1,1) in the 2x2 sub-block
    
    # For a 2x2x2 Strassen block, the 7 rank-1 terms have:
    # U factors (on A side, 4 entries): 
    #   M1: [1, 0, 0, 1]  -> A00 + A11
    #   M2: [0, 0, 1, 1]  -> A10 + A11
    #   M3: [1, 0, 0, 0]  -> A00
    #   M4: [0, 0, 0, 1]  -> A11
    #   M5: [1, 1, 0, 0]  -> A00 + A01
    #   M6: [-1, 0, 1, 0] -> -A00 + A10
    #   M7: [0, 1, 0, -1] -> A01 - A11
    
    # V factors (on B side, 4 entries):
    #   M1: [1, 1, 1, 1]  -> B00 + B01 + B10 + B11? No wait.
    #   Actually for 2x2: B is 2x2, B00, B01, B10, B11
    #   M1: (B00+B11) -> [1, 0, 0, 1] in B00,B01,B10,B11 order? 
    #   Let me be careful. B is 2x2: B[0,0]=B00, B[0,1]=B01, B[1,0]=B10, B[1,1]=B11
    #   M1: B00+B11 -> [1, 0, 0, 1]
    #   M2: B00 -> [1, 0, 0, 0]
    #   M3: B01-B11 -> [0, 1, 0, -1]
    #   M4: B10-B00 -> [0, 0, 1, -1] wait: B10 - B00 -> [-1, 0, 1, 0]
    #   M5: B11 -> [0, 0, 0, 1]
    #   M6: B00+B01 -> [1, 1, 0, 0]
    #   M7: B10+B11 -> [0, 0, 1, 1]
    
    # W factors (on C side, 4 entries): C00, C01, C10, C11
    #   M1: C00 += 1, C11 += 1 -> [1, 0, 0, 1]
    #   M2: C10 += 1, C11 -= 1 -> [0, 0, 1, -1]
    #   M3: C01 += 1, C11 += 1 -> [0, 1, 0, 1]
    #   M4: C00 += 1, C10 += 1 -> [1, 0, 1, 0]
    #   M5: C00 -= 1, C01 += 1 -> [-1, 1, 0, 0]
    #   M6: C11 += 1 -> [0, 0, 0, 1]
    #   M7: C00 += 1 -> [1, 0, 0, 0]
    
    # Let me double-check Strassen:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # So W factors:
    # M1: [1, 0, 0, 1]  (C00=1, C11=1)
    # M2: [0, 0, 1, -1] (C10=1, C11=-1)
    # M3: [0, 1, 0, 1]  (C01=1, C11=1)
    # M4: [1, 0, 1, 0]  (C00=1, C10=1)
    # M5: [-1, 1, 0, 0] (C00=-1, C01=1)
    # M6: [0, 0, 0, 1]  (C11=1)
    # M7: [1, 0, 0, 0]  (C00=1)
    
    # U factors (A side, 4 entries for 2x2):
    # M1: [1, 0, 0, 1]  (A00+A11)
    # M2: [0, 0, 1, 1]  (A10+A11)
    # M3: [1, 0, 0, 0]  (A00)
    # M4: [0, 0, 0, 1]  (A11)
    # M5: [1, 1, 0, 0]  (A00+A01)
    # M6: [-1, 0, 1, 0] (A10-A00)
    # M7: [0, 1, 0, -1] (A01-A11)
    
    # V factors (B side, 4 entries for 2x2):
    # M1: [1, 0, 0, 1]  (B00+B11)
    # M2: [1, 0, 0, 0]  (B00)
    # M3: [0, 1, 0, -1] (B01-B11)
    # M4: [-1, 0, 1, 0] (B10-B00)
    # M5: [0, 0, 0, 1]  (B11)
    # M6: [1, 1, 0, 0]  (B00+B01)
    # M7: [0, 0, 1, 1]  (B10+B11)
    
    strassen_U = [
        [1, 0, 0, 1],   # M1
        [0, 0, 1, 1],   # M2
        [1, 0, 0, 0],   # M3
        [0, 0, 0, 1],   # M4
        [1, 1, 0, 0],   # M5
        [-1, 0, 1, 0],  # M6
        [0, 1, 0, -1],  # M7
    ]
    strassen_V = [
        [1, 0, 0, 1],   # M1
        [1, 0, 0, 0],   # M2
        [0, 1, 0, -1],  # M3
        [-1, 0, 1, 0],  # M4
        [0, 0, 0, 1],   # M5
        [1, 1, 0, 0],   # M6
        [0, 0, 1, 1],   # M7
    ]
    strassen_W = [
        [1, 0, 0, 1],   # M1: C00+=1, C11+=1
        [0, 0, 1, -1],  # M2: C10+=1, C11-=1
        [0, 1, 0, 1],   # M3: C01+=1, C11+=1
        [1, 0, 1, 0],   # M4: C00+=1, C10+=1
        [-1, 1, 0, 0],  # M5: C00-=1, C01+=1
        [0, 0, 0, 1],   # M6: C11+=1
        [1, 0, 0, 0],   # M7: C00+=1
    ]
    
    # Part 1: 2x2x8 using Strassen on 4 blocks of 2x2x2
    # A_left = A[:, 0:2] (rows 0,1 of A, cols 0,1)
    # B_top = B[0:2, :] (rows 0,1 of B, all 8 cols)
    # For each column pair (2t, 2t+1), use Strassen
    
    for t in range(4):  # 4 column pairs
        k0 = 2 * t  # start column in B
        for s in range(7):  # 7 Strassen multiplications
            # U: A side - A[i, j] where i in {0,1}, j in {0,1}
            # U row index: i*m + j = i*3 + j
            for i in range(2):
                for j in range(2):
                    u_idx = i * m + j  # i*3 + j
                    U[u_idx][col] += strassen_U[s][i * 2 + j]
            
            # V: B side - B[j, k] where j in {0,1}, k in {k0, k0+1}
            # V row index: j*p + k
            for j in range(2):
                for dk in range(2):
                    k = k0 + dk
                    v_idx = j * p + k
                    V[v_idx][col] += strassen_V[s][j * 2 + dk]
            
            # W: C side - C[i, k] where i in {0,1}, k in {k0, k0+1}
            # W row index: k*n + i (trace convention: W[k*n+i, r])
            for i in range(2):
                for dk in range(2):
                    k = k0 + dk
                    w_idx = k * n + i
                    W[w_idx][col] += strassen_W[s][i * 2 + dk]
            
            col += 1
    
    # Part 2: 2x1x8 schoolbook
    # A_right = A[:, 2:3], B_bottom = B[2:3, :]
    # C[i,k] += A[i,2] * B[2,k]
    for i in range(2):
        for k in range(8):
            u_idx = i * m + 2  # i*3 + 2
            v_idx = 2 * p + k  # 2*8 + k
            w_idx = k * n + i  # k*2 + i
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
