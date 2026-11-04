import time
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin before 120s limit

    n, m, p = DIMENSIONS
    num_A = n * m  # 6
    num_B = m * p  # 18
    num_C = p * n  # 12

    # Build the target tensor for 2x3x6 multiplication
    # T[a, b, c] = 1 if (a,b,c) corresponds to valid (i,j,k)
    # a = i*m + j, b = j*p + k, c = k*n + i

    valid_triples = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                valid_triples.add((a, b, c))

    # Schoolbook solution: rank = n*m*p = 36
    best_rank = n * m * p
    rank = best_rank
    U = np.zeros((num_A, rank))
    V = np.zeros((num_B, rank))
    W = np.zeros((num_C, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U[a, col] = 1
                V[b, col] = 1
                W[c, col] = 1
                col += 1
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

    # Strassen-based block decomposition for rank 33
    # Split B (3x6) into B_left (3x3) and B_right (3x3)
    # For each 2x3 x 3x3 -> 2x3 block, use a structured rank-16 decomposition
    # Total rank: 16 + 16 + 1 = 33 (the +1 handles a shared term)
    
    # Let's try to build a rank-33 solution using block structure
    # Strategy: Use Strassen on 2x2x2 subblocks within each 2x3x3
    
    # For 2x3 x 3x3 -> 2x3:
    # Split A (2x3) into A1 (2x2) and A2 (2x1)
    # Split B (3x3) into B1 (2x3) and B2 (1x3)
    # C = A1*B1 + A2*B2
    # A1*B1: 2x2 x 2x3 -> 2x3, use Strassen-like: rank 11
    # A2*B2: 2x1 x 1x3 -> 2x3, rank 6
    # Total per block: 17, for 2 blocks: 34
    
    # Let's implement rank-34 first, then try to shave off 1
    
    def build_strassen_2x2x2():
        """Build Strassen decomposition for 2x2 x 2x2 -> 2x2, rank 7"""
        # Returns U, V, W matrices of shape (4, 7) each
        U_s = np.zeros((4, 7))
        V_s = np.zeros((4, 7))
        W_s = np.zeros((4, 7))
        
        # Indexing: a = i*2+j for A, b = j*2+k for B, c = k*2+i for C
        # A is 2x2: indices 0,1 (row 0), 2,3 (row 1)
        # B is 2x2: indices 0,1 (col 0), 2,3 (col 1) 
        # Wait, B is indexed as b = j*p+k where p=2 for 2x2
        # So B[0,0]=0, B[0,1]=1, B[1,0]=2, B[1,1]=3
        # C is 2x2: c = k*n+i where n=2
        # So C[0,0]=0, C[0,1]=1, C[1,0]=2, C[1,1]=3
        
        # Strassen:
        # M1 = (a00+a11)(b00+b11)
        # M2 = (a10+a11)b00
        # M3 = a00(b10-b11)
        # M4 = a11(b00-b10)
        # M5 = (a00+a01)b11
        # M6 = (a01-a10)(b00+b10)
        # M7 = (a10-a01)(b11+b01)
        
        # c00 = M1 - M4 + M6 - M7
        # c01 = M3 + M5 + M7
        # c10 = M2 + M4 - M6
        # c11 = M1 - M3 + M2 - M5
        
        # U: rows are A indices (0,1,2,3) = (a00, a01, a10, a11)
        # V: rows are B indices (0,1,2,3) = (b00, b01, b10, b11)
        # W: rows are C indices (0,1,2,3) = (c00, c01, c10, c11)
        
        # M1 = (a00+a11)(b00+b11)
        U_s[0, 0] = 1  # a00
        U_s[3, 0] = 1  # a11
        V_s[0, 0] = 1  # b00
        V_s[3, 0] = 1  # b11
        W_s[0, 0] = 1  # c00 += M1
        W_s[3, 0] = 1  # c11 += M1
        
        # M2 = (a10+a11)b00
        U_s[2, 1] = 1  # a10
        U_s[3, 1] = 1  # a11
        V_s[0, 1] = 1  # b00
        W_s[2, 1] = 1  # c10 += M2
        W_s[3, 1] = -1  # c11 -= M2
        
        # M3 = a00(b10-b11)
        U_s[0, 2] = 1  # a00
        V_s[2, 2] = 1  # b10
        V_s[3, 2] = -1  # b11
        W_s[1, 2] = 1  # c01 += M3
        W_s[3, 2] = -1  # c11 -= M3
        
        # M4 = a11(b00-b10)
        U_s[3, 3] = 1  # a11
        V_s[0, 3] = 1  # b00
        V_s[2, 3] = -1  # b10
        W_s[0, 3] = -1  # c00 -= M4
        W_s[2, 3] = 1  # c10 += M4
        
        # M5 = (a00+a01)b11
        U_s[0, 4] = 1  # a00
        U_s[1, 4] = 1  # a01
        V_s[3, 4] = 1  # b11
        W_s[1, 4] = 1  # c01 += M5
        W_s[3, 4] = -1  # c11 -= M5
        
        # M6 = (a01-a10)(b00+b10)
        U_s[1, 5] = 1  # a01
        U_s[2, 5] = -1  # a10
        V_s[0, 5] = 1  # b00
        V_s[2, 5] = 1  # b10
        W_s[0, 5] = -1  # c00 -= M6
        W_s[2, 5] = -1  # c10 -= M6
        
        # M7 = (a10-a01)(b11+b01)
        U_s[2, 6] = 1  # a10
        U_s[1, 6] = -1  # a01
        V_s[3, 6] = 1  # b11
        V_s[1, 6] = 1  # b01
        W_s[0, 6] = 1  # c00 += M7
        W_s[1, 6] = 1  # c01 += M7
        
        return U_s, V_s, W_s
    
    def build_2x1x1x3():
        """Build decomposition for 2x1 x 1x3 -> 2x3, rank 6"""
        # A is 2x1: indices 0, 1 (a00, a10)
        # B is 1x3: indices 0, 1, 2 (b00, b01, b02)
        # C is 2x3: c = k*n+i, n=2
        # c00 = a00*b00, c01 = a00*b01, c02 = a00*b02
        # c10 = a10*b00, c11 = a10*b01, c12 = a10*b02
        U_s = np.zeros((2, 6))
        V_s = np.zeros((3, 6))
        W_s = np.zeros((6, 6))
        
        for k in range(3):
            U_s[0, k] = 1
            V_s[k, k] = 1
            W_s[k, k] = 1  # c_{0,k}
            
            U_s[1, 3+k] = 1
            V_s[k, 3+k] = 1
            W_s[3+k, 3+k] = 1  # c_{1,k}
        
        return U_s, V_s, W_s
    
    # Build rank-34 solution
    U_34 = np.zeros((num_A, 34))
    V_34 = np.zeros((num_B, 34))
    W_34 = np.zeros((num_C, 34))
    
    # Block 1: columns 0-2 of B (k=0,1,2)
    # A indices: 0-5 (2x3), B indices: 0-5 (for k=0,1,2: b=j*6+k, j=0,1,2)
    # C indices: 0-5 (for k=0,1,2: c=k*2+i, k=0,1,2, i=0,1)
    
    # Split A into A1 (2x2, cols 0-1) and A2 (2x1, col 2)
    # Split B into B1 (2x3, rows 0-1) and B2 (1x3, row 2)
    
    # A1*B1: 2x2 x 2x3 -> 2x3
    # Further split B1 into B1a (2x2, cols 0-1) and B1b (2x1, col 2)
    # A1*B1a: 2x2 x 2x2 -> 2x2, rank 7 (Strassen)
    # A1*B1b: 2x2 x 2x1 -> 2x1, rank 4
    # A2*B2: 2x1 x 1x3 -> 2x3, rank 6
    # Total for block 1: 7 + 4 + 6 = 17
    
    col_offset = 0
    
    # A1*B1a: 2x2 x 2x2 -> 2x2
    # A1 indices in full A: rows 0-1, cols 0-1 -> A[0,0]=0, A[0,1]=1, A[1,0]=3, A[1,1]=4
    # B1a indices in full B: rows 0-1, cols 0-1 -> B[0,0]=0, B[0,1]=1, B[1,0]=6, B[1,1]=7
    # C indices: k=0,1; i=0,1 -> C[0,0]=0, C[0,1]=1, C[1,0]=2, C[1,1]=3
    
    U_s, V_s, W_s = build_strassen_2x2x2()
    # Map local indices to global
    # Local A: 0,1,2,3 -> Global A: 0,1,3,4 (a00, a01, a10, a11)
    a_map = [0, 1, 3, 4]
    # Local B: 0,1,2,3 -> Global B: 0,1,6,7 (b00, b01, b10, b11)
    b_map = [0, 1, 6, 7]
    # Local C: 0,1,2,3 -> Global C: 0,1,2,3 (c00, c01, c10, c11)
    c_map = [0, 1, 2, 3]
    
    for r in range(7):
        for li, gi in enumerate(a_map):
            U_34[gi, col_offset + r] += U_s[li, r]
        for li, gi in enumerate(b_map):
            V_34[gi, col_offset + r] += V_s[li, r]
        for li, gi in enumerate(c_map):
            W_34[gi, col_offset + r] += W_s[li, r]
    col_offset += 7
    
    # A1*B1b: 2x2 x 2x1 -> 2x1, rank 4
    # A1 indices: same as above: 0,1,3,4
    # B1b indices: rows 0-1, col 2 -> B[0,2]=2, B[1,2]=8
    # C indices: k=2; i=0,1 -> C[2,0]=4, C[2,1]=5
    
    U_2x2x2x1 = np.zeros((4, 4))
    V_2x2x2x1 = np.zeros((2, 4))
    W_2x2x2x1 = np.zeros((2, 4))
    # c0 = a00*b0 + a01*b1
    # c1 = a10*b0 + a11*b1
    U_2x2x2x1[0, 0] = 1  # a00
    V_2x2x2x1[0, 0] = 1  # b0
    W_2x2x2x1[0, 0] = 1  # c0
    
    U_2x2x2x1[1, 1] = 1  # a01
    V_2x2x2x1[1, 1] = 1  # b1
    W_2x2x2x1[0, 1] = 1  # c0 += a01*b1
    
    U_2x2x2x1[2, 2] = 1  # a10
    V_2x2x2x1[0, 2] = 1  # b0
    W_2x2x2x1[1, 2] = 1  # c1 += a10*b0
    
    U_2x2x2x1[3, 3] = 1  # a11
    V_2x2x2x1[1, 3] = 1  # b1
    W_2x2x2x1[1, 3] = 1  # c1 += a11*b1
    
    a_map2 = [0, 1, 3, 4]
    b_map2 = [2, 8]
    c_map2 = [4, 5]
    
    for r in range(4):
        for li, gi in enumerate(a_map2):
            U_34[gi, col_offset + r] += U_2x2x2x1[li, r]
        for li, gi in enumerate(b_map2):
            V_34[gi, col_offset + r] += V_2x2x2x1[li, r]
        for li, gi in enumerate(c_map2):
            W_34[gi, col_offset + r] += W_2x2x2x1[li, r]
    col_offset += 4
    
    # A2*B2: 2x1 x 1x3 -> 2x3, rank 6
    # A2 indices: col 2 -> A[0,2]=2, A[1,2]=5
    # B2 indices: row 2, cols 0-2 -> B[2,0]=12, B[2,1]=13, B[2,2]=14
    # C indices: k=0,1,2; i=0,1 -> C[0,0]=0, C[0,1]=1, C[1,0]=2, C[1,1]=3, C[2,0]=4, C[2,1]=5
    
    U_2x1x1x3 = np.zeros((2, 6))
    V_2x1x1x3 = np.zeros((3, 6))
    W_2x1x1x3 = np.zeros((6, 6))
    for k in range(3):
        U_2x1x1x3[0, k] = 1
        V_2x1x1x3[k, k] = 1
        W_2x1x1x3[k, k] = 1  # c_{0,k}
        
        U_2x1x1x3[1, 3+k] = 1
        V_2x1x1x3[k, 3+k] = 1
        W_2x1x1x3[3+k, 3+k] = 1  # c_{1,k}
    
    a_map3 = [2, 5]
    b_map3 = [12, 13, 14]
    c_map3 = [0, 1, 2, 3, 4, 5]
    
    for r in range(6):
        for li, gi in enumerate(a_map3):
            U_34[gi, col_offset + r] += U_2x1x1x3[li, r]
        for li, gi in enumerate(b_map3):
            V_34[gi, col_offset + r] += V_2x1x1x3[li, r]
        for li, gi in enumerate(c_map3):
            W_34[gi, col_offset + r] += W_2x1x1x3[li, r]
    col_offset += 6
    
    # Block 2: columns 3-5 of B (k=3,4,5)
    # B indices: b=j*6+k, j=0,1,2; k=3,4,5
    # B[0,3]=3, B[0,4]=4, B[0,5]=5
    # B[1,3]=9, B[1,4]=10, B[1,5]=11
    # B[2,3]=15, B[2,4]=16, B[2,5]=17
    # C indices: k=3,4,5; i=0,1 -> C[3,0]=6, C[3,1]=7, C[4,0]=8, C[4,1]=9, C[5,0]=10, C[5,1]=11
    
    # A1*B1a: 2x2 x 2x2 -> 2x2 (cols 3-4 of B)
    U_s, V_s, W_s = build_strassen_2x2x2()
    a_map = [0, 1, 3, 4]
    b_map = [3, 4, 9, 10]  # B[0,3], B[0,4], B[1,3], B[1,4]
    c_map = [6, 7, 8, 9]  # C[3,0], C[3,1], C[4,0], C[4,1]
    
    for r in range(7):
        for li, gi in enumerate(a_map):
            U_34[gi, col_offset + r] += U_s[li, r]
        for li, gi in enumerate(b_map):
            V_34[gi, col_offset + r] += V_s[li, r]
        for li, gi in enumerate(c_map):
            W_34[gi, col_offset + r] += W_s[li, r]
    col_offset += 7
    
    # A1*B1b: 2x2 x 2x1 -> 2x1 (col 5 of B)
    a_map2 = [0, 1, 3, 4]
    b_map2 = [5, 11]  # B[0,5], B[1,5]
    c_map2 = [10, 11]  # C[5,0], C[5,1]
    
    for r in range(4):
        for li, gi in enumerate(a_map2):
            U_34[gi, col_offset + r] += U_2x2x2x1[li, r]
        for li, gi in enumerate(b_map2):
            V_34[gi, col_offset + r] += V_2x2x2x1[li, r]
        for li, gi in enumerate(c_map2):
            W_34[gi, col_offset + r] += W_2x2x2x1[li, r]
    col_offset += 4
    
    # A2*B2: 2x1 x 1x3 -> 2x3 (row 2 of B, cols 3-5)
    a_map3 = [2, 5]
    b_map3 = [15, 16, 17]  # B[2,3], B[2,4], B[2,5]
    c_map3 = [6, 7, 8, 9, 10, 11]
    
    for r in range(6):
        for li, gi in enumerate(a_map3):
            U_34[gi, col_offset + r] += U_2x1x1x3[li, r]
        for li, gi in enumerate(b_map3):
            V_34[gi, col_offset + r] += V_2x1x1x3[li, r]
        for li, gi in enumerate(c_map3):
            W_34[gi, col_offset + r] += W_2x1x1x3[li, r]
    col_offset += 6
    
    # Verify rank-34 solution
    max_err = 0
    for (a, b, c) in valid_triples:
        val = np.sum(U_34[a, :] * V_34[b, :] * W_34[c, :])
        max_err = max(max_err, abs(val - 1.0))
    
    # Also check that non-triples are zero
    for a in range(num_A):
        for b in range(num_B):
            for c in range(num_C):
                if (a, b, c) not in valid_triples:
                    val = np.sum(U_34[a, :] * V_34[b, :] * W_34[c, :])
                    max_err = max(max_err, abs(val))
    
    if max_err < 1e-10:
        best_rank = 34
        best_U, best_V, best_W = U_34.copy(), V_34.copy(), W_34.copy()
    
    # Try to find lower rank using gradient descent on the rank-34 solution as initialization
    for target_rank in [30, 31, 32, 33]:
        if time.time() > deadline - 10:
            break
            
        found = False
        for restart in range(20):
            if time.time() > deadline - 5:
                break
                
            np.random.seed(12345 + target_rank * 100 + restart)
            U_curr = best_U[:, :target_rank].copy() + np.random.randn(num_A, target_rank) * 0.1
            V_curr = best_V[:, :target_rank].copy() + np.random.randn(num_B, target_rank) * 0.1
            W_curr = best_W[:, :target_rank].copy() + np.random.randn(num_C, target_rank) * 0.1
            
            lr = 0.005
            for step in range(500):
                if time.time() > deadline - 2:
                    break
                    
                grad_U = np.zeros_like(U_curr)
                grad_V = np.zeros_like(V_curr)
                grad_W = np.zeros_like(W_curr)
                residual_sum = 0
                
                for (a, b, c) in valid_triples:
                    val = np.sum(U_curr[a, :] * V_curr[b, :] * W_curr[c, :])
                    err = val - 1.0
                    residual_sum += err * err
                    
                    for r in range(target_rank):
                        grad_U[a, r] += err * V_curr[b, r] * W_curr[c, r]
                        grad_V[b, r] += err * U_curr[a, r] * W_curr[c, r]
                        grad_W[c, r] += err * U_curr[a, r] * V_curr[b, r]
                
                U_curr -= lr * grad_U
                V_curr -= lr * grad_V
                W_curr -= lr * grad_W
                lr *= 0.995
                
                if residual_sum < 1e-8:
                    found = True
                    break
            
            if found:
                max_err = 0
                for (a, b, c) in valid_triples:
                    val = np.sum(U_curr[a, :] * V_curr[b, :] * W_curr[c, :])
                    max_err = max(max_err, abs(val - 1.0))
                
                if max_err < 1e-6:
                    best_rank = target_rank
                    best_U, best_V, best_W = U_curr.copy(), V_curr.copy(), W_curr.copy()
                    break
        
        if found:
            break
            
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
