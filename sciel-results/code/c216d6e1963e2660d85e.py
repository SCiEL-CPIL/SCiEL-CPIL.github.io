"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    # Block decomposition: C (3x4) = A (3x4) * B (4x4)
    # Split A into A_top (1x4) and A_bot (2x4), B into 4 quadrants (2x2)
    # A_bot * B uses 7 Strassen products
    # A_top * B_{11} (1x2 * 2x2) needs 4 products
    # A_top * B_{12} (1x2 * 2x2) needs 4 products
    # A_top * B_{21} (1x2 * 2x2) needs 4 products
    # A_top * B_{22} (1x2 * 2x2) needs 4 products
    # Total rank = 7 + 16 = 23
    
    n, m, p = DIMENSIONS  # 3, 4, 4
    rank = 23
    U = [[0.0] * rank for _ in range(n * m)]  # 12 x rank
    V = [[0.0] * rank for _ in range(m * p)]  # 16 x rank
    W = [[0.0] * rank for _ in range(p * n)]  # 12 x rank
    
    def set_uv_w(u_idx, v_idx, w_idx, col, val):
        U[u_idx][col] += val
        V[v_idx][col] += val
        W[w_idx][col] += val
    
    col = 0
    
    # Strassen 7 products for A_bot (rows 1,2 of A) * B (all 4x4)
    # A_bot is 2x4, B is 4x4
    # Split A_bot into columns: A_L = A_bot[:, 0:2], A_R = A_bot[:, 2:4]
    # Split B into 2x2 blocks: B_{11}, B_{12}, B_{21}, B_{22}
    # 
    # P1 = A_L * B_{11}
    # P2 = A_R * B_{21}
    # P3 = A_L * B_{12}
    # P4 = A_R * B_{22}
    # P5 = (A_L + A_R) * B_{11}  -- wait, this isn't right for rectangular
    # 
    # Actually for 2x4 * 4x4, we can think of it as computing 2 outputs.
    # Row 1 of C = A[1,:] * B, Row 2 of C = A[2,:] * B
    # 
    # Let's use a different approach. For 2x4 * 4x4:
    # C_bot = A_bot * B
    # 
    # We can decompose:
    # C_bot[:, 0:2] = A_bot[:, 0:2] * B_{11} + A_bot[:, 2:4] * B_{21}
    # C_bot[:, 2:4] = A_bot[:, 0:2] * B_{12} + A_bot[:, 2:4] * B_{22}
    # 
    # Each term is 2x2 * 2x2 = 2x2
    # 
    # Using Strassen for each 2x2*2x2:
    # For M = X * Y where X, Y are 2x2:
    # M[0,0] = P1 - P2 - P4 + P5
    # M[0,1] = P3 + P4
    # M[1,0] = P1 + P2
    # M[1,1] = P1 - P3 + P5
    # Where:
    # P1 = X[0,0] * Y[0,0]
    # P2 = X[1,0] * Y[0,0]
    # P3 = X[0,1] * Y[1,1]
    # P4 = X[1,1] * Y[1,1]
    # P5 = (X[0,0] + X[1,1]) * (Y[0,0] + Y[1,1])
    # 
    # But these are scalar products in the matrix multiplication context.
    # In tensor decomposition, each "product" corresponds to a rank-1 term.
    
    # Let me think differently. The tensor for 2x4 * 4x4 has rank at most 2*4*4 = 32.
    # We want to use Strassen-like decomposition.
    
    # Actually, let's use a simpler approach. 
    # For 2x4 * 4x4, we can view it as:
    # C[i,k] = sum_j A[i,j] * B[j,k] for i=0,1 (relative to A_bot)
    # 
    # We can use the fact that 2x4*4x4 can be decomposed into:
    # - 4 products of form (2x2)*(2x2) for the four quadrant pairs
    # 
    # But that's still 4 * 4 = 16 rank terms if we do schoolbook on each 2x2*2x2.
    # 
    # Let me try a different strategy. Use the known result that 
    # rank of matrix multiplication tensor [3,4,4] is at most 44 (from the hint).
    
    # Let me just do a careful block decomposition:
    # C (3x4) = A (3x4) * B (4x4)
    # 
    # Row 0 of C: A[0,:] * B
    # Row 1 of C: A[1,:] * B  
    # Row 2 of C: A[2,:] * B
    # 
    # For rows 1,2 of A (A_bot, 2x4) times B (4x4):
    # This is a 2x4 * 4x4 problem.
    # 
    # I'll use the following decomposition for 2x4 * 4x4:
    # Think of it as two separate 1x4 * 4x4 problems (for each row of A_bot).
    # Each 1x4 * 4x4 has rank at most 16, so total 32 for A_bot * B.
    # Plus row 0: 1x4 * 4x4 has rank 16.
    # Total: 48. That's schoolbook.
    
    # To get below 48, I need to share structure.
    # 
    # Let me try: split B into left (cols 0,1) and right (cols 2,3).
    # C[:, 0:2] = A * B[:, 0:2]  (3x4 * 4x2)
    # C[:, 2:4] = A * B[:, 2:4]  (3x4 * 4x2)
    # 
    # Each 3x4 * 4x2 has rank at most 24, so total 48.
    
    # Hmm. Let me try the Strassen-based approach more carefully.
    # 
    # For A_bot (2x4) * B (4x4):
    # Let A_L = A_bot[:, 0:2] (2x2), A_R = A_bot[:, 2:4] (2x2)
    # B_{11} = B[0:2, 0:2], B_{12} = B[0:2, 2:4]
    # B_{21} = B[2:4, 0:2], B_{22} = B[2:4, 2:4]
    # 
    # C_bot[:, 0:2] = A_L * B_{11} + A_R * B_{21}
    # C_bot[:, 2:4] = A_L * B_{12} + A_R * B_{22}
    # 
    # Now, A_L * B_{11} is 2x2 * 2x2. Using schoolbook, that's 4 rank-1 terms.
    # So C_bot[:, 0:2] needs 4 + 4 = 8 rank-1 terms.
    # Similarly C_bot[:, 2:4] needs 8 rank-1 terms.
    # Total for A_bot * B: 16 rank-1 terms.
    # 
    # For row 0 of C: A[0,:] * B (1x4 * 4x4)
    # C[0, 0:2] = A[0, 0:2] * B_{11} + A[0, 2:4] * B_{21}
    # C[0, 2:4] = A[0, 0:2] * B_{12} + A[0, 2:4] * B_{22}
    # Each 1x2 * 2x2 needs 4 rank-1 terms.
    # So row 0 needs 4 + 4 + 4 + 4 = 16 rank-1 terms.
    # 
    # Total: 16 + 16 = 32. That's better than 48!
    # 
    # Wait, but 1x2 * 2x2: (1x2) * (2x2) = 1x2.
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    # That's 4 scalar products, so 4 rank-1 terms. Correct.
    # 
    # And 2x2 * 2x2: 4 scalar products each, so 4 rank-1 terms per product.
    # 
    # So total rank = 16 (for A_bot * B) + 16 (for A_top * B) = 32.
    # 
    # Let me verify: n*m*p = 3*4*4 = 48 for schoolbook.
    # Our decomposition: 32. That should be valid.
    
    # Now let me implement this carefully.
    # 
    # U has shape (n*m, rank) = (12, 32)
    # V has shape (m*p, rank) = (16, 32)  
    # W has shape (p*n, rank) = (12, 32)
    # 
    # Indexing:
    # U[i*m + j] for i in 0..2, j in 0..3
    # V[j*p + k] for j in 0..3, k in 0..3
    # W[k*n + i] for k in 0..3, i in 0..2
    
    rank = 32
    U = [[0.0] * rank for _ in range(12)]
    V = [[0.0] * rank for _ in range(16)]
    W = [[0.0] * rank for _ in range(12)]
    
    col = 0
    
    # Process row 0 of A (A_top)
    # C[0, 0:2] = A[0, 0:2] * B_{11} + A[0, 2:4] * B_{21}
    # C[0, 2:4] = A[0, 0:2] * B_{12} + A[0, 2:4] * B_{22}
    
    # A[0, 0:2] * B_{11}: 1x2 * 2x2
    # Terms:
    # (0,0): A[0,0] * B[0,0] -> C[0,0]
    # (0,1): A[0,0] * B[0,1] -> C[0,1]
    # (1,0): A[0,1] * B[1,0] -> C[0,0]
    # (1,1): A[0,1] * B[1,1] -> C[0,1]
    
    # Term: A[0,0] * B[0,0] -> contributes to C[0,0]
    u_idx = 0 * 4 + 0  # i=0, j=0
    v_idx = 0 * 4 + 0  # j=0, k=0
    w_idx = 0 * 3 + 0  # k=0, i=0
    U[u_idx][col] = 1
    V[v_idx][col] = 1
    W[w_idx][col] = 1
    col += 1
    
    # Term: A[0,0] * B[0,1] -> contributes to C[0,1]
    u_idx = 0 * 4 + 0
    v_idx = 0 * 4 + 1
    w_idx = 1 * 3 + 0
    U[u_idx][col] = 1
    V[v_idx][col] = 1
    W[w_idx][col] = 1
    col += 1
    
    # Term: A[0,1] * B[1,0] -> contributes to C[0,0]
    u_idx = 0 * 4 + 1
    v_idx = 1 * 4 + 0
    w_idx = 0 * 3 + 0
    U[u_idx][col] = 1
    V[v_idx][col] = 1
    W[w_idx][col] = 1
    col += 1
    
    # Term: A[0,1] * B[1,1] -> contributes to C[0,1]
    u_idx = 0 * 4 + 1
    v_idx = 1 * 4 + 1
    w_idx = 1 * 3 + 0
    U[u_idx][col] = 1
    V[v_idx][col] = 1
    W[w_idx][col] = 1
    col += 1
    
    # A[0, 2:4] * B_{21}: 1x2 * 2x2
    # B_{21} = B[2:4, 0:2]
    # Term: A[0,2] * B[2,0] -> C[0,0]
    u_idx = 0 * 4 + 2
    v_idx = 2 * 4 + 0
    w_idx = 0 * 3 + 0
    U[u_idx][col] = 1
    V[v_idx][col] = 1
    W[w_idx][col] = 1
    col += 1
    
    # Term: A[0,2] * B[2,1] -> C[0,1]
    u_idx = 0 * 4 + 2
    v_idx = 2 * 4 + 1
    w_idx = 1 * 3 + 0
    U[u_idx][col] = 1
    V[v_idx][col] = 1
    W[w_idx][col] = 1
    col += 1
    
    # Term: A[0,3] * B[3,0] -> C[0,0]
    u_idx = 0 * 4 + 3
    v_idx = 3 * 4 + 0
    w_idx = 0 * 3 + 0
    U[u_idx][col] = 1
    V[v_idx][col] = 1
    W[w_idx][col] = 1
    col += 1
    
    # Term: A[0,3] * B[3,1] -> C[0,1]
    u_idx = 0 * 4 + 3
    v_idx = 3 * 4 + 1
    w_idx = 1 * 3 + 0
    U[u_idx][col] = 1
    V[v_idx][col] = 1
    W[w_idx][col] = 1
    col += 1
    
    # A[0, 0:2] * B_{12}: 1x2 * 2x2
    # B_{12} = B[0:2, 2:4]
    # Term: A[0,0] * B[0,2] -> C[0,2]
    u_idx = 0 * 4 + 0
    v_idx = 0 * 4 + 2
    w_idx = 2 * 3 + 0
    U[u_idx][col] = 1
    V[v_idx][col] = 1
    W[w_idx][col] = 1
    col += 1
    
    # Term: A[0,0] * B[0,3] -> C[0,3]
    u_idx = 0 * 4 + 0
    v_idx = 0 * 4 + 3
    w_idx = 3 * 3 + 0
    U[u_idx][col] = 1
    V[v_idx][col] = 1
    W[w_idx][col] = 1
    col += 1
    
    # Term: A[0,1] * B[1,2] -> C[0,2]
    u_idx = 0 * 4 + 1
    v_idx = 1 * 4 + 2
    w_idx = 2 * 3 + 0
    U[u_idx][col] = 1
    V[v_idx][col] = 1
    W[w_idx][col] = 1
    col += 1
    
    # Term: A[0,1] * B[1,3] -> C[0,3]
    u_idx = 0 * 4 + 1
    v_idx = 1 * 4 + 3
    w_idx = 3 * 3 + 0
    U[u_idx][col] = 1
    V[v_idx][col] = 1
    W[w_idx][col] = 1
    col += 1
    
    # A[0, 2:4] * B_{22}: 1x2 * 2x2
    # B_{22} = B[2:4, 2:4]
    # Term: A[0,2] * B[2,2] -> C[0,2]
    u_idx = 0 * 4 + 2
    v_idx = 2 * 4 + 2
    w_idx = 2 * 3 + 0
    U[u_idx][col] = 1
    V[v_idx][col] = 1
    W[w_idx][col] = 1
    col += 1
    
    # Term: A[0,2] * B[2,3] -> C[0,3]
    u_idx = 0 * 4 + 2
    v_idx = 2 * 4 + 3
    w_idx = 3 * 3 + 0
    U[u_idx][col] = 1
    V[v_idx][col] = 1
    W[w_idx][col] = 1
    col += 1
    
    # Term: A[0,3] * B[3,2] -> C[0,2]
    u_idx = 0 * 4 + 3
    v_idx = 3 * 4 + 2
    w_idx = 2 * 3 + 0
    U[u_idx][col] = 1
    V[v_idx][col] = 1
    W[w_idx][col] = 1
    col += 1
    
    # Term: A[0,3] * B[3,3] -> C[0,3]
    u_idx = 0 * 4 + 3
    v_idx = 3 * 4 + 3
    w_idx = 3 * 3 + 0
    U[u_idx][col] = 1
    V[v_idx][col] = 1
    W[w_idx][col] = 1
    col += 1
    
    # Now A_bot (rows 1,2 of A) * B
    # C[1, 0:2] = A[1, 0:2] * B_{11} + A[1, 2:4] * B_{21}
    # C[1, 2:4] = A[1, 0:2] * B_{12} + A[1, 2:4] * B_{22}
    # C[2, 0:2] = A[2, 0:2] * B_{11} + A[2, 2:4] * B_{21}
    # C[2, 2:4] = A[2, 0:2] * B_{12} + A[2, 2:4] * B_{22}
    
    # For each of these, we have 1x2 * 2x2 = 4 terms.
    # Total: 8 * 4 = 32 terms. But we already used 16, so 32 - 16 = 16 more.
    # Total rank = 32.
    
    # Let me just do all remaining terms systematically.
    # For i in [1, 2] (rows of A in A_bot):
    #   For each quadrant of B:
    #     4 terms
    
    for i in [1, 2]:
        # A[i, 0:2] * B_{11}
        for j in [0, 1]:
            for k in [0, 1]:
                u_idx = i * 4 + j
                v_idx = j * 4 + k  # j is row of B, k is col
                # Wait, B_{11} = B[0:2, 0:2], so rows 0,1 and cols 0,1
                # The term is A[i,j] * B[j,k] where j in 0,1 and k in 0,1
                # But j here is the column of A and row of B, which is the same index
                # So v_idx = j * 4 + k
                w_idx = k * 3 + i
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1
        
        # A[i, 2:4] * B_{21}
        for j in [2, 3]:
            for k in [0, 1]:
                u_idx = i * 4 + j
                v_idx = j * 4 + k
                w_idx = k * 3 + i
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1
        
        # A[i, 0:2] * B_{12}
        for j in [0, 1]:
            for k in [2, 3]:
                u_idx = i * 4 + j
                v_idx = j * 4 + k
                w_idx = k * 3 + i
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1
        
        # A[i, 2:4] * B_{22}
        for j in [2, 3]:
            for k in [2, 3]:
                u_idx = i * 4 + j
                v_idx = j * 4 + k
                w_idx = k * 3 + i
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1
    
    # Total columns should be 16 + 2*16 = 48? No wait.
    # I did 16 terms for row 0, and for rows 1,2 I did 16 terms each = 32.
    # Total = 48. That's just schoolbook!
    
    # I made an error. Let me reconsider.
    # 
    # The issue is that I'm not actually sharing anything. 
    # Each 1x2 * 2x2 product is done independently.
    # 
    # To get below 48, I need to actually use Strassen's method or some other sharing.
    
    # Let me just return the schoolbook decomposition with rank 48, 
    # but with the correct indexing.
    
    # Actually, the current code above has rank=32 but I'm filling 48 columns.
    # That's a bug. Let me fix this.
    
    # For now, let me just do schoolbook correctly with rank 48.
    rank = 48
    U = [[0.0] * rank for _ in range(12)]
    V = [[0.0] * rank for _ in range(16)]
    W = [[0.0] * rank for _ in range(12)]
    
    col = 0
    for i in range(3):
        for j in range(4):
            for k in range(4):
                u_idx = i * 4 + j
                v_idx = j * 4 + k
                w_idx = k * 3 + i
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

