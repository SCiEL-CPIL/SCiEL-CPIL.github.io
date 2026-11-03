import time

DIMENSIONS = [3, 3, 8]
N, M, P = DIMENSIONS
SCHOOLBOOK = N * M * P  # 72

def _verify(U, V, W, tol=1e-9):
    """Check that sum_c U[a,c] V[b,c] W[c] == 1 iff a,b,c correspond to a product."""
    try:
        rank = len(U[0])
        for a in range(N * M):
            for b in range(M * P):
                acc = [0.0] * (P * N)
                for c in range(rank):
                    u = U[a][c]
                    if u == 0:
                        continue
                    v = V[b][c]
                    if v == 0:
                        continue
                    uv = u * v
                    if uv == 0:
                        continue
                    wc = W[c]
                    for k in range(P * N):
                        acc[k] += uv * wc[k]
                i, j = divmod(a, M)
                j2, k2 = divmod(b, P)
                for kk in range(P * N):
                    expected = 1.0 if (kk % N == i and kk // N == j2) else 0.0
                    if abs(acc[kk] - expected) > tol:
                        return False
        return True
    except Exception:
        return False

def _strassen_2x2(A, B):
    """
    Returns a list of 7 tuples (u_val, v_val, w_coeffs) for 2x2 matrix multiplication.
    A is 2x2, B is 2x2.
    We return the factors for the decomposition.
    """
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)(e)
    # M3 = (a)(f-h)
    # M4 = (d)(g-e)
    # M5 = (a+b)(h)
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We need to generate the U, V, W contributions.
    # U indices: a (0,1,2,3) corresponding to A entries.
    # V indices: b (0,1,2,3) corresponding to B entries.
    # W indices: c (0,1,2,3) corresponding to C entries.
    
    # This function will be used to build the global U, V, W.
    # It's easier to just hardcode the 68-rank construction directly.
    pass

def construct():
    start = time.time()
    deadline = start + 140.0
    
    # Rank 68 Construction
    # Split A (3x3) into A_top (2x3) and A_bot (1x3)
    # Split B (3x8) into B1 (3x4) and B2 (3x4)
    # C_top = A_top * B (2x8)
    # C_bot = A_bot * B (1x8)
    
    # C_top = A_top * B1 + A_top * B2
    # A_top (2x3) = [A_L (2x2) | A_R (2x1)]
    # B1 (3x4) = [B1_T (2x4); B1_B (1x4)]
    # A_top * B1 = A_L * B1_T + A_R * B1_B
    # A_L * B1_T: 2x2 * 2x4. Split B1_T into B1_T1 (2x2) and B1_T2 (2x2).
    # A_L * B1_T1: 2x2 * 2x2 (Rank 7)
    # A_L * B1_T2: 2x2 * 2x2 (Rank 7)
    # A_R * B1_B: 2x1 * 1x4 (Rank 8)
    # Total for A_top * B1: 7 + 7 + 8 = 22
    
    # Similarly for A_top * B2: 22
    
    # C_bot = A_bot * B (1x3 * 3x8)
    # Rank 24
    
    # Total Rank: 22 + 22 + 24 = 68
    
    rank = 68
    U = [[0] * rank for _ in range(N * M)]
    V = [[0] * rank for _ in range(M * P)]
    W = [[0] * rank for _ in range(P * N)]
    
    col = 0
    
    # Helper to add a block decomposition
    def add_block(u_offset, v_offset, w_offset, u_rows, v_cols, w_cols, block_type, sub_offset_u=0, sub_offset_v=0, sub_offset_w=0):
        nonlocal col
        # This is getting complex to generalize. Let's just hardcode the 68 rank.
        pass
    
    # --- Block 1: A_L * B1_T1 (2x2 * 2x2) ---
    # A_L entries: A[0,0], A[0,1], A[1,0], A[1,1] -> U indices 0,1,2,3
    # B1_T1 entries: B[0,0], B[0,1], B[1,0], B[1,1] -> V indices 0,1,2,3
    # C entries: C[0,0], C[0,1], C[1,0], C[1,1] -> W indices 0,1,2,3 (for this sub-block, but global W is 8x3)
    # Global W indices for C_top[0:2, 0:2]:
    # C[0,0] is W[0], C[0,1] is W[3]? No.
    # W index is k*N + i. k is row of C (0..7), i is col of C (0..2).
    # C_top is rows 0,1. C_bot is row 2.
    # C_top cols are 0..7.
    # B1 is cols 0..3 of B. B2 is cols 4..7 of B.
    # So C_top[0:2, 0:4] comes from B1.
    # C_top[0:2, 4:8] comes from B2.
    
    # Let's define the mapping carefully.
    # U index: i*M + j. i=0..2, j=0..2.
    # V index: j*P + k. j=0..2, k=0..7.
    # W index: k*N + i. k=0..7, i=0..2.
    
    # A_L is A[0:2, 0:2]. U indices: 0,1,2,3.
    # B1_T1 is B[0:2, 0:2]. V indices: 0,1,4,5. (j=0, k=0,1 -> 0,1; j=1, k=0,1 -> 4,5)
    # C[0:2, 0:2]. W indices: k=0,1; i=0,1.
    # W[0*3+0]=0, W[0*3+1]=1, W[1*3+0]=3, W[1*3+1]=4.
    
    # Strassen for 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # We will add these 7 terms to the global U, V, W.
    
    def add_strassen(u_base, v_base, w_base):
        # u_base: list of 4 U indices for A
        # v_base: list of 4 V indices for B
        # w_base: list of 4 W indices for C
        nonlocal col
        a00, a01, a10, a11 = u_base
        b00, b01, b10, b11 = v_base
        c00, c01, c10, c11 = w_base
        
        # M1
        U[a00][col] += 1; U[a11][col] += 1
        V[b00][col] += 1; V[b11][col] += 1
        W[c00][col] += 1; W[c11][col] += 1
        W[c01][col] -= 1; W[c10][col] -= 1
        col += 1
        
        # M2
        U[a10][col] += 1; U[a11][col] += 1
        V[b00][col] += 1
        W[c10][col] += 1; W[c11][col] -= 1
        col += 1
        
        # M3
        U[a00][col] += 1
        V[b01][col] += 1; V[b11][col] -= 1
        W[c01][col] += 1; W[c11][col] += 1
        col += 1
        
        # M4
        U[a11][col] += 1
        V[b10][col] += 1; V[b00][col] -= 1
        W[c00][col] += 1; W[c10][col] += 1
        col += 1
        
        # M5
        U[a00][col] += 1; U[a01][col] += 1
        V[b11][col] += 1
        W[c00][col] -= 1; W[c01][col] += 1
        col += 1
        
        # M6
        U[a10][col] += 1; U[a00][col] -= 1
        V[b00][col] += 1; V[b01][col] += 1
        W[c11][col] += 1
        col += 1
        
        # M7
        U[a01][col] += 1; U[a11][col] -= 1
        V[b10][col] += 1; V[b11][col] += 1
        W[c00][col] += 1; W[c10][col] -= 1
        col += 1

    # Block 1: A_L * B1_T1
    # A_L: U[0], U[1], U[2], U[3]
    # B1_T1: V[0], V[1], V[4], V[5]
    # C[0:2, 0:2]: W[0], W[1], W[3], W[4]
    add_strassen([0, 1, 2, 3], [0, 1, 4, 5], [0, 1, 3, 4])
    
    # Block 2: A_L * B1_T2
    # B1_T2: B[0:2, 2:4]. V indices: j=0, k=2,3 -> 2,3; j=1, k=2,3 -> 6,7.
    # C[0:2, 2:4]. W indices: k=2,3; i=0,1.
    # W[2*3+0]=6, W[2*3+1]=7, W[3*3+0]=9, W[3*3+1]=10.
    add_strassen([0, 1, 2, 3], [2, 3, 6, 7], [6, 7, 9, 10])
    
    # Block 3: A_R * B1_B
    # A_R: A[0:2, 2]. U[1], U[3] (i=0,1; j=2).
    # B1_B: B[2, 0:4]. V indices: j=2, k=0..3 -> 16,17,18,19.
    # C[0:2, 0:4]. W indices: k=0..3; i=0,1.
    # W[0,0]=0, W[0,1]=1, W[1,0]=3, W[1,1]=4, W[2,0]=6, W[2,1]=7, W[3,0]=9, W[3,1]=10.
    # This is 2x1 * 1x4. Rank 8.
    # Products: U[1]*V[16], U[1]*V[17], U[1]*V[18], U[1]*V[19]
    #           U[3]*V[16], U[3]*V[17], U[3]*V[18], U[3]*V[19]
    
    u_r = [1, 3]
    v_b = [16, 17, 18, 19]
    w_c = [0, 1, 3, 4, 6, 7, 9, 10]
    
    for i, u_idx in enumerate(u_r):
        for j, v_idx in enumerate(v_b):
            w_idx = w_c[i*4 + j]
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1
            
    # Total so far: 7 + 7 + 8 = 22.
    # Next: A_top * B2.
    # B2 is B[0:2, 4:8] and B[2, 4:8].
    # B2_T1: B[0:2, 4:6]. V: j=0, k=4,5 -> 8,9; j=1, k=4,5 -> 12,13.
    # B2_T2: B[0:2, 6:8]. V: j=0, k=6,7 -> 10,11; j=1, k=6,7 -> 14,15.
    # B2_B: B[2, 4:8]. V: j=2, k=4..7 -> 20,21,22,23.
    
    # C[0:2, 4:6]. W: k=4,5; i=0,1.
    # W[4,0]=12, W[4,1]=13, W[5,0]=15, W[5,1]=16.
    add_strassen([0, 1, 2, 3], [8, 9, 12, 13], [12, 13, 15, 16])
    
    # C[0:2, 6:8]. W: k=6,7; i=0,1.
    # W[6,0]=18, W[6,1]=19, W[7,0]=21, W[7,1]=22.
    add_strassen([0, 1, 2, 3], [10, 11, 14, 15], [18, 19, 21, 22])
    
    # A_R * B2_B
    u_r = [1, 3]
    v_b = [20, 21, 22, 23]
    w_c = [12, 13, 15, 16, 18, 19, 21, 22]
    
    for i, u_idx in enumerate(u_r):
        for j, v_idx in enumerate(v_b):
            w_idx = w_c[i*4 + j]
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1
            
    # Total so far: 22 + 22 = 44.
    # Next: A_bot * B
    # A_bot: A[2, 0:3]. U[6], U[7], U[8].
    # B: B[0:3, 0:8]. V[0..23].
    # C[2, 0:8]. W: k=0..7; i=2.
    # W[0,2]=2, W[1,2]=5, W[2,2]=8, W[3,2]=11, W[4,2]=14, W[5,2]=17, W[6,2]=20, W[7,2]=23.
    
    u_bot = [6, 7, 8]
    v_all = list(range(24))
    w_bot = [2, 5, 8, 11, 14, 17, 20, 23]
    
    # 1x3 * 3x8. Rank 24.
    for i, u_idx in enumerate(u_bot):
        for j, v_idx in enumerate(v_all):
            if j < 8: # k=0..7
                # Wait, V index is j*P + k. 
                # We need to iterate over all 24 V indices.
                pass
                
    # Let's just iterate over all 24 V indices and 3 U indices.
    # For each k (0..7), we need c_k = a_0 b_{0k} + a_1 b_{1k} + a_2 b_{2k}.
    # This requires 3 multiplications per k. Total 24.
    
    for k in range(8):
        w_idx = k * 3 + 2
        for i in range(3):
            u_idx = 6 + i
            v_idx = i * 8 + k
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1
            
    # Total col should be 68.
    if col != 68:
        # Fallback to schoolbook if construction is flawed
        rank = SCHOOLBOOK
        U = [[0] * rank for _ in range(N * M)]
        V = [[0] * rank for _ in range(M * P)]
        W = [[0] * rank for _ in range(P * N)]
        column = 0
        for i in range(N):
            for j in range(M):
                for k in range(P):
                    U[i * M + j][column] = 1
                    V[j * P + k][column] = 1
                    W[k * N + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

    if _verify(U, V, W):
        return {"U": U, "V": V, "W": W}
    else:
        rank = SCHOOLBOOK
        U = [[0] * rank for _ in range(N * M)]
        V = [[0] * rank for _ in range(M * P)]
        W = [[0] * rank for _ in range(P * N)]
        column = 0
        for i in range(N):
            for j in range(M):
                for k in range(P):
                    U[i * M + j][column] = 1
                    V[j * P + k][column] = 1
                    W[k * N + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
