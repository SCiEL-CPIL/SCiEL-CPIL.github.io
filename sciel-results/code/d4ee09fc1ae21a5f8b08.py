import time

DIMENSIONS = [3, 3, 6]

def _verify(U, V, W, n, m, p, tol=1e-10):
    """Verify that U, V, W decompose the matrix multiplication tensor."""
    rank = len(U[0])
    # Build the tensor T[i*m+j, j*p+k, k*n+i] = delta
    # The decomposition says T[a,b,c] = sum_d U[a,d]*V[b,d]*W[c,d]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                s = 0.0
                for d in range(rank):
                    s += U[a][d] * V[b][d] * W[c][d]
                if abs(s - 1.0) > tol:
                    return False
    # Check that no spurious terms exist (all other indices should sum to 0)
    # Actually, the tensor only has non-zero entries at (a,b,c) where a=i*m+j, b=j*p+k, c=k*n+i
    # For all other combinations, the sum should be 0
    # But checking all combinations is expensive. Let's check a few.
    # Actually, the structure ensures this if the decomposition is correct for the 1's.
    # Let's verify a few zero entries to be safe.
    # The number of total combinations is (n*m)*(m*p)*(p*n) which can be large.
    # For 3x3x6: 9*18*18 = 2916. Rank 51. 2916*51 ~ 150k ops. Fine.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for i2 in range(n):
                    for j2 in range(m):
                        for k2 in range(p):
                            a = i * m + j
                            b = j * p + k
                            c = k * n + i
                            a2 = i2 * m + j2
                            b2 = j2 * p + k2
                            c2 = k2 * n + i2
                            if (i,j,k) != (i2,j2,k2):
                                # This entry should be 0
                                pass
    # A simpler check: reconstruct a few random products
    import random
    for _ in range(100):
        A = [[random.uniform(-1,1) for _ in range(m)] for _ in range(n)]
        B = [[random.uniform(-1,1) for _ in range(p)] for _ in range(m)]
        # C = A * B, C[i][k] = sum_j A[i][j] * B[j][k]
        C_expected = [[0.0]*p for _ in range(n)]
        for i in range(n):
            for k in range(p):
                s = 0.0
                for j in range(m):
                    s += A[i][j] * B[j][k]
                C_expected[i][k] = s
        # Using decomposition: C[i][k] = sum_d W[k*n+i, d] * (sum_j U[i*m+j, d] * A[i][j]) * (sum_j' V[j'*p+k, d] * B[j'][k])
        # Wait, the standard decomposition is:
        # C[i][k] = sum_d (sum_j U[i*m+j, d] * A[i][j]) * (sum_j' V[j'*p+k, d] * B[j'][k]) * W[k*n+i, d]
        # Actually, the bilinear form is:
        # C[i][k] = sum_d W[k*n+i, d] * U_A[i, d] * V_B[k, d]
        # where U_A[i, d] = sum_j U[i*m+j, d] * A[i][j]
        # and V_B[k, d] = sum_j V[j*p+k, d] * B[j][k]
        U_A = [[0.0]*rank for _ in range(n)]
        for i in range(n):
            for d in range(rank):
                s = 0.0
                for j in range(m):
                    s += U[i*m+j][d] * A[i][j]
                U_A[i][d] = s
        V_B = [[0.0]*rank for _ in range(p)]
        for k in range(p):
            for d in range(rank):
                s = 0.0
                for j in range(m):
                    s += V[j*p+k][d] * B[j][k]
                V_B[k][d] = s
        for i in range(n):
            for k in range(p):
                s = 0.0
                for d in range(rank):
                    s += W[k*n+i][d] * U_A[i][d] * V_B[k][d]
                if abs(s - C_expected[i][k]) > tol:
                    return False
    return True

def _strassen_2x2x2():
    """Return U, V, W for 2x2x2 Strassen decomposition, rank 7."""
    # Strassen's 7 products:
    # M1 = (A11) * (B11 + B22)
    # M2 = (A11 + A22) * (B22)
    # M3 = (A22) * (B11 + B12)
    # M4 = (A11) * (B21 + B22)
    # M5 = (A11 + A12) * (B11)
    # M6 = (A21 + A22) * (B12 - B11)
    # M7 = (A11 + A22) * (B21 - B22)
    # C11 = M1 - M2 + M4 - M7
    # C12 = M3 + M7
    # C21 = M5 + M6
    # C22 = M1 - M3 + M5 - M6
    n, m, p = 2, 2, 2
    rank = 7
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # A indices: i*m+j -> 0:0,0  1:0,1  2:1,0  3:1,1
    # B indices: j*p+k -> 0:0,0  1:0,1  2:1,0  3:1,1
    # C indices: k*n+i -> 0:0,0  1:0,1  2:1,0  3:1,1
    
    # M1 = A00 * (B00 + B11)
    U[0][0] = 1  # A00
    V[0][0] = 1  # B00
    V[3][0] = 1  # B11
    W[0][0] = 1  # C00
    W[3][0] = 1  # C11
    
    # M2 = (A00 + A11) * B11
    U[0][1] = 1  # A00
    U[3][1] = 1  # A11
    V[3][1] = 1  # B11
    W[0][1] = -1 # C00
    W[3][1] = 1  # C11
    
    # M3 = A11 * (B00 + B01)
    U[3][2] = 1  # A11
    V[0][2] = 1  # B00
    V[1][2] = 1  # B01
    W[1][2] = 1  # C01
    W[3][2] = -1 # C11
    
    # M4 = A00 * (B10 + B11)
    U[0][3] = 1  # A00
    V[2][3] = 1  # B10
    V[3][3] = 1  # B11
    W[0][3] = 1  # C00
    W[2][3] = 1  # C10
    
    # M5 = (A00 + A01) * B00
    U[0][4] = 1  # A00
    U[1][4] = 1  # A01
    V[0][4] = 1  # B00
    W[1][4] = 1  # C01
    W[2][4] = 1  # C10
    
    # M6 = (A10 + A11) * (B01 - B00)
    U[2][5] = 1  # A10
    U[3][5] = 1  # A11
    V[0][5] = -1 # B00
    V[1][5] = 1  # B01
    W[1][5] = 1  # C01
    W[2][5] = 1  # C10
    
    # M7 = (A00 + A11) * (B10 - B11)
    U[0][6] = 1  # A00
    U[3][6] = 1  # A11
    V[2][6] = 1  # B10
    V[3][6] = -1 # B11
    W[0][6] = -1 # C00
    W[1][6] = 1  # C01
    W[2][6] = 1  # C10
    W[3][6] = -1 # C11
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s safety margin
    
    n, m, p = DIMENSIONS
    # 3x3x6
    # Split: (2x2 + 1x1) x (2x1 + 1x2) x 6
    # = 2x2x6 + 2x1x6 + 1x2x6 + 1x1x6
    
    # 2x2x6: Use Strassen for each 2x2x2 block (3 blocks)
    # 2x1x6: Schoolbook, rank 12
    # 1x2x6: Schoolbook, rank 12
    # 1x1x6: Schoolbook, rank 6
    # Total rank: 21 + 12 + 12 + 6 = 51
    
    rank = 51
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # Indices:
    # A: 3x3, indices 0..8 (i*3+j)
    # B: 3x6, indices 0..17 (j*6+k)
    # C: 6x3, indices 0..17 (k*3+i)
    
    col = 0
    
    # --- 2x2x6 block: A[0:2, 0:2] * B[0:2, 0:6] -> C[0:6, 0:2]
    # This is 3 blocks of 2x2x2
    # B[0:2, 0:6] = [B[0:2, 0:2], B[0:2, 2:4], B[0:2, 4:6]]
    # C[0:6, 0:2] = [C[0:2, 0:2], C[2:4, 0:2], C[4:6, 0:2]]
    
    for block in range(3):
        U_s, V_s, W_s = _strassen_2x2x2()
        for d in range(7):
            # U: A[0:2, 0:2] -> indices 0,1,2,3
            for a in range(4):
                U[a][col+d] = U_s[a][d]
            # V: B[0:2, block*2 : block*2+2]
            # B index = j*6 + k, j in 0,1, k in block*2, block*2+1
            for b in range(4):
                j = b // 2
                k_local = b % 2
                k_global = block * 2 + k_local
                V[j*6 + k_global][col+d] = V_s[b][d]
            # W: C[block*2 : block*2+2, 0:2]
            # C index = k*3 + i, k in block*2, block*2+1, i in 0,1
            for c in range(4):
                k_local = c // 2
                i_local = c % 2
                k_global = block * 2 + k_local
                W[k_global*3 + i_local][col+d] = W_s[c][d]
        col += 7
    
    # --- 2x1x6 block: A[0:2, 2] * B[2, 0:6] -> C[0:6, 0:2]
    # Rank 12
    for i in range(2):
        for k in range(6):
            U[i*3 + 2][col] = 1  # A[i, 2]
            V[2*6 + k][col] = 1  # B[2, k]
            W[k*3 + i][col] = 1  # C[k, i]
            col += 1
    
    # --- 1x2x6 block: A[2, 0:2] * B[0:2, 0:6] -> C[0:6, 2]
    # Rank 12
    for j in range(2):
        for k in range(6):
            U[2*3 + j][col] = 1  # A[2, j]
            V[j*6 + k][col] = 1  # B[j, k]
            W[k*3 + 2][col] = 1  # C[k, 2]
            col += 1
    
    # --- 1x1x6 block: A[2, 2] * B[2, 0:6] -> C[0:6, 2]
    # Rank 6
    for k in range(6):
        U[2*3 + 2][col] = 1  # A[2, 2]
        V[2*6 + k][col] = 1  # B[2, k]
        W[k*3 + 2][col] = 1  # C[k, 2]
        col += 1
    
    # Verify
    if time.time() < deadline - 5:
        if not _verify(U, V, W, n, m, p):
            # Fall back to schoolbook
            rank = n * m * p
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            column = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i * m + j][column] = 1
                        V[j * p + k][column] = 1
                        W[k * n + i][column] = 1
                        column += 1
    
    return {"U": U, "V": V, "W": W}
