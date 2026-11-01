"""Matrix multiplication 2,2,11 - Strassen-based rank reduction."""
DIMENSIONS = [2, 2, 11]

def _verify(U, V, W):
    """Verify that the decomposition exactly reproduces the multiplication tensor."""
    import numpy as np
    n, m, p = DIMENSIONS  # 2, 2, 11
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Build the tensor T of shape (n*m, m*p, p*n)
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Compute the tensor from the decomposition
    T_decomp = np.einsum('ar,br,cr->abc', U, V, W)
    
    # Use exact integer comparison since all values are integers
    diff = T_decomp - T
    return np.max(np.abs(diff)) < 1e-6

def _build_strassen_2x2_2x11():
    """Build a rank-37 decomposition using Strassen for 5 blocks of 2 columns + schoolbook for last column."""
    n, m, p = 2, 2, 11
    
    # Strassen terms: each term t has U_t (A-space, len 4) and V_t (local B-space, len 4)
    # U index: i*2+j for A[i,j]
    # V local index: j*2+k' for B[j,k']
    strassen_U = [
        [1, 0, 0, 1],   # P1: A00 + A11
        [0, 0, 1, 1],   # P2: A10 + A11
        [1, 0, 0, 0],   # P3: A00
        [0, 0, 0, 1],   # P4: A11
        [1, 1, 0, 0],   # P5: A00 + A01
        [-1, 0, 1, 0],  # P6: A10 - A00
        [0, 1, -1, 0],  # P7: A01 - A11
    ]
    strassen_V = [
        [1, 0, 0, 1],   # P1: B00 + B11
        [1, 0, 0, 0],   # P2: B00
        [0, 0, 1, -1],  # P3: B10 - B11
        [-1, 0, 0, 1],  # P4: B01 - B00
        [0, 0, 0, 1],   # P5: B11
        [1, 0, 0, 1],   # P6: B00 + B01
        [0, 0, 1, 1],   # P7: B10 + B11
    ]
    
    # Reconstruction: recon[i][k'][t] = coefficient of P_t in C[i,k']
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    recon = [
        # i=0
        [
            [1, 0, 0, 1, -1, 0, 1],  # C[0,0]
            [0, 0, 1, 0, 1, 0, 0],   # C[0,1]
        ],
        # i=1
        [
            [0, 1, 0, 1, 0, 0, 0],   # C[1,0]
            [1, -1, 1, 0, 0, 1, 0],  # C[1,1]
        ],
    ]
    
    num_strassen_blocks = 5
    total_rank = num_strassen_blocks * 7 + 2
    
    U = [[0] * total_rank for _ in range(4)]   # 4 = n*m
    V = [[0] * total_rank for _ in range(22)]  # 22 = m*p
    W = [[0] * total_rank for _ in range(22)]  # 22 = p*n
    
    col = 0
    for b in range(num_strassen_blocks):
        for t in range(7):
            # U: A-space
            for a_idx in range(4):
                U[a_idx][col] = strassen_U[t][a_idx]
            # V: B-space, block b covers B columns 2b, 2b+1
            for j in range(2):
                for kp in range(2):
                    local_idx = j * 2 + kp
                    global_idx = j * 11 + (2 * b + kp)
                    V[global_idx][col] = strassen_V[t][local_idx]
            # W: C-space, block b covers C columns 2b, 2b+1
            for i in range(2):
                for kp in range(2):
                    global_idx = (2 * b + kp) * 2 + i
                    W[global_idx][col] = recon[i][kp][t]
            col += 1
    
    # Schoolbook for column 10
    # C[i,10] = A[i,0]*B[0,10] + A[i,1]*B[1,10]
    # Term 1: A[i,0]*B[0,10]
    U[0][col] = 1  # A[0,0]
    U[2][col] = 1  # A[1,0]
    V[10][col] = 1  # B[0,10] at index 0*11+10=10
    W[20][col] = 1  # C[0,10] at index 10*2+0=20
    W[21][col] = 1  # C[1,10] at index 10*2+1=21
    col += 1
    
    # Term 2: A[i,1]*B[1,10]
    U[1][col] = 1  # A[0,1]
    U[3][col] = 1  # A[1,1]
    V[21][col] = 1  # B[1,10] at index 1*11+10=21
    W[20][col] = 1  # C[0,10] at index 20
    W[21][col] = 1  # C[1,10] at index 21
    col += 1
    
    return {'U': U, 'V': V, 'W': W}

def construct():
    import time
    start_time = time.time()
    DEADLINE = 140  # seconds, leave margin
    
    # Try rank-37 Strassen construction
    result = _build_strassen_2x2_2x11()
    if _verify(result['U'], result['V'], result['W']):
        return result
    
    # Fall back to schoolbook rank 44
    n, m, p = DIMENSIONS
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
    return {'U': U, 'V': V, 'W': W}
