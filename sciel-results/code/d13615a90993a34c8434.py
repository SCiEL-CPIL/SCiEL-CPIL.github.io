"""Matrix multiplication 2,2,11 - Strassen-based rank reduction."""
DIMENSIONS = [2, 2, 11]

def _verify(U, V, W):
    """Verify that the decomposition exactly reproduces the multiplication tensor."""
    import numpy as np
    n, m, p = DIMENSIONS  # 2, 2, 11
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Build the tensor T of shape (n*m, m*p, p*n)
    # T[a, b, c] = 1 if the term A[i,j]*B[j,k] contributes to C[i,k]
    # a = i*m + j, b = j*p + k, c = k*n + i
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Compute the tensor from the decomposition
    # T_decomp[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    T_decomp = np.einsum('ar,br,cr->abc', U, V, W)
    
    return np.allclose(T, T_decomp, atol=1e-9)

def _build_strassen_2x2_2x11():
    """Build a rank-37 decomposition using Strassen for 5 blocks of 2 columns + schoolbook for last column."""
    n, m, p = 2, 2, 11
    # Strassen terms for 2x2 x 2x2 -> 2x2
    # U vectors (length 4, for A[i,j]): index = i*2+j
    # V vectors (length 4, for B[j,k'] local): index = j*2+k'
    # W vectors (length 4, for C[i,k'] local): index = i*2+k'
    
    # Standard Strassen:
    # P1 = (A00+A11)(B00+B11) -> contributes to C
    # P2 = (A10+A11)B00
    # P3 = A00(B10-B11)
    # P4 = A11(B01-B00)
    # P5 = (A00+A01)B11
    # P6 = (A10-A00)(B00+B01)
    # P7 = (A01-A11)(B10+B11)
    
    # The output C[i,k'] = sum of contributions from each P_t
    # Standard Strassen reconstruction:
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    # U_t: coefficient vector for A (length 4, index i*2+j)
    # V_t: coefficient vector for B block (length 4, index j*2+k')
    # W_t: coefficient vector for C block (length 4, index i*2+k')
    
    strassen_terms = [
        # (U, V, W) for each of 7 terms
        # P1 = (A00+A11)(B00+B11)
        ([1,0,0,1], [1,0,0,1], [0,0,0,0]),  # placeholder, W determined by reconstruction
    ]
    
    # Actually, let me define each product term and then the reconstruction.
    # Each term t has a U_t (in A-space) and V_t (in B-space).
    # The product P_t = (U_t · A)(V_t · B_t)
    # Then C = sum_t coeff_t(t) * P_t where coeff depends on which C element.
    
    # Let me define the 7 terms:
    terms = []
    # P1: (A00+A11) * (B00+B11)
    terms.append({'U': [1,0,0,1], 'V': [1,0,0,1]})
    # P2: (A10+A11) * B00
    terms.append({'U': [0,0,1,1], 'V': [1,0,0,0]})
    # P3: A00 * (B10-B11)
    terms.append({'U': [1,0,0,0], 'V': [0,0,1,-1]})
    # P4: A11 * (B01-B00)
    terms.append({'U': [0,0,0,1], 'V': [-1,0,0,1]})
    # P5: (A00+A01) * B11
    terms.append({'U': [1,1,0,0], 'V': [0,0,0,1]})
    # P6: (A10-A00) * (B00+B01)
    terms.append({'U': [-1,0,1,0], 'V': [1,0,0,1]})
    # P7: (A01-A11) * (B10+B11)
    terms.append({'U': [0,1,-1,0], 'V': [0,0,1,1]})
    
    # Reconstruction coefficients: C[i,k'] = sum_t recon[i,k',t] * P_t
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    recon = {
        (0,0): [1, 0, 0, 1, -1, 0, 1],
        (0,1): [0, 0, 1, 0, 1, 0, 0],
        (1,0): [0, 1, 0, 1, 0, 0, 0],
        (1,1): [1, -1, 1, 0, 0, 1, 0],
    }
    
    # Now build the full U, V, W matrices
    # 5 Strassen blocks (columns 0-9) + 1 schoolbook column (column 10)
    # Each Strassen block contributes 7 rank terms
    # Schoolbook column contributes 2 rank terms
    
    num_strassen_blocks = 5
    total_rank = num_strassen_blocks * 7 + 2
    
    U = [[0] * total_rank for _ in range(4)]   # 4 = n*m
    V = [[0] * total_rank for _ in range(22)]  # 22 = m*p
    W = [[0] * total_rank for _ in range(22)]  # 22 = p*n
    
    col = 0
    for b in range(num_strassen_blocks):
        for t in range(7):
            term = terms[t]
            # U: A-space, 4 entries
            for idx in range(4):
                U[idx][col] = term['U'][idx]
            # V: B-space, 22 entries. Block b uses columns 2b, 2b+1
            # Local index j*2+k' maps to global j*11 + (2b + k')
            for j in range(2):
                for kp in range(2):
                    local_idx = j * 2 + kp
                    global_idx = j * 11 + (2 * b + kp)
                    V[global_idx][col] = term['V'][local_idx]
            # W: C-space, 22 entries. Block b uses output columns 2b, 2b+1
            # Local index i*2+k' maps to global (2b+k')*2 + i
            for i in range(2):
                for kp in range(2):
                    local_idx = i * 2 + kp
                    global_idx = (2 * b + kp) * 2 + i
                    W[global_idx][col] = recon[(i, kp)][t]
            col += 1
    
    # Schoolbook for column 10
    # C[i,10] = A[i,0]*B[0,10] + A[i,1]*B[1,10]
    # Term 1: A[i,0]*B[0,10] for i=0,1 -> rank-1 tensor
    #   U: [1,0,1,0] (A00, A10)
    #   V: [0,0,...,1,0,...] (B[0,10] at index 0*11+10=10)
    #   W: [1,0,...,1,0,...] (C[0,10] at 10*2+0=20, C[1,10] at 10*2+1=21)
    U[0][col] = 1
    U[2][col] = 1
    V[10][col] = 1
    W[20][col] = 1
    W[21][col] = 1
    col += 1
    
    # Term 2: A[i,1]*B[1,10] for i=0,1
    #   U: [0,1,0,1] (A01, A11)
    #   V: B[1,10] at index 1*11+10=21
    #   W: C[0,10] at 20, C[1,10] at 21
    U[1][col] = 1
    U[3][col] = 1
    V[21][col] = 1
    W[20][col] = 1
    W[21][col] = 1
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
