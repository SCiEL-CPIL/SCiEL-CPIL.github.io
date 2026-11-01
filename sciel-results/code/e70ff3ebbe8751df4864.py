import time

DIMENSIONS = [2, 2, 8]
N, M, P = DIMENSIONS
NM = N * M   # 4
MP = M * P   # 16
PN = P * N   # 16

def verify(U, V, W, n=N, m=M, p=P):
    """Verify that U, V, W form an exact decomposition. Returns True/False."""
    nm, mp, pn = n*m, m*p, p*n
    if not U or not V or not W:
        return False
    rank = len(U[0]) if U else 0
    if len(U) != nm or len(V) != mp or len(W) != pn:
        return False
    for row in U:
        if len(row) != rank:
            return False
    for row in V:
        if len(row) != rank:
            return False
    for row in W:
        if len(row) != rank:
            return False
    
    # Build a set of valid (a,b,c) indices
    # a = i*m + j
    # b = j*p + k
    # c = k*n + i
    valid = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                valid.add((i*m+j, j*p+k, k*n+i))
                
    # Check all entries
    for a in range(nm):
        for b in range(mp):
            for c in range(pn):
                s = 0
                for r in range(rank):
                    s += U[a][r] * V[b][r] * W[c][r]
                expected = 1 if (a, b, c) in valid else 0
                if s != expected:
                    return False
    return True

def schoolbook():
    """Return the schoolbook rank-32 decomposition."""
    n, m, p = N, M, P
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W

def strassen_decomposition_228():
    """
    Construct a rank-28 decomposition for M_{2,2,8} using 4 blocks of Strassen M_{2,2,2}.
    The 2x8 matrix is split into four 2x2 blocks.
    """
    n, m, p = N, M, P
    
    # Strassen terms for M_{2,2,2}
    # Indices for U (A matrix 2x2): 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # Indices for V (B matrix 2x2): 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # Indices for W (C matrix 2x2): 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    # The 7 Strassen products:
    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    # M2 = (A10+A11)B00       -> C10 += M2
    # M3 = A00(B10-B11)       -> C01 += M3
    # M4 = A11(B01-B00)       -> C11 += M4 (Wait, C11 = M1 + M4 - M2 - M3? Let's check standard Strassen)
    
    # Standard Strassen for C = AB:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B10 - B11)
    # M4 = A11(B01 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Let's map these to vectors u, v, w of length 4.
    # u vector corresponds to A coefficients
    # v vector corresponds to B coefficients
    # w vector corresponds to C coefficients
    
    terms = [
        # M1
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]), # C00, C11
        # M2
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]), # C10
        # M3
        ([1, 0, 0, 0], [0, 1, 1, 0], [0, 1, 0, 0]), # C01 (B10-B11 -> B indices 2,3? No. B10 is idx 2, B11 is idx 3? 
        # Wait. B indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1).
        # M3 = A00 * (B10 - B11). A00 is idx 0. B10 is idx 2, B11 is idx 3.
        # So v = [0, 0, 1, -1].
        # C01 is idx 1. w = [0, 1, 0, 0].
        # Correction for M3:
        # ([1, 0, 0, 0], [0, 0, 1, -1], [0, 1, 0, 0]),
        # M4
        # M4 = A11 * (B01 - B00). A11 is idx 3. B01 is idx 1, B00 is idx 0.
        # v = [-1, 1, 0, 0].
        # C11 is idx 3. w = [0, 0, 0, 1].
        # Correction for M4:
        # ([0, 0, 0, 1], [-1, 1, 0, 0], [0, 0, 0, 1]),
        # M5
        # M5 = (A00 + A01) * B11. A00 idx 0, A01 idx 1. B11 idx 3.
        # u = [1, 1, 0, 0]. v = [0, 0, 0, 1].
        # C00 idx 0, C01 idx 1.
        # w = [1, 1, 0, 0].
        # Correction for M5:
        # ([1, 1, 0, 0], [0, 0, 0, 1], [1, 1, 0, 0]),
        # M6
        # M6 = (A10 - A00) * (B00 + B01). A10 idx 2, A00 idx 0. B00 idx 0, B01 idx 1.
        # u = [-1, 0, 1, 0]. v = [1, 1, 0, 0].
        # C11 idx 3.
        # w = [0, 0, 0, 1].
        # Correction for M6:
        # ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        # M7
        # M7 = (A01 - A11) * (B10 + B11). A01 idx 1, A11 idx 3. B10 idx 2, B11 idx 3.
        # u = [0, 1, 0, -1]. v = [0, 0, 1, 1].
        # C00 idx 0.
        # w = [1, 0, 0, 0].
        # Correction for M7:
        # ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])
    ]
    
    # Re-defining terms with corrections
    terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]), # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]), # M2
        ([1, 0, 0, 0], [0, 0, 1, -1], [0, 1, 0, 0]), # M3
        ([0, 0, 0, 1], [-1, 1, 0, 0], [0, 0, 0, 1]), # M4
        ([1, 1, 0, 0], [0, 0, 0, 1], [1, 1, 0, 0]), # M5
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]), # M6
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])  # M7
    ]

    rank = 28
    U = [[0] * rank for _ in range(NM)]
    V = [[0] * rank for _ in range(MP)]
    W = [[0] * rank for _ in range(PN)]
    
    num_blocks = p // 2 # 4 blocks
    
    for block_idx in range(num_blocks):
        for term_idx in range(7):
            col = block_idx * 7 + term_idx
            u, v, w = terms[term_idx]
            
            # Fill U: indices are i*m + j. 
            # Block structure doesn't affect A, so U is same for all blocks.
            for i in range(4):
                U[i][col] = u[i]
                
            # Fill V: indices are j*p + k.
            # Block block_idx uses k = 2*block_idx and 2*block_idx+1.
            # Local v indices 0,1,2,3 correspond to:
            # 0: j=0, k_local=0 -> j=0, k_global=2*block_idx
            # 1: j=0, k_local=1 -> j=0, k_global=2*block_idx+1
            # 2: j=1, k_local=0 -> j=1, k_global=2*block_idx
            # 3: j=1, k_local=1 -> j=1, k_global=2*block_idx+1
            for l in range(4):
                if l == 0:
                    b_idx = 0 * p + (2 * block_idx)
                elif l == 1:
                    b_idx = 0 * p + (2 * block_idx + 1)
                elif l == 2:
                    b_idx = 1 * p + (2 * block_idx)
                elif l == 3:
                    b_idx = 1 * p + (2 * block_idx + 1)
                V[b_idx][col] = v[l]
                
            # Fill W: indices are k*n + i.
            # Block block_idx uses k = 2*block_idx and 2*block_idx+1.
            # Local w indices 0,1,2,3 correspond to:
            # 0: i=0, k_local=0 -> i=0, k_global=2*block_idx
            # 1: i=0, k_local=1 -> i=0, k_global=2*block_idx+1
            # 2: i=1, k_local=0 -> i=1, k_global=2*block_idx
            # 3: i=1, k_local=1 -> i=1, k_global=2*block_idx+1
            for l in range(4):
                if l == 0:
                    c_idx = (2 * block_idx) * n + 0
                elif l == 1:
                    c_idx = (2 * block_idx + 1) * n + 0
                elif l == 2:
                    c_idx = (2 * block_idx) * n + 1
                elif l == 3:
                    c_idx = (2 * block_idx + 1) * n + 1
                W[c_idx][col] = w[l]
                    
    return U, V, W

def construct():
    """Entry point: return the best decomposition found."""
    deadline = time.time() + 140  # safety margin before 150s kill
    
    # First try the known rank-28 Strassen decomposition
    try:
        U, V, W = strassen_decomposition_228()
        if verify(U, V, W):
            return {"U": U, "V": V, "W": W}
    except Exception:
        pass
    
    # Fallback to schoolbook
    U, V, W = schoolbook()
    return {"U": U, "V": V, "W": W}
