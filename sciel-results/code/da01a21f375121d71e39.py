"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, rank):
    """Check that U,V,W exactly decompose the matmul tensor."""
    for a in range(n*m):
        i = a // m
        j = a % m
        for b in range(m*p):
            j2 = b // p
            k = b % p
            if j2 != j:
                continue
            for c in range(p*n):
                k2 = c // n
                i2 = c % n
                if k2 != k or i2 != i:
                    expected = 0
                else:
                    expected = 1
                s = 0
                for r in range(rank):
                    s += U[a][r] * V[b][r] * W[c][r]
                if s != expected:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    return U, V, W

def _strassen_2x2x2():
    """Return U,V,W for 2x2x2 Strassen (rank 7)."""
    U = [
        [1, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [1, 1, 0, 0, 0, 1, 0],
        [0, 1, 0, 1, 0, -1, -1],
    ]
    V = [
        [1, 1, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 0, 1],
        [1, 0, 0, 1, 0, 1, 0],
        [0, 0, -1, -1, 1, 0, 1],
    ]
    W = [
        [1, 0, 0, 1, 0, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 0, 0, 0, 0, 1, 0],
    ]
    if not _verify(U, V, W, 2, 2, 2, 7):
        return _schoolbook(2, 2, 2)
    return U, V, W

def _combine_decompositions(parts, n, m, p):
    """Combine a list of (U_part, V_part, W_part, offset_i, offset_j, offset_k, ni, nj, np) 
    into a single decomposition for the full n x m x p problem.
    Each part computes a sub-multiplication at position (offset_i, offset_j, offset_k)
    with dimensions ni x nj x np.
    """
    total_rank = sum(len(U_part) and len(U_part[0]) for U_part, _, _, _, _, _, _, _, _ in parts)
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    for U_part, V_part, W_part, off_i, off_j, off_k, ni, nj, np in parts:
        rank_part = len(U_part[0])
        for i in range(ni):
            for j in range(nj):
                a_full = (off_i + i) * m + (off_j + j)
                a_part = i * nj + j
                for r in range(rank_part):
                    U[a_full][col + r] = U_part[a_part][r]
        for j in range(nj):
            for k in range(np):
                b_full = (off_j + j) * p + (off_k + k)
                b_part = j * np + k
                for r in range(rank_part):
                    V[b_full][col + r] = V_part[b_part][r]
        for k in range(np):
            for i in range(ni):
                c_full = (off_k + k) * n + (off_i + i)
                c_part = k * ni + i
                for r in range(rank_part):
                    W[c_full][col + r] = W_part[c_part][r]
        col += rank_part
    
    return U, V, W

def _strassen_rectangular(ni, nj, np):
    """Try to decompose ni x nj x np using Strassen-like blocks if possible.
    Returns (U, V, W, rank) or None if not applicable."""
    # For 2x2x2, use Strassen
    if ni == 2 and nj == 2 and np == 2:
        U, V, W = _strassen_2x2x2()
        return U, V, W, 7
    # For other cases, fall back to schoolbook
    U, V, W = _schoolbook(ni, nj, np)
    return U, V, W, ni * nj * np

def _decompose_4x5x6():
    """Decompose 4x5x6 using block structure.
    
    Split: 4 = 2+2, 5 = 2+3, 6 = 3+3
    
    A (4x5) = [A11(2x2) A12(2x3)]
               [A21(2x2) A22(2x3)]
    B (5x6) = [B11(2x3) B12(2x3)]
               [B21(3x3) B22(3x3)]
    C (4x6) = [C11(2x3) C12(2x3)]
               [C21(2x3) C22(2x3)]
    
    C11 = A11*B11 + A12*B21  (2x3 = 2x2*2x3 + 2x3*3x3)
    C12 = A11*B12 + A12*B22  (2x3 = 2x2*2x3 + 2x3*3x3)
    C21 = A21*B11 + A22*B21  (2x3 = 2x2*2x3 + 2x3*3x3)
    C22 = A21*B12 + A22*B22  (2x3 = 2x2*2x3 + 2x3*3x3)
    
    This gives 8 sub-multiplications. Total schoolbook rank for these:
    4 * (2*2*3) + 4 * (2*3*3) = 4*12 + 4*18 = 48 + 72 = 120. Same as schoolbook.
    
    Alternative: Use Strassen on the 2x2 blocks where possible.
    The 2x2*2x3 multiplications can be seen as 2x2*2x2 + 2x2*2x1.
    This is getting complex. Let me try a different split.
    
    Try: 4=2+2, 5=5, 6=3+3
    C1 = A * B1 where B1 is 5x3, C2 = A * B2 where B2 is 5x3.
    Each is 4x5x3. Schoolbook: 2 * 4*5*3 = 120. No gain.
    
    Try: 4=4, 5=2+3, 6=6
    C = A1*B1 + A2*B2 where A1 is 4x2, B1 is 2x6, A2 is 4x3, B2 is 3x6.
    Schoolbook: 4*2*6 + 4*3*6 = 48 + 72 = 120. No gain.
    
    The fundamental issue: for rectangular matmul, the schoolbook rank n*m*p is often
    hard to beat without specific structure. Let me try to use the 2x2x2 Strassen
    in a creative way.
    
    New idea: Pad to 4x6x6. Then 4x6x6 can be viewed as two 2x6x6 blocks.
    Each 2x6x6: split 6=3+3. C = A1*B1 + A1*B2 where A1 is 2x6, B1,B2 are 6x3.
    Still schoolbook-like.
    
    Let me try: view 4x5x6 as embedding in 6x6x6 and use known R(6,6,6) bounds?
    That's too complex.
    
    For now, return schoolbook. The improvement will come from a more careful
    search for rank-deficient decompositions.
    """
    return _schoolbook(4, 5, 6)

def construct():
    import time
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_result = {"U": U, "V": V, "W": W}
    
    # Try the block decomposition approach
    if time.time() < deadline - 5:
        U2, V2, W2 = _decompose_4x5x6()
        rank2 = len(U2[0]) if U2 else 0
        if rank2 < best_rank:
            if _verify(U2, V2, W2, n, m, p, rank2):
                best_rank = rank2
                best_result = {"U": U2, "V": V2, "W": W2}
    
    return best_result
# EVOLVE-BLOCK-END
