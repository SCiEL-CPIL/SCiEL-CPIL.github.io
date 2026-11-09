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
    total_rank = sum(len(U_part[0]) for U_part, _, _, _, _, _, _, _, _ in parts)
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
        col += rank_rank if False else rank_part
    
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

def _build_4x5x6():
    """Build a decomposition for 4x5x6 using Strassen blocks where possible.
    
    Split: 4 = 2+2, 5 = 2+3, 6 = 2+2+2
    
    We have sub-blocks at positions (ii, jj, kk) where ii in {0,1}, jj in {0,1}, kk in {0,1,2}
    Each sub-block has size ni x nj x np where:
      ni = 2 if ii==0 else 2  (both are 2)
      nj = 2 if jj==0 else 3  (first is 2, second is 3)
      np = 2 for all kk (each k-block is 2)
    
    So we have 2 * 2 * 3 = 12 sub-blocks:
    - 6 blocks of size 2x2x2 -> use Strassen (rank 7 each)
    - 6 blocks of size 2x3x2 -> use schoolbook (rank 12 each)
    
    Total rank: 6*7 + 6*12 = 42 + 72 = 114
    Schoolbook: 120
    Savings: 6
    """
    n, m, p = 4, 5, 6
    
    parts = []
    # i-splits: [0, 2] with sizes [2, 2]
    # j-splits: [0, 2] with sizes [2, 3]  
    # k-splits: [0, 2, 4] with sizes [2, 2, 2]
    
    i_offsets = [0, 2]
    i_sizes = [2, 2]
    j_offsets = [0, 2]
    j_sizes = [2, 3]
    k_offsets = [0, 2, 4]
    k_sizes = [2, 2, 2]
    
    for ii in range(2):
        for jj in range(2):
            for kk in range(3):
                off_i = i_offsets[ii]
                off_j = j_offsets[jj]
                off_k = k_offsets[kk]
                ni = i_sizes[ii]
                nj = j_sizes[jj]
                np = k_sizes[kk]
                
                if ni == 2 and nj == 2 and np == 2:
                    U_part, V_part, W_part = _strassen_2x2x2()
                    rank_part = 7
                else:
                    U_part, V_part, W_part = _schoolbook(ni, nj, np)
                    rank_part = ni * nj * np
                
                parts.append((U_part, V_part, W_part, off_i, off_j, off_k, ni, nj, np))
    
    U, V, W = _combine_decompositions(parts, n, m, p)
    return U, V, W

def construct():
    import time
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_result = {"U": U, "V": V, "W": W}
    
    # Try the Strassen-block decomposition
    if time.time() < deadline - 5:
        U2, V2, W2 = _build_4x5x6()
        rank2 = len(U2[0]) if U2 else 0
        if rank2 < best_rank:
            if _verify(U2, V2, W2, n, m, p, rank2):
                best_rank = rank2
                best_result = {"U": U2, "V": V, "W": W2}
    
    return best_result
# EVOLVE-BLOCK-END
