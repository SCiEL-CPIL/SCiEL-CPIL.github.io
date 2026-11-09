"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START

def _validate_decomposition(U, V, W, n, m, p):
    """
    Validate if U, V, W form a valid decomposition of n x m by m x p matrix multiplication.
    """
    rank = U[0].__len__() if isinstance(U[0], list) else U.shape[1]
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Check all 6^6 = 46656 conditions
    for i in range(n):
        for j in range(p):
            for a in range(n):
                for b in range(m):
                    for c in range(m):
                        for d in range(p):
                            expected = 1.0 if (a == i and d == j and b == c) else 0.0
                            s = 0.0
                            for r in range(rank):
                                s += W_np[j * n + i, r] * U_np[a * m + b, r] * V_np[c * p + d, r]
                            if abs(s - expected) > 1e-9:
                                return False
    return True

def _validate_3x3_terms(terms):
    """
    Validate if the given terms form a valid 3x3 matrix multiplication decomposition.
    """
    n = 3
    rank = len(terms)
    U = [[0] * rank for _ in range(n * n)]
    V = [[0] * rank for _ in range(n * n)]
    W = [[0] * rank for _ in range(n * n)]
    
    for r, term in enumerate(terms):
        for (i, k, coeff) in term['U']:
            U[i * n + k][r] += coeff
        for (k, j, coeff) in term['V']:
            V[k * n + j][r] += coeff
        for (i, j, coeff) in term['W']:
            W[j * n + i][r] += coeff
            
    return _validate_decomposition(U, V, W, n, n, n)

def _construct_rank_189():
    """Fallback: Block-wise Strassen (2x2 blocks in 3x3 grid). Rank 189."""
    n, m, p = DIMENSIONS
    rank = 189
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    strassen_U = [
        [(0, 0, 1), (1, 1, 1)],
        [(1, 0, 1), (1, 1, 1)],
        [(0, 0, 1)],
        [(1, 1, 1)],
        [(0, 0, 1), (0, 1, 1)],
        [(1, 0, 1), (0, 0, -1)],
        [(0, 1, 1), (1, 1, -1)],
    ]
    strassen_V = [
        [(0, 0, 1), (1, 1, 1)],
        [(0, 0, 1)],
        [(0, 1, 1), (1, 1, -1)],
        [(1, 0, 1), (0, 0, -1)],
        [(1, 1, 1)],
        [(0, 0, 1), (0, 1, 1)],
        [(1, 0, 1), (1, 1, 1)],
    ]
    strassen_W = [
        [(0, 0, 1), (1, 1, 1)],
        [(1, 0, 1), (1, 1, -1)],
        [(0, 1, 1), (1, 1, 1)],
        [(0, 0, 1), (1, 0, 1)],
        [(0, 0, -1), (0, 1, 1)],
        [(1, 1, 1)],
        [(0, 0, 1)],
    ]

    col = 0
    for bi in range(3):
        for bk in range(3):
            for bj in range(3):
                for t in range(7):
                    for (li, lk, coeff) in strassen_U[t]:
                        gi = 2 * bi + li
                        gk = 2 * bk + lk
                        U[gi * m + gk][col] += coeff
                    for (lk, lj, coeff) in strassen_V[t]:
                        gk = 2 * bk + lk
                        gj = 2 * bj + lj
                        V[gk * p + gj][col] += coeff
                    for (li, lj, coeff) in strassen_W[t]:
                        gi = 2 * bi + li
                        gj = 2 * bj + lj
                        W[gj * n + gi][col] += coeff
                    col += 1
    return {"U": U, "V": V, "W": W}

def _construct_rank_184():
    """
    Attempt rank-184 decomposition using 3x3 blocks (2x2 grid) with rank-23 3x3 scheme.
    """
    n, m, p = DIMENSIONS
    # Rank 23 decomposition for 3x3
    # Source: Inspiration 2
    terms_3x3 = [
        {'U': [(0,0,1), (2,2,1)], 'V': [(0,0,1), (2,2,1)], 'W': [(0,0,1), (2,2,1)]},
        {'U': [(2,0,1), (2,2,1)], 'V': [(0,0,1)], 'W': [(2,0,1), (2,2,-1)]},
        {'U': [(0,0,1)], 'V': [(0,1,1), (2,1,-1)], 'W': [(0,1,1), (2,1,1)]},
        {'U': [(2,2,1)], 'V': [(2,0,1), (0,0,-1)], 'W': [(0,2,1), (2,0,1)]},
        {'U': [(0,0,1), (0,1,1)], 'V': [(2,2,1)], 'W': [(0,0,-1), (0,1,1)]},
        {'U': [(2,0,1), (0,0,-1)], 'V': [(0,0,1), (0,1,1)], 'W': [(2,1,1)]},
        {'U': [(0,1,1), (2,2,-1)], 'V': [(2,0,1), (2,2,1)], 'W': [(0,0,1)]},
        {'U': [(0,2,1), (2,2,1)], 'V': [(0,2,1)], 'W': [(0,2,1), (2,2,1)]},
        {'U': [(2,2,1)], 'V': [(0,2,1), (2,2,-1)], 'W': [(0,2,-1), (2,2,1)]},
        {'U': [(0,0,1), (0,2,1)], 'V': [(2,0,1)], 'W': [(0,0,1), (0,2,1)]},
        {'U': [(2,0,1)], 'V': [(0,2,1), (2,0,-1)], 'W': [(2,0,1), (2,2,1)]},
        {'U': [(0,2,1), (2,0,-1)], 'V': [(2,0,1), (2,2,1)], 'W': [(0,2,1)]},
        {'U': [(1,1,1)], 'V': [(1,1,1)], 'W': [(1,1,1)]},
        {'U': [(0,1,1), (1,1,1)], 'V': [(1,0,1)], 'W': [(0,1,1), (1,1,-1)]},
        {'U': [(1,1,1)], 'V': [(0,1,1), (1,1,-1)], 'W': [(0,1,1), (1,1,1)]},
        {'U': [(1,0,1)], 'V': [(1,1,1), (0,1,-1)], 'W': [(1,0,1), (1,1,1)]},
        {'U': [(0,1,1), (1,0,1)], 'V': [(1,1,1)], 'W': [(0,1,-1), (1,0,1)]},
        {'U': [(1,1,1), (1,0,-1)], 'V': [(0,1,1), (1,0,1)], 'W': [(1,1,1)]},
        {'U': [(1,0,1), (1,1,1)], 'V': [(1,2,1)], 'W': [(1,2,1), (1,1,-1)]},
        {'U': [(1,1,1)], 'V': [(1,2,1), (0,2,-1)], 'W': [(0,2,1), (1,2,1)]},
        {'U': [(0,2,1), (1,1,1)], 'V': [(1,2,1)], 'W': [(0,2,-1), (1,2,1)]},
        {'U': [(1,2,1)], 'V': [(1,1,1), (0,2,-1)], 'W': [(1,2,1), (1,1,1)]},
        {'U': [(0,2,1), (1,2,-1)], 'V': [(1,1,1), (1,2,1)], 'W': [(0,2,1)]},
    ]
    
    rank = 8 * 23 # 2x2 grid of 3x3 blocks, each rank 23
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # 2x2 grid of 3x3 blocks
    for bi in range(2): # Row block of A/C
        for bk in range(2): # Col block of A / Row block of B
            for bj in range(2): # Col block of B / Col block of C
                for t in range(23):
                    term = terms_3x3[t]
                    for (li, lk, coeff) in term['U']:
                        gi = 3 * bi + li
                        gk = 3 * bk + lk
                        U[gi * m + gk][col] += coeff
                    for (lk, lj, coeff) in term['V']:
                        gk = 3 * bk + lk
                        gj = 3 * bj + lj
                        V[gk * p + gj][col] += coeff
                    for (li, lj, coeff) in term['W']:
                        gi = 3 * bi + li
                        gj = 3 * bj + lj
                        W[gj * n + gi][col] += coeff
                    col += 1
                    
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Try rank 184 first
    try:
        # First validate the 3x3 terms to avoid expensive 6x6 validation if they are wrong
        # Extract terms from _construct_rank_184 is hard, so we just try the full construction
        # but we can add a time check
        res_184 = _construct_rank_184()
        if _validate_decomposition(res_184['U'], res_184['V'], res_184['W'], n, m, p):
            return res_184
    except Exception:
        pass
        
    # Fallback to rank 189
    return _construct_rank_189()

# EVOLVE-BLOCK-END
