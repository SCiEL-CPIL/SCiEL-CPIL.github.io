"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

_start_time = time.time()
TIME_LIMIT = 145.0  # Safety margin before 150s kill

# Known exact ranks for small matrix multiplication tensors
# Key: (n, m, p) -> rank
_KNOWN_RANKS = {}

def _get_rank(n, m, p):
    """Return the known/estimated rank for M_{n,m,p}."""
    key = (n, m, p)
    if key in _KNOWN_RANKS:
        return _KNOWN_RANKS[key]
    
    # Base cases
    if n == 1:
        # C (1×p) = A (1×m) * B (m×p), C[k] = sum_j A[j]*B[j,k]
        # Rank = m * p (each (j,k) pair needs its own term)
        r = m * p
        _KNOWN_RANKS[key] = r
        return r
    if m == 1:
        # C (n×p) = A (n×1) * B (1×p), C[i,k] = A[i]*B[k]
        # Rank = n * p
        r = n * p
        _KNOWN_RANKS[key] = r
        return r
    if p == 1:
        # C (n×1) = A (n×m) * B (m×1), C[i] = sum_j A[i,j]*B[j]
        # Rank = n * m
        r = n * m
        _KNOWN_RANKS[key] = r
        return r
    
    # Known: 2x2x2 = 7 (Strassen)
    if (n, m, p) == (2, 2, 2):
        _KNOWN_RANKS[key] = 7
        return 7
    
    # Try all splits of m into m1 + m2
    best = n * m * p  # schoolbook
    for m1 in range(1, m):
        m2 = m - m1
        r1 = _get_rank(n, m1, p)
        r2 = _get_rank(n, m2, p)
        candidate = r1 + r2
        if candidate < best:
            best = candidate
    
    _KNOWN_RANKS[key] = best
    return best

def _build_decomp(n, m, p):
    """
    Build U, V, W for M_{n,m,p} decomposition.
    Returns (U, V, W, rank) where U is n×m × rank, V is m×p × rank, W is n×p × rank.
    Using the block-splitting recursion.
    """
    if time.time() - _start_time > TIME_LIMIT:
        raise TimeoutError("Time limit exceeded")
    
    key = (n, m, p)
    
    # Base case: schoolbook
    if m == 1:
        rank = n * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for k in range(p):
                U[i * m + 0][col] = 1
                V[0 * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
        return U, V, W, rank
    
    if n == 1:
        rank = m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for j in range(m):
            for k in range(p):
                U[0 * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * 1 + 0][col] = 1
                col += 1
        return U, V, W, rank
    
    if p == 1:
        rank = n * m
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                U[i * m + j][col] = 1
                V[j * p + 0][col] = 1
                W[0 * n + i][col] = 1
                col += 1
        return U, V, W, rank
    
    if (n, m, p) == (2, 2, 2):
        # Strassen's algorithm for 2x2x2
        rank = 7
        U = [[0] * rank for _ in range(4)]  # 2x2 = 4 entries, flatten as A11, A12, A21, A22
        V = [[0] * rank for _ in range(4)]  # B11, B12, B21, B22
        W = [[0] * rank for _ in range(4)]  # C11, C12, C21, C22 (row-major: C11, C12, C21, C22)
        
        # A indices: A11=0, A12=1, A21=2, A22=3
        # B indices: B11=0, B12=1, B21=2, B22=3
        # C indices: C11=0, C12=1, C21=2, C22=3
        
        # M1 = (A11+A22)(B11+B22)
        U[0][0] += 1; U[3][0] += 1
        V[0][0] += 1; V[3][0] += 1
        W[0][0] += 1; W[3][0] += 1  # C11 += M1, C22 += M1
        
        # M2 = (A21+A22)B11
        U[2][1] += 1; U[3][1] += 1
        V[0][1] += 1
        W[2][1] += 1; W[3][1] -= 1  # C21 += M2, C22 -= M2
        
        # M3 = A11(B12-B22)
        U[0][2] += 1
        V[1][2] += 1; V[3][2] -= 1
        W[1][2] += 1; W[3][2] += 1  # C12 += M3, C22 += M3
        
        # M4 = A22(B21-B11)
        U[3][3] += 1
        V[2][3] += 1; V[0][3] -= 1
        W[0][3] += 1; W[2][3] += 1  # C11 += M4, C21 += M4
        
        # M5 = (A11+A12)B22
        U[0][4] += 1; U[1][4] += 1
        V[3][4] += 1
        W[0][4] -= 1; W[1][4] += 1  # C11 -= M5, C12 += M5
        
        # M6 = (A21-A11)(B11+B12)
        U[2][5] += 1; U[0][5] -= 1
        V[0][5] += 1; V[1][5] += 1
        W[3][5] += 1  # C22 += M6
        
        # M7 = (A12-A22)(B21+B22)
        U[1][6] += 1; U[3][6] -= 1
        V[2][6] += 1; V[3][6] += 1
        W[0][6] += 1  # C11 += M7
        
        return U, V, W, rank
    
    # Recursive case: find best split of m
    best_rank = n * m * p
    best_split = None
    for m1 in range(1, m):
        m2 = m - m1
        r1 = _get_rank(n, m1, p)
        r2 = _get_rank(n, m2, p)
        candidate = r1 + r2
        if candidate < best_rank:
            best_rank = candidate
            best_split = (m1, m2)
    
    if best_split is None:
        # No improvement, use schoolbook
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
        return U, V, W, rank
    
    m1, m2 = best_split
    U1, V1, W1, r1 = _build_decomp(n, m1, p)
    U2, V2, W2, r2 = _build_decomp(n, m2, p)
    
    rank = r1 + r2
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Copy U1 into U for first m1 columns of A
    for i in range(n):
        for j in range(m1):
            for c in range(r1):
                U[i * m + j][c] = U1[i * m1 + j][c]
    
    # Copy U2 into U for last m2 columns of A, offset by r1
    for i in range(n):
        for j in range(m2):
            for c in range(r2):
                U[i * m + (m1 + j)][r1 + c] = U2[i * m2 + j][c]
    
    # Copy V1 into V for first m1 rows of B
    for j in range(m1):
        for k in range(p):
            for c in range(r1):
                V[j * p + k][c] = V1[j * p + k][c]
    
    # Copy V2 into V for last m2 rows of B, offset by r1
    for j in range(m2):
        for k in range(p):
            for c in range(r2):
                V[(m1 + j) * p + k][r1 + c] = V2[j * p + k][c]
    
    # Copy W1 and W2 (they both contribute to same C)
    for i in range(n):
        for k in range(p):
            for c in range(r1):
                W[k * n + i][c] = W1[k * n + i][c]
            for c in range(r2):
                W[k * n + i][r1 + c] = W2[k * n + i][c]
    
    return U, V, W, rank

def construct():
    n, m, p = DIMENSIONS
    
    # Build the decomposition
    U, V, W, rank = _build_decomp(n, m, p)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
