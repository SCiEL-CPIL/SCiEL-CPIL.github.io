"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

_start_time = time.time()
TIME_LIMIT = 140.0  # Safety margin before 150s kill

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
    
    # Known: 3x3x3 = 23 (Bini et al.)
    if (n, m, p) == (3, 3, 3):
        _KNOWN_RANKS[key] = 23
        return 23
    
    # Try all splits of m into m1 + m2
    best = n * m * p  # schoolbook
    for m1 in range(1, m):
        m2 = m - m1
        r1 = _get_rank(n, m1, p)
        r2 = _get_rank(n, m2, p)
        candidate = r1 + r2
        if candidate < best:
            best = candidate
    
    # Also try splitting n
    for n1 in range(1, n):
        n2 = n - n1
        r1 = _get_rank(n1, m, p)
        r2 = _get_rank(n2, m, p)
        candidate = r1 + r2
        if candidate < best:
            best = candidate
    
    # Also try splitting p
    for p1 in range(1, p):
        p2 = p - p1
        r1 = _get_rank(n, m, p1)
        r2 = _get_rank(n, m, p2)
        candidate = r1 + r2
        if candidate < best:
            best = candidate
    
    _KNOWN_RANKS[key] = best
    return best

def _build_strassen(n, m, p):
    """
    Build U, V, W for M_{n,m,p} using Strassen-based recursion.
    Returns (U, V, W, rank) where U is n×m × rank, V is m×p × rank, W is p×n × rank.
    """
    if time.time() - _start_time > TIME_LIMIT:
        raise TimeoutError("Time limit exceeded")
    
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
    
    # Try to use Strassen if dimensions allow
    if n >= 2 and m >= 2 and p >= 2:
        # Check if we can pad to even dimensions and use Strassen
        # For now, use recursive splitting with Strassen for 2x2x2 base
        pass
    
    # Recursive case: find best split
    best_rank = n * m * p
    best_split = None
    best_split_type = None  # 'm', 'n', or 'p'
    
    # Try splitting m
    for m1 in range(1, m):
        m2 = m - m1
        r1 = _get_rank(n, m1, p)
        r2 = _get_rank(n, m2, p)
        candidate = r1 + r2
        if candidate < best_rank:
            best_rank = candidate
            best_split = (m1, m2)
            best_split_type = 'm'
    
    # Try splitting n
    for n1 in range(1, n):
        n2 = n - n1
        r1 = _get_rank(n1, m, p)
        r2 = _get_rank(n2, m, p)
        candidate = r1 + r2
        if candidate < best_rank:
            best_rank = candidate
            best_split = (n1, n2)
            best_split_type = 'n'
    
    # Try splitting p
    for p1 in range(1, p):
        p2 = p - p1
        r1 = _get_rank(n, m, p1)
        r2 = _get_rank(n, m, p2)
        candidate = r1 + r2
        if candidate < best_rank:
            best_rank = candidate
            best_split = (p1, p2)
            best_split_type = 'p'
    
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
    
    s1, s2 = best_split
    
    if best_split_type == 'm':
        m1, m2 = s1, s2
        U1, V1, W1, r1 = _build_strassen(n, m1, p)
        U2, V2, W2, r2 = _build_strassen(n, m2, p)
        
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
    
    elif best_split_type == 'n':
        n1, n2 = s1, s2
        U1, V1, W1, r1 = _build_strassen(n1, m, p)
        U2, V2, W2, r2 = _build_strassen(n2, m, p)
        
        rank = r1 + r2
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        # Copy U1 into U for first n1 rows of A
        for i in range(n1):
            for j in range(m):
                for c in range(r1):
                    U[i * m + j][c] = U1[i * m + j][c]
        
        # Copy U2 into U for last n2 rows of A, offset by r1
        for i in range(n2):
            for j in range(m):
                for c in range(r2):
                    U[(n1 + i) * m + j][r1 + c] = U2[i * m + j][c]
        
        # Copy V1 and V2 (same B, so sum them)
        for j in range(m):
            for k in range(p):
                for c in range(r1):
                    V[j * p + k][c] = V1[j * p + k][c]
                for c in range(r2):
                    V[j * p + k][r1 + c] = V2[j * p + k][c]
        
        # Copy W1 into W for first n1 rows of C
        for i in range(n1):
            for k in range(p):
                for c in range(r1):
                    W[k * n + i][c] = W1[k * n1 + i][c]
        
        # Copy W2 into W for last n2 rows of C, offset by r1
        for i in range(n2):
            for k in range(p):
                for c in range(r2):
                    W[k * n + (n1 + i)][r1 + c] = W2[k * n2 + i][c]
        
        return U, V, W, rank
    
    else:  # best_split_type == 'p'
        p1, p2 = s1, s2
        U1, V1, W1, r1 = _build_strassen(n, m, p1)
        U2, V2, W2, r2 = _build_strassen(n, m, p2)
        
        rank = r1 + r2
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        # Copy U1 and U2 (same A, so sum them)
        for i in range(n):
            for j in range(m):
                for c in range(r1):
                    U[i * m + j][c] = U1[i * m + j][c]
                for c in range(r2):
                    U[i * m + j][r1 + c] = U2[i * m + j][c]
        
        # Copy V1 into V for first p1 cols of B
        for j in range(m):
            for k in range(p1):
                for c in range(r1):
                    V[j * p + k][c] = V1[j * p1 + k][c]
        
        # Copy V2 into V for last p2 cols of B, offset by r1
        for j in range(m):
            for k in range(p2):
                for c in range(r2):
                    V[j * p + (p1 + k)][r1 + c] = V2[j * p2 + k][c]
        
        # Copy W1 into W for first p1 cols of C
        for i in range(n):
            for k in range(p1):
                for c in range(r1):
                    W[k * n + i][c] = W1[k * n + i][c]
        
        # Copy W2 into W for last p2 cols of C, offset by r1
        for i in range(n):
            for k in range(p2):
                for c in range(r2):
                    W[(p1 + k) * n + i][r1 + c] = W2[k * n + i][c]
        
        return U, V, W, rank

def construct():
    n, m, p = DIMENSIONS
    
    # Build the decomposition
    U, V, W, rank = _build_strassen(n, m, p)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
