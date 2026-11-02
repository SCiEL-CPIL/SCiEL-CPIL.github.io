"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
import time
import itertools
from fractions import Fraction
import random

def _verify(U, V, W, n, m, p):
    """Verify exact tensor decomposition: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all i,j,k"""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if s != 1:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return (U, V, W)

def _try_grouped_decomposition(n, m, p):
    """
    Try a grouped decomposition that exploits the structure of matrix multiplication.
    
    Key insight: C[i,k] = sum_j A[i,j]*B[j,k]
    We can try to group the j-index and use a lower-rank decomposition for each group.
    
    For m=3, we can try to use the fact that a 3-term sum can sometimes be
    computed with fewer than 3 rank-1 terms using Winograd's method.
    
    Winograd's method for convolution:
    To compute y[k] = sum_j x[j]*h[k-j] for a 3-point convolution,
    we can use 4 evaluations instead of 9 (rank 4 instead of 9 for the j-sum).
    
    For our case, for each fixed (i,k), we need sum_j A[i,j]*B[j,k].
    If we can compute this sum with rank r_j < m, we can save rank.
    
    Let's try: for each (i,k), use a rank-4 decomposition of the 3-term sum.
    This would give total rank = n*p*r_j = 3*7*4 = 84, which is worse than schoolbook.
    
    Better: share the rank across all (i,k) pairs.
    
    Alternative: use the "bordered" approach where we add extra dimensions.
    
    For now, let's try a direct search with random initialization.
    """
    n, m, p = 3, 3, 7
    rank_lb = max(n*m, m*p, n*p)  # 21
    best_rank = n * m * p  # 63
    best_decomp = None
    
    # Try a few random initializations for ranks near the lower bound
    random.seed(42)
    
    for r in range(rank_lb, min(rank_lb + 10, best_rank)):
        # Try multiple random initializations
        for attempt in range(50):
            # Random initialization with small values
            U = [[Fraction(random.randint(-2, 2)) for _ in range(r)] for _ in range(n*m)]
            V = [[Fraction(random.randint(-2, 2)) for _ in range(r)] for _ in range(m*p)]
            W = [[Fraction(random.randint(-2, 2)) for _ in range(r)] for _ in range(p*n)]
            
            if _verify(U, V, W, n, m, p):
                best_rank = r
                best_decomp = (U, V, W)
                break
        if best_decomp is not None:
            break
    
    if best_decomp is not None:
        return best_decomp
    return _schoolbook(n, m, p)

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix by a 3x7 matrix.
    The output factor follows the official trace convention W[k*n+i, r].
    
    Returns {'U', 'V', 'W'}: factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)
    """
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin for 120s limit
    
    n, m, p = DIMENSIONS
    
    # Get schoolbook decomposition as fallback
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    best_result = (U_sb, V_sb, W_sb)
    best_rank = n * m * p
    
    # Try grouped decomposition
    if time.time() < deadline - 10:
        result = _try_grouped_decomposition(n, m, p)
        if result is not None:
            U, V, W = result
            r = len(U[0])
            if r < best_rank:
                best_result = (U, V, W)
                best_rank = r
    
    U, V, W = best_result
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
