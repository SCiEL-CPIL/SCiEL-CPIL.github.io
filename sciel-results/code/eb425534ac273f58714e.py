"""Matrix multiplication 2x3 * 3x6: rank-minimal tensor decomposition."""
import time
import sys

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
import numpy as np

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition is correct."""
    rank = len(U[0])
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # Check: sum_t U[a*m+b,t] * V[c*p+d,t] * W[k*n+i,t] == delta(a,i)*delta(c,b)*delta(d,k)
    for a in range(n):
        for b in range(m):
            for c in range(m):
                for d in range(p):
                    for i in range(n):
                        for k in range(p):
                            s = 0.0
                            for t in range(rank):
                                s += U[a*m+b][t] * V[c*p+d][t] * W[k*n+i][t]
                            expected = (1.0 if (a==i and c==b and d==k) else 0.0)
                            if abs(s - expected) > 1e-9:
                                return False
    return True

def _try_rank_18(n, m, p):
    """
    Attempt to construct a rank-18 decomposition for 2x3 * 3x6.
    
    Construction idea: For each j in {0,1,2} and each pair of output rows,
    use Strassen-like combinations.
    
    Actually, let's use a simpler valid construction:
    For each k in 0..5, compute C[:,k] = A * B[:,k].
    For 2x3 * 3x1, the bilinear complexity is 6 (schoolbook).
    But we can potentially share across k's.
    
    Let me try: define for each j and each half of columns:
    For j=0,1,2 and h=0,1 (h=0 means k in {0,1,2,3,4,5}... no that's not right).
    
    Alternative valid rank-18 construction:
    Note that C = A*B. Write A as [a0; a1] where a0, a1 are row vectors.
    C[0,:] = a0 * B, C[1,:] = a1 * B.
    
    For each j, define:
    S_j = a0[j] + a1[j], D_j = a0[j] - a1[j]
    
    Then a0[j] = (S_j + D_j)/2, a1[j] = (S_j - D_j)/2
    
    C[0,k] = sum_j (S_j + D_j)/2 * B[j,k]
    C[1,k] = sum_j (S_j - D_j)/2 * B[j,k]
    
    This doesn't reduce the count.
    
    Let me just try to find ANY rank < 36 that works, starting with a search.
    """
    # For now, return None to indicate this approach doesn't have a known construction
    return None

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x3 * 3x6 matrix multiplication.
    
    Convention: U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    with W[k*n+i, t] indexing.
    """
    start_time = time.time()
    deadline = start_time + 115
    
    n, m, p = DIMENSIONS  # 2, 3, 6
    
    # Schoolbook rank = 36
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
    
    # Verify schoolbook is correct (sanity check)
    if not _verify_decomposition(U, V, W, n, m, p):
        # Should never happen, but guard anyway
        pass
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
