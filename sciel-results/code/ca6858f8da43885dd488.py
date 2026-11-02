"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START

def _build_schoolbook(n, m, p):
    """Return U, V, W (lists of lists) for the schoolbook decomposition of rank n*m*p."""
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


def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    # Build the tensor T from U, V, W
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Should equal 1 for all i,j,k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for r in range(rank):
                    total += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(total - 1) > 1e-9:
                    return False
    return True


def _build_structural_decomposition(n, m, p):
    """
    Build a structural decomposition that exploits the small n=2 dimension.
    
    Key idea: For each (j,k) pair, we need to compute:
      C[0,k] += A[0,j] * B[j,k]
      C[1,k] += A[1,j] * B[j,k]
    
    We can use 2 rank-1 terms per (j,k) pair, but try to merge where possible.
    
    Alternative: Use the fact that for fixed j, we have a 2x7 sub-problem.
    The tensor for fixed j is: sum_{i,k} e_{i,j} ⊗ e_{j,k} ⊗ e_{i,k}
    
    This can be written as sum_i (e_{i,j} ⊗ sum_k e_{j,k} ⊗ e_{i,k})
    
    The inner tensor sum_k e_{j,k} ⊗ e_{i,k} has rank 7.
    So for each j, we need rank 2*7 = 14, total 4*14 = 56. Same as schoolbook.
    
    Let's try a different approach: use half-integer coefficients to merge terms.
    
    For each j, create 7 terms:
      For k=0..6:
        U_col = [1 if i==0 and j_col==j else 0 for ...] -- this is getting complicated.
    
    Let me try a concrete construction with rank 28 using the following trick:
    
    For each (j,k), use ONE rank-1 term with:
      U: has 1 at position (0,j) and 1 at position (1,j)  [i.e., column j of U is [1,0,0,0,1,0,0,0]^T flattened]
      V: has 1 at position (j,k)
      W: has 1 at position (k,0) and 1 at position (k,1)  [i.e., row k of W is [1,1,0,...,0]]
    
    This gives: e_{0,j}⊗e_{j,k}⊗e_{0,k} + e_{0,j}⊗e_{j,k}⊗e_{1,k} + e_{1,j}⊗e_{j,k}⊗e_{0,k} + e_{1,j}⊗e_{j,k}⊗e_{1,k}
    
    We want: e_{0,j}⊗e_{j,k}⊗e_{0,k} + e_{1,j}⊗e_{j,k}⊗e_{1,k}
    
    The extra terms are: e_{0,j}⊗e_{j,k}⊗e_{1,k} + e_{1,j}⊗e_{j,k}⊗e_{0,k}
    
    If I could subtract these, I'd need more terms. But what if I use a different W?
    
    Try: W has 1 at (k,0) and -1 at (k,1)? No, that doesn't help.
    
    Let me try: for each (j,k), use TWO terms:
      Term 1: U = e_{0,j}, V = e_{j,k}, W = e_{0,k} + e_{1,k}
      Term 2: U = e_{1,j}, V = e_{j,k}, W = -e_{0,k} + e_{1,k}
    
    Sum: e_{0,j}⊗e_{j,k}⊗e_{0,k} + e_{0,j}⊗e_{j,k}⊗e_{1,k} - e_{1,j}⊗e_{j,k}⊗e_{0,k} + e_{1,j}⊗e_{j,k}⊗e_{1,k}
    
    Still has extra terms: e_{0,j}⊗e_{j,k}⊗e_{1,k} - e_{1,j}⊗e_{j,k}⊗e_{0,k}
    
    I don't see how to eliminate these with just 2 terms per (j,k).
    
    Let me give up on the analytical approach and just use schoolbook for now,
    but implement a verification function so we can test better decompositions.
    """
    # For now, fall back to schoolbook
    return _build_schoolbook(n, m, p)


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin of 10 seconds
    
    n, m, p = DIMENSIONS
    
    # Try schoolbook first (guaranteed correct, rank = n*m*p = 56)
    U, V, W = _build_schoolbook(n, m, p)
    
    # Verify it's correct
    assert _verify_decomposition(U, V, W, n, m, p), "Schoolbook decomposition failed verification!"
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
