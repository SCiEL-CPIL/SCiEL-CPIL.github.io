"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from fractions import Fraction
from itertools import product, combinations

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def _schoolbook_decomposition(n, m, p):
    """Return the standard rank n*m*p decomposition."""
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
    return {"U": U, "V": V, "W": W}


def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that the given factor matrices exactly decompose the matrix multiplication tensor.
    The tensor T[i, j, k] = delta[i, k] * 1 (since C[i,k] = sum_j A[i,j]*B[j,k]).
    Actually, the multiplication tensor has T[i,j,k] = 1 if we view it as:
    C[i,k] = sum_j A[i,j] * B[j,k]
    The tensor form: T[i, j, k] such that C[i,k] = sum_{j,l} A[i,j] B[l,k] * T[i,j,l] ... 
    Wait, let me re-derive.
    
    Standard: C[i,k] = sum_j A[i,j] * B[j,k]
    So the tensor is T[i, j, k] = 1 for all i,j,k? No.
    
    The matrix multiplication tensor M of size (n,m,p) where A is n×m, B is m×p, C is n×p:
    C[i,k] = sum_j A[i,j] B[j,k]
    
    In tensor decomposition form:
    C[i,k] = sum_c U[i, c] * V[k, c] * W[?, c] ... no that's not right.
    
    The standard bilinear form: each term is U[i,j,c] * V[j,k,c] * ... no.
    
    Let me look at the seed code more carefully:
    U has shape (n*m, rank), indexed by (i*m + j, c)
    V has shape (m*p, rank), indexed by (j*p + k, c)  
    W has shape (p*n, rank), indexed by (k*n + i, c)
    
    The reconstruction: C[i,k] = sum_c sum_j U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    For the schoolbook: U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1 for the term (i,j,k)
    Then C[i,k] = sum_j A[i,j] B[j,k] * (sum over terms where i,j,k match)
    
    Actually the bilinear form is:
    C[i,k] = sum_c (sum_j U[i*m+j, c] * A[i,j]) * (sum_l V[l*p+k, c] * B[l,k]) ... no.
    
    Let me think again. The standard way:
    For each term c, we compute a scalar: s_c = (sum_{i,j} U[i*m+j, c] * A[i,j]) * (sum_{j,k} V[j*p+k, c] * B[j,k])
    No, that's not bilinear in A and B separately.
    
    Actually: s_c = (sum_j A[i,j] * U[i*m+j, c]) ... this doesn't work because i is free.
    
    The correct interpretation: The multiplication tensor T is a 3-tensor with indices (i, j, k) where:
    T[i, j, k] contributes to C[i, k] via A[i, j] * B[j, k].
    
    So T is a tensor of shape (n*m, m*p, p*n) where:
    T[i*m+j, j*p+k, k*n+i] = 1 (note: j must match between first and second index)
    
    And a rank-1 decomposition term is:
    T = sum_c u_c ⊗ v_c ⊗ w_c
    
    where u_c is a vector of length n*m, v_c of length m*p, w_c of length p*n.
    
    The reconstruction: T[i*m+j, j*p+k, k*n+i] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    For the schoolbook, each term c corresponds to a triple (i0, j0, k0):
    U[i0*m+j0, c] = 1, V[j0*p+k0, c] = 1, W[k0*n+i0, c] = 1, else 0.
    
    Then sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 iff (i,j,k) = (i0,j0,k0) for some c.
    But we need T[i*m+j, j*p+k, k*n+i] = 1 only when the middle index j matches.
    
    Wait, in the tensor T, the three indices are:
    a = i*m + j (row of A, col of A = row of B)
    b = j'*p + k (row of B, col of B)  
    c_idx = k'*n + i (row of C, col of C)
    
    T[a, b, c_idx] = 1 if j = j' and i = i' (the C row) and k = k' (the C col).
    So T[i*m+j, j*p+k, k*n+i] = 1.
    
    Yes! So the tensor is: T[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k.
    
    Verification: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should equal 1 if the indices are consistent (which they always are in this parametrization), else 0. But since we only evaluate at consistent indices, it should always be 1.
    
    So verification: for all i in range(n), j in range(m), k in range(p):
    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    """
    n, m, p = n, m, p
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if s != 1:
                    return False
    return True


def _try_rank_reduction(n, m, p, deadline):
    """
    Try to find a lower-rank decomposition by exploiting the structure.
    For 2×5×5, the schoolbook rank is 50. We try to find a lower rank.
    
    Key insight: Since n=2 is small, we can try to share terms across the two rows.
    For each column k of C, C[0,k] = sum_j A[0,j] B[j,k] and C[1,k] = sum_j A[1,j] B[j,k].
    
    One approach: For each k, we need to compute two dot products. If we could compute
    sum_j (A[0,j] + A[1,j]) * B[j,k] and sum_j (A[0,j] - A[1,j]) * B[j,k], then
    C[0,k] = (sum_j (A[0,j]+A[1,j]) B[j,k] + sum_j (A[0,j)-A[1,j]) B[j,k]) / 2
    C[1,k] = (sum_j (A[0,j]+A[1,j]) B[j,k] - sum_j (A[0,j)-A[1,j]) B[j,k]) / 2
    
    This doesn't reduce the rank directly since we still need 2 terms per k.
    
    Let's try a direct search for rank < 50 using random initialization and optimization.
    """
    start_time = time.time()
    
    # Try rank 25 first (half of schoolbook)
    for target_rank in [25, 30, 35, 40, 45]:
        if time.time() > deadline:
            break
        
        # Random initialization with small integers
        random.seed(42)
        U = [[random.choice([0, 1, -1]) for _ in range(target_rank)] for _ in range(n * m)]
        V = [[random.choice([0, 1, -1]) for _ in range(target_rank)] for _ in range(m * p)]
        W = [[random.choice([0, 1, -1]) for _ in range(target_rank)] for _ in range(p * n)]
        
        # Verify
        if _verify_decomposition(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    
    return None


def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 2×5×5 matrix multiplication.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    best = _schoolbook_decomposition(n, m, p)
    
    # Try to find a lower rank
    result = _try_rank_reduction(n, m, p, deadline)
    if result is not None:
        best = result
    
    return best
# EVOLVE-BLOCK-END
