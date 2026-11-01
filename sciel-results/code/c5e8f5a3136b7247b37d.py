import time
import numpy as np

DIMENSIONS = [2, 2, 3]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 140.0
    
    n, m, p = DIMENSIONS
    
    # Attempt rank-10 decomposition via structured search
    result = _try_rank10(n, m, p, start_time, deadline)
    if result is not None:
        return result
    
    # Fallback to known rank-11 solution
    return _rank11_solution(n, m, p)

def _try_rank10(n, m, p, start_time, deadline):
    """Try to construct a rank-10 decomposition."""
    if time.time() > deadline:
        return None
    
    rank = 10
    dim_U = n * m  # 4
    dim_V = m * p  # 6
    dim_W = p * n  # 6
    
    # Build on Strassen's 7 terms for (2,2,2) and add 3 terms for the k=2 column.
    # The key idea: combine the 4 terms needed for column 2 into 3 rank-1 terms
    # by exploiting linear dependencies.
    
    # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2]
    # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2]
    
    # Define:
    # M8 = (A[0,0] + A[1,1]) * B[0,2]
    # M9 = (A[0,1] + A[1,0]) * B[1,2]
    # M10 = (A[0,0] - A[1,1]) * B[0,2]  -- this doesn't help directly
    
    # Alternative: 
    # M8 = A[0,0]*B[0,2]
    # M9 = A[1,1]*B[1,2]
    # M10 = (A[0,1] + A[1,0]) * B[0,2]  -- no, B indices don't match
    
    # Let's try:
    # M8 = (A[0,0] + A[0,1]) * B[0,2]  -> contributes A[0,0]*B[0,2] + A[0,1]*B[0,2]
    # But A[0,1]*B[0,2] is not in C[0,2] or C[1,2]. This creates spurious terms.
    
    # The problem is that any combination that mixes A indices with the wrong B indices
    # will create spurious nonzero tensor entries.
    
    # For a rank-1 term with U having entries at (i1,j1) and (i2,j2), V having entry at (j3,k3),
    # and W having entry at (k4,i4), the tensor gets contributions at:
    # (k4*n+i4, j3*p+k3, i1*m+j1) and (k4*n+i4, j3*p+k3, i2*m+j2)
    # These must be valid (i,j,k) triples, meaning j1 == j3 and i4 == i1, etc.
    
    # So for term r:
    # U[r] has support on some set S_U of (i,j) pairs
    # V[r] has support on some set S_V of (j,k) pairs  
    # W[r] has support on some set S_W of (k,i) pairs
    # For the product to be valid, we need: for each (i,j) in S_U, (j,k) in S_V, (k,i) in S_W,
    # the triple (i,j,k) must be consistent (j from U matches j from V, k from V matches k from W, i from W matches i from U).
    
    # This is very constrained. Let me just verify rank 11 is minimal and return it.
    
    # Actually, let me try a specific rank-10 construction:
    # Use all 7 Strassen terms, then:
    # M8 = A[0,0]*B[0,2]
    # M9 = A[1,0]*B[0,2]  
    # M10 = (A[0,1] + A[1,1])*B[1,2]
    
    # Then:
    # C[0,2] = M8 + A[0,1]*B[1,2] = M8 + (contribution from M10 with A[0,1])
    # C[1,2] = M9 + A[1,1]*B[1,2] = M9 + (contribution from M10 with A[1,1])
    
    # But M10 is a single rank-1 term: U has A[0,1] and A[1,1], V has B[1,2], W has...
    # M10 contributes to W at positions for C[0,2] and C[1,2].
    # If W[2,0] = 1 and W[2,1] = 1, then M10 contributes:
    #   To C[0,2]: (A[0,1] + A[1,1])*B[1,2] * 1
    #   To C[1,2]: (A[0,1] + A[1,1])*B[1,2] * 1
    # But we need C[0,2] to get A[0,1]*B[1,2] and C[1,2] to get A[1,1]*B[1,2].
    # M10 gives both C[0,2] and C[1,2] the same value (A[0,1]+A[1,1])*B[1,2].
    # So C[0,2] = M8 + (A[0,1]+A[1,1])*B[1,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2] + A[1,1]*B[1,2]  -- WRONG
    # C[1,2] = M9 + (A[0,1]+A[1,1])*B[1,2] = A[1,0]*B[0,2] + A[0,1]*B[1,2] + A[1,1]*B[1,2]  -- WRONG
    
    # This approach fails. I believe rank 11 is minimal for (2,2,3).
    
    return None

def _rank11_solution(n, m, p):
    rank = 11
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def set_u(i, j, r, val):
        U[i * m + j][r] = val
    def set_v(j, k, r, val):
        V[j * p + k][r] = val
    def set_w(k, i, r, val):
        W[k * n + i][r] = val

    # Term 1: (A00 + A11) * (B00 + B11)
    set_u(0, 0, 0, 1); set_u(1, 1, 0, 1)
    set_v(0, 0, 0, 1); set_v(1, 1, 0, 1)
    set_w(0, 0, 0, 1)
    set_w(1, 1, 0, 1)

    # Term 2: (A10 + A11) * B00
    set_u(1, 0, 1, 1); set_u(1, 1, 1, 1)
    set_v(0, 0, 1, 1)
    set_w(0, 1, 1, 1)
    set_w(1, 1, 1, -1)

    # Term 3: A00 * (B01 - B11)
    set_u(0, 0, 2, 1)
    set_v(0, 1, 2, 1); set_v(1, 1, 2, -1)
    set_w(1, 0, 2, 1)
    set_w(1, 1, 2, 1)

    # Term 4: A11 * (B10 - B00)
    set_u(1, 1, 3, 1)
    set_v(1, 0, 3, 1); set_v(0, 0, 3, -1)
    set_w(0, 0, 3, 1)
    set_w(0, 1, 3, 1)

    # Term 5: (A00 + A01) * B11
    set_u(0, 0, 4, 1); set_u(0, 1, 4, 1)
    set_v(1, 1, 4, 1)
    set_w(0, 0, 4, -1)
    set_w(1, 0, 4, 1)

    # Term 6: (A10 - A00) * (B00 + B01)
    set_u(1, 0, 5, 1); set_u(0, 0, 5, -1)
    set_v(0, 0, 5, 1); set_v(0, 1, 5, 1)
    set_w(1, 1, 5, 1)

    # Term 7: (A01 - A11) * (B10 + B11)
    set_u(0, 1, 6, 1); set_u(1, 1, 6, -1)
    set_v(1, 0, 6, 1); set_v(1, 1, 6, 1)
    set_w(0, 0, 6, 1)

    # Term 8: A00 * B02
    set_u(0, 0, 7, 1)
    set_v(0, 2, 7, 1)
    set_w(2, 0, 7, 1)

    # Term 9: A01 * B12
    set_u(0, 1, 8, 1)
    set_v(1, 2, 8, 1)
    set_w(2, 0, 8, 1)

    # Term 10: A10 * B02
    set_u(1, 0, 9, 1)
    set_v(0, 2, 9, 1)
    set_w(2, 1, 9, 1)

    # Term 11: A11 * B12
    set_u(1, 1, 10, 1)
    set_v(1, 2, 10, 1)
    set_w(2, 1, 10, 1)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
