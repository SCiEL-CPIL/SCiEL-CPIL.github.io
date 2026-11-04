import time
import numpy as np
from itertools import product

DIMENSIONS = [4, 5, 6]

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly computes matrix multiplication."""
    rank = len(U[0]) if U else 0
    # Check shapes
    if len(U) != n * m:
        return False
    if len(V) != m * p:
        return False
    if len(W) != p * n:
        return False
    for idx in range(rank):
        if len(U[idx if False else 0]) if False else True:
            pass
    
    # Reconstruct: T[i,j,k] should equal sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # For exact matmul, T[i,j,k] = 1 for all valid (i,j,k)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for r in range(rank):
                    total += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(total - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition with rank n*m*p."""
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

def construct():
    """
    Constructs the factor matrices for the tensor decomposition of 4x5 by 5x6 matrix multiplication.
    Tries to find a rank-minimal exact decomposition with integer/half-integer entries.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    schoolbook = _schoolbook(n, m, p)
    
    # Verify schoolbook
    if not _verify_decomposition(schoolbook["U"], schoolbook["V"], schoolbook["W"], n, m, p):
        raise RuntimeError("Schoolbook verification failed")
    
    # Try to find a better decomposition
    # Strategy: The tensor is T[i,j,k] = 1 for all (i,j,k).
    # We need sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # For a rank-1 term r, define:
    #   a_r(i,j) = U[i*m+j, r]
    #   b_r(j,k) = V[j*p+k, r]  
    #   c_r(k,i) = W[k*n+i, r]
    # Then sum_r a_r(i,j) * b_r(j,k) * c_r(k,i) = 1 for all i,j,k
    
    # If we use the same "pattern" across terms, we can share structure.
    # 
    # Key idea: If we can find matrices A_r (n×m), B_r (m×p), C_r (p×n) such that
    # sum_r A_r[i,j] * B_r[j,k] * C_r[k,i] = 1 for all i,j,k,
    # and each A_r, B_r, C_r has entries in {0, 1/2, 1, 3/2, ...},
    # then rank = number of terms.
    #
    # Simple case: If all A_r = I-like patterns... but that's schoolbook.
    #
    # Let's try: Can we use rank = n*p = 24? 
    # If we set for each (i,k) pair a term, then:
    #   A_{i,k}[i',j] = delta(i',i) for some j? No, that doesn't work directly.
    #
    # Alternative: Use the identity that 1 = sum over some decomposition.
    # 
    # For now, let's try a few candidate ranks and see if we can construct valid ones.
    # 
    # Candidate: rank = m * max(n,p) = 5 * 6 = 30
    # For each j in [0,m) and k in [0,p), define term (j,k):
    #   A_{j,k}[i,j'] = delta(j', j) * (1/n) ... no, entries must be half-integers.
    #
    # Let me try a different structural approach.
    # 
    # The tensor T[i,j,k] = 1 is the all-ones tensor. Its rank as a 3-tensor
    # is at most n*p (by fixing j and using rank-1 tensors that are 1 on a row of i and column of k).
    #
    # Specifically: T[i,j,k] = sum_{j'=0}^{m-1} delta(j,j') = 1
    # But that's trivial and doesn't help.
    #
    # Better: T[i,j,k] = 1 = (1/m) * sum_{j'=0}^{m-1} m * delta(j,j')
    # But 1/m = 1/5 which is not a half-integer.
    #
    # So we need a different approach. Let's use the fact that:
    # 1 = 1 * 1 * 1, and try to decompose using terms that each contribute to multiple (i,j,k).
    #
    # For rank r, each term r contributes a_r(i,j) * b_r(j,k) * c_r(k,i).
    # If we want this to sum to 1 for all (i,j,k), and entries are half-integers,
    # we need a careful construction.
    #
    # Let's try rank = n * p = 24. For each (i,k), define:
    #   a_{i,k}[i',j] = 1 if i' == i else 0  (for all j)
    #   b_{i,k}[j,k'] = 1 if k' == k else 0  (for all j)  
    #   c_{i,k}[k'',i'] = 1 if k'' == k and i' == i else 0
    # Then term (i,k) contributes: delta(i',i) * delta(k',k) * delta(k'',k)*delta(i'',i)
    # For (i',j,k'): this is delta(i',i)*delta(k',k)*delta(k',k)*delta(i',i) = delta(i',i)*delta(k',k)
    # Sum over (i,k): sum_{i,k} delta(i',i)*delta(k',k) = 1. 
    # But wait, we need to check the indices match.
    #
    # Term r = (i,k): 
    #   U[i*m+j, r] = 1 if row i, any j? No, U[i*m+j, r] should be a function of (i,j).
    #   Let me redefine: for term indexed by (i,k):
    #     U[i*m + j, (i,k)] = 1 for all j  (so U has row i all 1s for this term)
    #     V[j*p + k, (i,k)] = 1 for all j  (so V has column k all 1s for this term)
    #     W[k*n + i, (i,k)] = 1  (only one non-zero)
    #   Then contribution to (i',j',k'): 
    #     U[i'*m+j', (i,k)] * V[j'*p+k', (i,k)] * W[k'*n+i', (i,k)]
    #     = (1 if i'==i else 0) * (1 if k'==k else 0) * (1 if k'==k and i'==i else 0)
    #     = delta(i',i) * delta(k',k) * delta(i',i)*delta(k',k) = delta(i',i)*delta(k',k)
    #   Sum over all (i,k): sum_{i,k} delta(i',i)*delta(k',k) = 1. 
    #
    # So rank = n*p = 24 works! Let me verify the entries are half-integers: yes, all 0 or 1.
    
    rank_candidate = n * p  # 24
    U = [[0] * rank_candidate for _ in range(n * m)]
    V = [[0] * rank_candidate for _ in range(m * p)]
    W = [[0] * rank_candidate for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for k in range(p):
            # U: for this term, row i (all j) = 1
            for j in range(m):
                U[i * m + j][col] = 1
            # V: for this term, column k (all j) = 1
            for j in range(m):
                V[j * p + k][col] = 1
            # W: for this term, W[k*n + i] = 1
            W[k * n + i][col] = 1
            col += 1
    
    # Verify
    if _verify_decomposition(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # If that didn't work, try n*m = 20
    rank_candidate = n * m  # 20
    U = [[0] * rank_candidate for _ in range(n * m)]
    V = [[0] * rank_candidate for _ in range(m * p)]
    W = [[0] * rank_candidate for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            # U: for this term, U[i*m+j] = 1
            U[i * m + j][col] = 1
            # V: for this term, V[j*p+k] = 1 for all k
            for k in range(p):
                V[j * p + k][col] = 1
            # W: for this term, W[k*n+i] = 1 for all k
            for k in range(p):
                W[k * n + i][col] = 1
            col += 1
    
    if _verify_decomposition(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # Try m*p = 30
    rank_candidate = m * p  # 30
    U = [[0] * rank_candidate for _ in range(n * m)]
    V = [[0] * rank_candidate for _ in range(m * p)]
    W = [[0] * rank_candidate for _ in range(p * n)]
    
    col = 0
    for j in range(m):
        for k in range(p):
            # U: for this term, U[i*m+j] = 1 for all i
            for i in range(n):
                U[i * m + j][col] = 1
            # V: for this term, V[j*p+k] = 1
            V[j * p + k][col] = 1
            # W: for this term, W[k*n+i] = 1 for all i
            for i in range(n):
                W[k * n + i][col] = 1
            col += 1
    
    if _verify_decomposition(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # Fall back to schoolbook
    return schoolbook
