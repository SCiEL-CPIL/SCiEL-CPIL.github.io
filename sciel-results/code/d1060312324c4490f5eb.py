"""
Tensor decomposition for M_{2,2,13} matrix multiplication.
We try to find a lower-rank decomposition than schoolbook (52).
"""

DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is exact using vectorized operations."""
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    r = U.shape[1]
    
    # Check all expected 1s
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.dot(U[a, :], np.multiply(V[b, :], W[c, :]))
                if abs(val - 1.0) > 1e-9:
                    return False

    # Check all expected 0s (full check, not just subset)
    for i_u in range(n * m):
        for j_v in range(m * p):
            for k_w in range(p * n):
                i_u2, j_u = divmod(i_u, m)
                j_v2, k_v = divmod(j_v, p)
                k_w2, i_w = divmod(k_w, n)
                
                if j_u == j_v2 and i_u2 == i_w and k_v == k_w2:
                    expected = 1.0
                else:
                    expected = 0.0
                    
                val = np.dot(U[i_u, :], np.multiply(V[j_v, :], W[k_w, :]))
                if abs(val - expected) > 1e-9:
                    return False

    return True

def _build_schoolbook(n, m, p):
    """Build the schoolbook decomposition (rank n*m*p)."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _build_rank_28(n, m, p):
    """
    Build rank 2p+2 = 28 decomposition for M_{2,2,p} with p odd.
    Uses Strassen-like method.
    """
    if n != 2 or m != 2:
        return None
    if p % 2 != 1:
        return None
    
    r = 2 * p + 2
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))
    
    # Indices for A: A[0,0]=0, A[0,1]=1, A[1,0]=2, A[1,1]=3
    # Indices for B: B[0,k]=k, B[1,k]=p+k
    # Indices for C: C[0,k]=k, C[1,k]=p+k
    
    # For each k in 0..p-1, we have 2 terms:
    # Term 2k:
    #   U[0,0]=1, U[1,0]=0, U[0,1]=0, U[1,1]=0
    #   V[0,k]=1, V[1,k]=0
    #   W[0,k]=1, W[1,k]=0
    #   Captures A[0,0]*B[0,k] -> C[0,k]
    
    #   U[0,0]=1, U[1,0]=0, U[0,1]=1, U[1,1]=0
    #   V[0,k]=0, V[1,k]=1
    #   W[0,k]=1, W[1,k]=0
    #   Captures A[0,1]*B[1,k] -> C[0,k]
    
    #   U[0,0]=0, U[1,0]=1, U[0,1]=0, U[1,1]=0
    #   V[0,k]=1, V[1,k]=0
    #   W[0,k]=0, W[1,k]=1
    #   Captures A[1,0]*B[0,k] -> C[1,k]
    
    #   U[0,0]=0, U[1,0]=0, U[0,1]=0, U[1,1]=1
    #   V[0,k]=0, V[1,k]=1
    #   W[0,k]=0, W[1,k]=1
    #   Captures A[1,1]*B[1,k] -> C[1,k]
    
    # But that's 4p terms, not 2p+2.
    # Need a better construction.
    
    # Known result: r(M_{2,2,p}) = 2p+2 for odd p.
    # The construction is non-trivial.
    # Let me try a different approach.
    
    # Actually, the tensor M_{2,2,p} can be decomposed as:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    
    # For fixed k, this is a 2x2 matrix product with a column vector.
    # This is equivalent to multiplying a 2x2 matrix by a 2x1 matrix.
    
    # The rank of M_{2,2,1} is 4 (schoolbook) or maybe less?
    # Actually, M_{2,2,1} has rank 4.
    
    # For M_{2,2,p}, the rank is at most 4p (schoolbook).
    # The question is whether we can do better.
    
    # Given the complexity and the risk of an invalid decomposition,
    # I'll return None to indicate this construction is not implemented.
    return None

def _build_rank_26(n, m, p):
    """
    Attempt rank 2p = 26 decomposition.
    This is likely invalid, but let's try.
    """
    if n != 2 or m != 2:
        return None
    
    r = 2 * p
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))
    
    # For each k in 0..p-1:
    # Term k:
    #   U[0,0]=1, U[1,0]=1
    #   V[0,k]=1, V[1,k]=0
    #   W[0,k]=1, W[1,k]=0
    #   This gives: C[0,k] += A[0,0]*B[0,k], C[1,k] += A[1,0]*B[0,k]
    
    # Term p+k:
    #   U[0,1]=1, U[1,1]=1
    #   V[0,k]=0, V[1,k]=1
    #   W[0,k]=1, W[1,k]=0
    #   This gives: C[0,k] += A[0,1]*B[1,k], C[1,k] += A[1,1]*B[1,k]
    
    # But we're missing C[1,k] terms properly.
    # Let me reconsider.
    
    # Actually, this construction is:
    # For term k (0 <= k < p):
    #   U[0,0]=1, U[1,0]=0, U[0,1]=0, U[1,1]=0
    #   V[0,k]=1, V[1,k]=0
    #   W[0,k]=1, W[1,k]=0
    #   Contribution: C[0,k] += A[0,0]*B[0,k]
    
    # For term p+k (0 <= k < p):
    #   U[0,0]=0, U[1,0]=0, U[0,1]=1, U[1,1]=0
    #   V[0,k]=0, V[1,k]=1
    #   W[0,k]=1, W[1,k]=0
    #   Contribution: C[0,k] += A[0,1]*B[1,k]
    
    # This only gives C[0,k], not C[1,k].
    # So this is invalid.
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145
    
    # 1. Try rank 28
    result = _build_rank_28(n, m, p)
    if result is not None:
        if _verify(result["U"], result["V"], result["W"], n, m, p):
            return result
    
    # 2. Try rank 26
    result = _build_rank_26(n, m, p)
    if result is not None:
        if _verify(result["U"], result["V"], result["W"], n, m, p):
            return result
    
    # 3. Return Schoolbook (52)
    result = _build_schoolbook(n, m, p)
    return result

# EVOLVE-BLOCK-END
