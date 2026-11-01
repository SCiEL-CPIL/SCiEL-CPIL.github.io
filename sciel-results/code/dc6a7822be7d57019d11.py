"""Tensor decomposition for 2x2 matrix multiplication (Strassen, rank 7)."""
DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
import time
import numpy as np

_T0 = time.time()
_DEADLINE = _T0 + 140.0  # safety margin before 150 s kill


def _strassen_factors():
    """Build the canonical Strassen rank-7 factor matrices.

    Convention (as required by the task):
        U[i*m + j, c]  for A[i, j]
        V[j*p + k, c]  for B[j, k]
        W[k*n + i, c]  for C[i, k]
    with n = m = p = 2.
    """
    n = m = p = 2
    rank = 7
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    # m0 = (a00+a11)(b00+b11)
    U[0, 0] = 1.0; U[3, 0] = 1.0
    V[0, 0] = 1.0; V[3, 0] = 1.0
    W[0, 0] = 1.0; W[3, 0] = 1.0

    # m1 = (a10+a11)b00
    U[2, 1] = 1.0; U[3, 1] = 1.0
    V[0, 1] = 1.0
    W[1, 1] = 1.0; W[3, 1] = -1.0

    # m2 = a00(b01-b11)
    U[0, 2] = 1.0
    V[1, 2] = 1.0; V[3, 2] = -1.0
    W[2, 2] = 1.0; W[3, 2] = 1.0

    # m3 = a11(b10-b00)
    U[3, 3] = 1.0
    V[2, 3] = 1.0; V[0, 3] = -1.0
    W[0, 3] = 1.0; W[1, 3] = 1.0

    # m4 = (a00+a01)b11
    U[0, 4] = 1.0; U[1, 4] = 1.0
    V[3, 4] = 1.0
    W[0, 4] = -1.0; W[2, 4] = 1.0

    # m5 = (a10-a00)(b00+b01)
    U[2, 5] = 1.0; U[0, 5] = -1.0
    V[0, 5] = 1.0; V[1, 5] = 1.0
    W[3, 5] = 1.0

    # m6 = (a01-a11)(b10+b11)
    U[1, 6] = 1.0; U[3, 6] = -1.0
    V[2, 6] = 1.0; V[3, 6] = 1.0
    W[0, 6] = 1.0

    return U, V, W


def _verify(U, V, W, n=2, m=2, p=2, tol=1e-10):
    """Check that sum_c U[:,c] outer V[:,c] outer W[:,c] matches matmul tensor."""
    n, m, p = 2, 2, 2
    # Build the multiplication tensor T[a_idx, b_idx, c_idx]
    # where a_idx = i*m+j, b_idx = j'*p+k, c_idx = k'*n+i'
    # T[a,b,c] = 1 iff i==j', j==k', and output index k'*n+i' == c_idx
    # More directly: for each c (output position k*n+i), the matmul tensor
    # T[(i,m,j),(j,p,k),(k,n,i)] = sum_{j,k} delta(i,j') delta(j,k') ... 
    # Actually the matmul tensor is: T[a,b,c] where a=(i,j), b=(j',k), c=(k',i')
    # T[a,b,c] = 1 if j==j' and k==k' and i==i' ... no.
    # The matmul tensor M[i,j, j',k, k',i'] = delta_{j,j'} delta_{k,k'} 
    # Wait, matrix multiplication: C[i,k'] = sum_{j} A[i,j] B[j,k']
    # So the tensor is T[i,j, j',k'] = delta_{j,j'}
    # In flattened indices: a = i*m+j, b = j'*p+k', c = k'*n+i
    # T[a,b,c] = 1 if (j from a) == (j' from b) AND (k' from b) == (k' from c) AND (i from a) == (i from c)
    
    # Let's just check: for random matrices A, B, compute C = A@B
    # and compare with sum_c U_c * V_c * W_c (broadcasting)
    # Actually the decomposition says: C[i,k] = sum_c (sum_j U[i,j,c]*A[i,j]) * (sum_j' V[j',k,c]*B[j',k]) ... no
    # The correct check: for the tensor T[i,j,j',k'] = delta_{j,j'} (with output indexed by i,k')
    # The decomposition is: T[i,j, j',k'] = sum_c U[i,j,c] * V[j',k',c] * W[k',i,c]
    # Wait, W has shape (p*n, rank) = (k'*n+i, rank) so W[k'*n+i, c]
    # So: T[i,j, j',k'] = sum_c U[i*m+j, c] * V[j'*p+k', c] * W[k'*n+i, c]
    # And T[i,j,j',k'] = 1 if j == j' (since C[i,k'] = sum_j A[i,j]B[j,k'], the tensor element
    #   for inputs (i,j) and (j',k') contributing to output (i,k') is delta_{j,j'})
    
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    expected = 1.0 if j == jp else 0.0
                    actual = 0.0
                    for c in range(7):
                        actual += U[i*m + j, c] * V[jp*p + k, c] * W[k*n + i, c]
                    if abs(actual - expected) > tol:
                        return False
    return True


def construct():
    """
    Returns Strassen's rank-7 decomposition for 2x2 matrix multiplication.
    The rank is 7, which is the proven theoretical minimum.
    """
    U, V, W = _strassen_factors()
    
    # Verify correctness; if somehow broken, fall back to a corrected version
    if not _verify(U, V, W):
        # This should never happen, but as a safety net:
        # Rebuild with explicit known-good values
        U = np.array([
            [1, 0, 1, 0, 1, -1, 0],
            [0, 0, 0, 0, 1, 0, 1],
            [0, 1, 0, 0, 0, 1, 0],
            [1, 1, 0, 1, 0, 0, -1]
        ], dtype=float)
        V = np.array([
            [1, 1, 0, -1, 0, 1, 0],
            [0, 0, 1, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 1],
            [1, 0, -1, 0, 1, 0, 1]
        ], dtype=float)
        W = np.array([
            [1, 0, 0, 1, -1, 0, 1],
            [0, 1, 0, 1, 0, 0, 0],
            [0, 0, 1, 0, 1, 0, 0],
            [1, -1, 1, 0, 0, 1, 0]
        ], dtype=float)
    
    # Convert to lists for the return interface
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
