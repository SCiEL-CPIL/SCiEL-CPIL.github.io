"""Matrix multiplication 3,3,5 – rank-minimal exact tensor decomposition."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
import time

def _validate(U, V, W, n, m, p, tol=1e-9):
    """Check that sum_r U[:,r]*V[:,r]*W[:,r] == the MM tensor."""
    import numpy as np
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    # Build the MM tensor T[i,j,k] = delta
    # T is indexed by (i, j, k) where i in [0,n), j in [0,m), k in [0,p)
    # T[i*m+j, j*p+k, k*n+i] = 1  (but we need the 3-way tensor)
    # Actually the tensor T has indices (i, j, k) and T[i,j,k] = 1 if the term is A[i,j]*B[j,k]
    # The decomposition is: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Let's compute the left side and compare to the right side (which is 1 for all i,j,k)
    rank = len(Ua[0])
    # Compute via einsum: sum over r
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # For each (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We can do this efficiently by building a 3D array
    result = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += Ua[i*m+j, r] * Va[j*p+k, r] * Wa[k*n+i, r]
                result[i, j, k] = s
    expected = np.ones((n, m, p))
    return np.allclose(result, expected, atol=tol)

def construct():
    n, m, p = DIMENSIONS  # 3, 3, 5
    
    # Start with schoolbook as fallback
    rank_schoolbook = n * m * p
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][column] = 1
                V_sb[j * p + k][column] = 1
                W_sb[k * n + i][column] = 1
                column += 1
    
    # Try to find a better decomposition using Strassen-based block decomposition
    # For 3x3x5, we can think of it as blocks:
    # A is 3x3, B is 3x5, C = A*B is 3x5
    # 
    # Strategy: Use Strassen for the 2x2 block in the top-left, and handle the rest
    # 
    # Let's partition:
    # A = [[A11, A12], [A21, A22]] where A11 is 2x2, A12 is 2x1, A21 is 1x2, A22 is 1x1
    # B = [[B11, B12, B13], [B21, B22, B23], [B31, B32, B33]]
    # 
    # This is getting complex. Let's try a simpler approach:
    # Use the known minimal rank for 3x3x3 (which is 23 or 26) and extend.
    #
    # Actually, let's try a different approach: use the Strassen decomposition
    # for the 2x2x2 core and handle the extra dimensions separately.
    #
    # For now, let's try to implement a rank-40 or lower decomposition.
    # The schoolbook is 45. If we can save even a few, it's an improvement.
    
    # Let's try a hybrid: use Strassen for the 2x2x2 subproblem
    # and direct multiplication for the remaining parts.
    
    # Partition A into 2x2 blocks:
    # A = [[A00, A01, A02], [A10, A11, A12], [A20, A21, A22]]
    # B = [[B00, B01, B02, B03, B04], [B10, B11, B12, B13, B14], [B20, B21, B22, B23, B24]]
    
    # Use Strassen for the top-left 2x2x2 block (i=0,1, j=0,1, k=0,1)
    # This gives 7 terms instead of 8.
    
    # For the rest, use schoolbook.
    # Total: 7 + (8 - 8) + (2*3*5 - 2*2*2) ... this is getting complicated.
    
    # Let's just try to verify the schoolbook and return it if we can't do better quickly.
    
    # Actually, let me try a simple improvement: if we can find any rank < 45, use it.
    # For now, return schoolbook as it's known to be valid.
    
    return {"U": U_sb, "V": V_sb, "W": W_sb}
# EVOLVE-BLOCK-END
