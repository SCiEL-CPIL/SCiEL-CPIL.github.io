"""Tensor decomposition for 3x3 * 3x8 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [3, 3, 8]
DEADLINE = 120.0

# EVOLVE-BLOCK-START

def _schoolbook(n, m, p):
    """Standard rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W


def _verify(U, V, W, n, m, p, tol=1e-9):
    """
    Verify that the tensor decomposition is exact.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    Check: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 for all (i,j,k)
    """
    rank = U.shape[1]
    T = np.zeros((n, m, p))
    for c in range(rank):
        uc = U[:, c]
        vc = V[:, c]
        wc = W[:, c]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += uc[i * m + j] * vc[j * p + k] * wc[k * n + i]
    expected = np.ones((n, m, p))
    return np.allclose(T, expected, atol=tol)


def _construct_strassen_blocks(n, m, p):
    """
    Construct a decomposition using 2x2 Strassen blocks where possible.
    
    For 3x3 * 3x8:
    - The 3x3 matrix A can be partitioned with a 2x2 block (rows 0,1; cols 0,1)
    - The 3x8 matrix B can be partitioned with 2x2 blocks
    
    Strategy: Use Strassen for the (0,1)x(0,1) block of A and corresponding B blocks.
    
    Let's use a simpler approach:
    Partition A into:
      A = [[A11, A12, A13],
           [A21, A22, A23],
           [A31, A32, A33]]
    
    Partition B into columns:
      B = [B1, B2, B3] where B1, B2, B3 are 3x8 blocks (rows)
    
    Actually, let's think of it differently.
    
    For the 2x2 Strassen trick to work, we need to compute:
    C_block = A_block * B_block
    
    Let's partition the computation:
    - Rows 0,1 of C come from rows 0,1 of A
    - Columns 0,1 of C come from columns 0,1 of B
    
    So C[0:2, 0:2] = A[0:2, :] * B[:, 0:2]
    
    This is a 2x3 * 3x2 multiplication.
    
    We can decompose this as:
    A[0:2,:] = [[A00, A01, A02], [A10, A11, A12]]
    B[:,0:2] = [[B00, B01], [B10, B11], [B20, B21]]
    
    C[0:2, 0:2] = A[0:2,:] * B[:,0:2]
    
    This is a 2x3 * 3x2 matrix multiplication.
    
    Known: M(2,3,2) <= 10 (vs schoolbook 12)
    
    Actually, let me try a different approach. Let's use the fact that:
    M(3,3,8) can be bounded using known results.
    
    Known bounds:
    - M(3,3,3) <= 23 (Bini et al.)
    - M(3,3,8) <= M(3,3,3) * M(3,8,8)? No, that's not how it works.
    
    Let me try a direct construction using tensor rank properties.
    
    Actually, for now, let's just return the schoolbook and add a simple optimization.
    """
    return _schoolbook(n, m, p)


def _construct_gf3_decomposition(n, m, p):
    """
    Attempt to construct a decomposition using algebraic structure over GF(3).
    
    The idea: map matrix indices to field elements and use multiplicative properties.
    For 3x3 * 3x8, we can try to exploit the cyclic group structure of GF(3)^*.
    
    However, this is complex. Let's try a simpler algebraic approach.
    
    For the 3x3 part, we know that M(3,3,3) <= 23.
    For the 3x8 part, we can use linearity in the last dimension.
    
    Actually, M(3,3,8) = M(3,3,3) + M(3,3,3) + M(3,3,2) <= 23 + 23 + M(3,3,2)
    And M(3,3,2) <= 18 (schoolbook) or possibly less.
    
    This gives M(3,3,8) <= 23 + 23 + 18 = 64, which is better than 72.
    
    But we need an explicit construction.
    
    For now, return schoolbook.
    """
    return _schoolbook(n, m, p)


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Start with schoolbook as baseline
    U, V, W = _schoolbook(n, m, p)
    
    # Verify schoolbook
    assert _verify(U, V, W, n, m, p), "Schoolbook verification failed"
    
    best_rank = U.shape[1]
    best_U, best_V, best_W = U, V, W
    
    # Try to find a lower-rank decomposition
    # For 3x3 * 3x8, the schoolbook rank is 72.
    
    # Check time
    if time.time() - start_time > DEADLINE - 5:
        U_list = best_U.tolist()
        V_list = best_V.tolist()
        W_list = best_W.tolist()
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Convert to lists of lists as required
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
