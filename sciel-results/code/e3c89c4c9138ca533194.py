import time
import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START

def _strassen_2x2x2():
    """Return (U, V, W) each of shape (4,7) for local 2x2x2 Strassen.
    
    U[row, col] where row = i*2 + a, col = c
    V[row, col] where row = b*2 + k, col = c
    W[row, col] where row = k*2 + i, col = c  (k*n+i with n=2 locally)
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # c=0 (M1): (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # c=1 (M2): (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1
    
    # c=2 (M3): A00(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = -1
    
    # c=3 (M4): A11(B10-B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[2, 3] = 1; W[0, 3] = -1
    
    # c=4 (M5): (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[1, 4] = 1
    
    # c=5 (M6): (A10-A00)(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[0, 5] = 1
    
    # c=6 (M7): (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[3, 6] = 1
    
    return U, V, W


def _validate_decomposition(U, V, W, n, m, p, tol=1e-7):
    """Check that the factor matrices define the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Check: for all i, a, b, k: sum_c W[k*n+i, c] * U[i*m+a, c] * V[b*p+k, c] == (a==b)
    # This is O(n * m * m * p * rank) checks.
    
    for i in range(n):
        for a in range(m):
            for b in range(m):
                for k in range(p):
                    s = np.sum(W[k * n + i] * U[i * m + a] * V[b * p + k])
                    expected = 1.0 if a == b else 0.0
                    if abs(s - expected) > tol:
                        return False
    return True


def _try_block_decomposition(n, m, p):
    """
    Try to construct a decomposition using block Strassen for 2x2x2 sub-blocks.
    
    Strategy: Partition the dimensions into blocks, and for each 2x2x2 block of
    (I_block, J_block, K_block) use Strassen (rank 7) instead of schoolbook (rank 8).
    For remaining 1-wide slices, use schoolbook (rank 1 each).
    """
    # Define block partitions
    I_blocks = [(0, 2), (2, 1)]
    J_blocks = [(0, 2), (2, 2), (4, 1)]
    K_blocks = [(0, 2), (2, 2), (4, 2), (6, 1)]
    
    # Calculate total rank
    total_rank = 0
    for I_start, I_size in I_blocks:
        for J_start, J_size in J_blocks:
            for K_start, K_size in K_blocks:
                if I_size == 2 and J_size == 2 and K_size == 2:
                    total_rank += 7
                else:
                    total_rank += I_size * J_size * K_size
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    strassen_U, strassen_V, strassen_W = _strassen_2x2x2()
    
    rank = 0
    for I_start, I_size in I_blocks:
        for J_start, J_size in J_blocks:
            for K_start, K_size in K_blocks:
                if I_size == 2 and J_size == 2 and K_size == 2:
                    # Use Strassen
                    for li in range(2):
                        for la in range(2):
                            U[(I_start + li) * m + (J_start + la), rank:rank+7] = strassen_U[li * 2 + la]
                    for lb in range(2):
                        for lk in range(2):
                            V[(J_start + lb) * p + (K_start + lk), rank:rank+7] = strassen_V[lb * 2 + lk]
                    for lk in range(2):
                        for li in range(2):
                            W[(K_start + lk) * n + (I_start + li), rank:rank+7] = strassen_W[lk * 2 + li]
                    rank += 7
                else:
                    # Schoolbook for this block
                    for li in range(I_size):
                        for lj in range(J_size):
                            for lk in range(K_size):
                                U[(I_start + li) * m + (J_start + lj), rank] = 1
                                V[(J_start + lj) * p + (K_start + lk), rank] = 1
                                W[(K_start + lk) * n + (I_start + li), rank] = 1
                                rank += 1
    
    return U.tolist(), V.tolist(), W.tolist()


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin
    
    # Try the block Strassen decomposition
    U, V, W = _try_block_decomposition(n, m, p)
    
    # Validate
    if time.time() < deadline:
        if _validate_decomposition(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
