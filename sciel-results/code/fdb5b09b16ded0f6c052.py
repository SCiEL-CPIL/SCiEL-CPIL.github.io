"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = 150.0
_SAFETY_MARGIN = 5.0

def _remaining_time():
    return _DEADLINE - _SAFETY_MARGIN - (time.time() - _START_TIME)

def _verify(U, V, W):
    """Verify that the decomposition is correct."""
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Check valid entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.sum(U[a, :] * V[b, :] * W[c, :])
                if abs(val - 1.0) > 1e-6:
                    return False
    
    # Check some invalid entries (spot check)
    for i in range(n):
        for j in range(m):
            for j2 in range(m):
                if j2 == j:
                    continue
                for k in range(p):
                    a = i * m + j
                    b = j2 * p + k
                    c = k * n + i
                    val = np.sum(U[a, :] * V[b, :] * W[c, :])
                    if abs(val) > 1e-6:
                        return False
    return True

def _strassen_2x2x2(offset_a, offset_b, offset_c, n, m, p):
    """
    Return (U, V, W) for a 2x2x2 block at given offsets.
    offset_a: starting index in A (i*m + j)
    offset_b: starting index in B (j*p + k)
    offset_c: starting index in C (k*n + i)
    
    The 2x2x2 block corresponds to:
    A rows [i0, i0+1], cols [j0, j0+1]
    B rows [j0, j0+1], cols [k0, k0+1]
    C rows [i0, i0+1], cols [k0, k0+1]
    """
    # Strassen 7 terms for 2x2x2
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    U = np.zeros((n*m, 7))
    V = np.zeros((m*p, 7))
    W = np.zeros((p*n, 7))
    
    # M1 = (A00+A11)(B00+B11)
    U[offset_a + 0, 0] = 1  # A00
    U[offset_a + 2, 0] = 1  # A11 (row 1, col 1 -> index i0+1, j0+1 -> (i0+1)*m + (j0+1))
    V[offset_b + 0, 0] = 1  # B00
    V[offset_b + 2, 0] = 1  # B11 (row 1, col 1 -> index j0+1, k0+1 -> (j0+1)*p + (k0+1))
    W[offset_c + 0, 0] = 1  # C00
    W[offset_c + 2, 0] = 1  # C11 (row 1, col 1 -> index k0+1, i0+1 -> (k0+1)*n + (i0+1))
    
    # M2 = (A10+A11)B00
    U[offset_a + 1, 1] = 1  # A10 (row 1, col 0)
    U[offset_a + 2, 1] = 1  # A11
    V[offset_b + 0, 1] = 1  # B00
    W[offset_c + 1, 1] = 1  # C10 (row 1, col 0)
    W[offset_c + 2, 1] = 1  # C11
    
    # M3 = A00(B01-B11)
    U[offset_a + 0, 2] = 1  # A00
    V[offset_b + 1, 2] = 1  # B01 (row 0, col 1)
    V[offset_b + 2, 2] = -1  # -B11
    W[offset_c + 1, 2] = 1  # C01 (row 0, col 1)
    W[offset_c + 2, 2] = 1  # C11
    
    # M4 = A11(B10-B00)
    U[offset_a + 2, 3] = 1  # A11
    V[offset_b + 2, 3] = 1  # B10 (row 1, col 0)
    V[offset_b + 0, 3] = -1  # -B00
    W[offset_c + 1, 3] = 1  # C10
    W[offset_c + 2, 3] = 1  # C11
    
    # M5 = (A00+A01)B11
    U[offset_a + 0, 4] = 1  # A00
    U[offset_a + 1, 4] = 1  # A01 (row 0, col 1)
    V[offset_b + 2, 4] = 1  # B11
    W[offset_c + 1, 4] = 1  # C01
    W[offset_c + 2, 4] = 1  # C11
    
    # M6 = (A10-A00)(B00+B01)
    U[offset_a + 1, 5] = 1  # A10
    U[offset_a + 0, 5] = -1  # -A00
    V[offset_b + 0, 5] = 1  # B00
    V[offset_b + 1, 5] = 1  # B01
    W[offset_c + 1, 5] = 1  # C10
    W[offset_c + 2, 5] = 1  # C11
    
    # M7 = (A01-A11)(B10+B11)
    U[offset_a + 1, 6] = 1  # A01
    U[offset_a + 2, 6] = -1  # -A11
    V[offset_b + 2, 6] = 1  # B10
    V[offset_b + 3, 6] = 1  # B11 (row 1, col 1 -> (j0+1)*p + (k0+1))
    W[offset_c + 0, 6] = 1  # C00
    W[offset_c + 2, 6] = 1  # C11
    
    return U, V, W

def _dot_1x2x2(offset_a, offset_b, offset_c, n, m, p):
    """
    Return (U, V, W) for a 1x2x2 block at given offsets.
    offset_a: starting index in A (i*m + j) where i is fixed
    offset_b: starting index in B (j*p + k)
    offset_c: starting index in C (k*n + i) where i is fixed
    
    The 1x2x2 block corresponds to:
    A row [i0], cols [j0, j0+1]
    B rows [j0, j0+1], cols [k0, k0+1]
    C row [i0], cols [k0, k0+1]
    """
    U = np.zeros((n*m, 2))
    V = np.zeros((m*p, 2))
    W = np.zeros((p*n, 2))
    
    # r=0: A0 * (B00, B01)
    U[offset_a + 0, 0] = 1  # A0 (col 0)
    V[offset_b + 0, 0] = 1  # B00
    V[offset_b + 1, 0] = 1  # B01
    W[offset_c + 0, 0] = 1  # C0
    W[offset_c + 1, 0] = 1  # C1
    
    # r=1: A1 * (B10, B11)
    U[offset_a + 1, 1] = 1  # A1 (col 1)
    V[offset_b + 2, 1] = 1  # B10
    V[offset_b + 3, 1] = 1  # B11
    W[offset_c + 0, 1] = 1  # C0
    W[offset_c + 1, 1] = 1  # C1
    
    return U, V, W

def _try_strassen_blocks():
    """
    Try to use Strassen-like block structure for 3x4x8.
    
    3x4x8 = (2+1)x4x8
    = 2x4x8 + 1x4x8
    
    2x4x8 = 2x(2+2)x8 = 2x2x8 + 2x2x8
    Each 2x2x8 = 2x2x(4+4) = 2x2x4 + 2x2x4
    Each 2x2x4 = 2x2x(2+2) = 2x2x2 + 2x2x2
    Each 2x2x2 has rank 7 (Strassen)
    
    1x4x8 = 1x(2+2)x8 = 1x2x8 + 1x2x8
    Each 1x2x8 = 1x2x(4+4) = 1x2x4 + 1x2x4
    Each 1x2x4 = 1x2x(2+2) = 1x2x2 + 1x2x2
    Each 1x2x2 has rank 2
    
    Total: 4*7 + 4*2 = 28 + 8 = 36
    """
    n, m, p = DIMENSIONS  # 3, 4, 8
    
    U_total = np.zeros((n*m, 0))
    V_total = np.zeros((m*p, 0))
    W_total = np.zeros((p*n, 0))
    
    def _add_block(U_block, V_block, W_block):
        nonlocal U_total, V_total, W_total
        U_total = np.hstack([U_total, U_block])
        V_total = np.hstack([V_total, V_block])
        W_total = np.hstack([W_total, W_block])
    
    # Block structure for 3x4x8:
    # A: 3x4, B: 4x8, C: 3x8
    #
    # C[0:2, 0:4] = A[0:2, 0:4] * B[0:4, 0:4]  (2x4x4)
    # C[0:2, 4:8] = A[0:2, 0:4] * B[0:4, 4:8]  (2x4x4)
    # C[2:3, 0:4] = A[2:3, 0:4] * B[0:4, 0:4]  (1x4x4)
    # C[2:3, 4:8] = A[2:3, 0:4] * B[0:4, 4:8]  (1x4x4)
    #
    # 2x4x4 = 2x2x2 + 2x2x2 (splitting the 4-dimension of A into two 2s)
    # 1x4x4 = 1x2x2 + 1x2x2 (splitting the 4-dimension of A into two 2s)
    
    # C[0:2, 0:4]:
    # Part 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    #   A offset: 0 (rows 0-1, cols 0-1)
    #   B offset: 0 (rows 0-1, cols 0-1)
    #   C offset: 0 (rows 0-1, cols 0-1)
    U_b, V_b, W_b = _strassen_2x2x2(0, 0, 0, n, m, p)
    _add_block(U_b, V_b, W_b)
    
    # Part 2: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    #   A offset: 2 (rows 0-1, cols 2-3)
    #   B offset: 4 (rows 2-3, cols 0-1)
    #   C offset: 0 (rows 0-1, cols 0-1)
    U_b, V_b, W_b = _strassen_2x2x2(2, 4, 0, n, m, p)
    _add_block(U_b, V_b, W_b)
    
    # C[0:2, 4:8]:
    # Part 1: A[0:2, 0:2] * B[0:2, 4:6] -> C[0:2, 4:6]
    #   A offset: 0
    #   B offset: 16 (rows 0-1, cols 4-5)
    #   C offset: 4 (rows 0-1, cols 4-5)
    U_b, V_b, W_b = _strassen_2x2x2(0, 16, 4, n, m, p)
    _add_block(U_b, V_b, W_b)
    
    # Part 2: A[0:2, 2:4] * B[2:4, 4:6] -> C[0:2, 4:6]
    #   A offset: 2
    #   B offset: 20 (rows 2-3, cols 4-5)
    #   C offset: 4
    U_b, V_b, W_b = _strassen_2x2x2(2, 20, 4, n, m, p)
    _add_block(U_b, V_b, W_b)
    
    # C[2:3, 0:4]:
    # Part 1: A[2:3, 0:2] * B[0:2, 0:2] -> C[2:3, 0:2]
    #   A offset: 8 (row 2, cols 0-1)
    #   B offset: 0
    #   C offset: 6 (row 2, cols 0-1)
    U_b, V_b, W_b = _dot_1x2x2(8, 0, 6, n, m, p)
    _add_block(U_b, V_b, W_b)
    
    # Part 2: A[2:3, 2:4] * B[2:4, 0:2] -> C[2:3, 0:2]
    #   A offset: 10 (row 2, cols 2-3)
    #   B offset: 4
    #   C offset: 6
    U_b, V_b, W_b = _dot_1x2x2(10, 4, 6, n, m, p)
    _add_block(U_b, V_b, W_b)
    
    # C[2:3, 4:8]:
    # Part 1: A[2:3, 0:2] * B[0:2, 4:6] -> C[2:3, 4:6]
    #   A offset: 8
    #   B offset: 16
    #   C offset: 8 (row 2, cols 4-5)
    U_b, V_b, W_b = _dot_1x2x2(8, 16, 8, n, m, p)
    _add_block(U_b, V_b, W_b)
    
    # Part 2: A[2:3, 2:4] * B[2:4, 4:6] -> C[2:3, 4:6]
    #   A offset: 10
    #   B offset: 20
    #   C offset: 8
    U_b, V_b, W_b = _dot_1x2x2(10, 20, 8, n, m, p)
    _add_block(U_b, V_b, W_b)
    
    # Convert to list format
    U_list = U_total.tolist()
    V_list = V_total.tolist()
    W_list = W_total.tolist()
    
    result = {"U": U_list, "V": V_list, "W": W_list}
    
    if _verify(U_list, V_list, W_list):
        return result, U_total.shape[1]
    else:
        # Fallback to schoolbook
        return _schoolbook(), n * m * p

def _schoolbook():
    n, m, p = DIMENSIONS
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
    n, m, p = DIMENSIONS
    
    # Try the block Strassen decomposition
    try:
        result, rank = _try_strassen_blocks()
        if rank < n * m * p:
            return result
    except Exception:
        pass
    
    # Fallback to schoolbook
    return _schoolbook()
# EVOLVE-BLOCK-END
