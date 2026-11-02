"""Matrix multiplication 3,4,8: rank-minimal exact tensor decomposition."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = 150.0
_SAFETY_MARGIN = 5.0

def _remaining_time():
    return _DEADLINE - _SAFETY_MARGIN - (time.time() - _START_TIME)

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

def _verify(U, V, W):
    """Verify that the decomposition is correct."""
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Check valid entries: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.sum(U[a, :] * V[b, :] * W[c, :])
                if abs(val - 1.0) > 1e-9:
                    return False
    
    # Check invalid entries (where j indices don't match)
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
                    if abs(val) > 1e-9:
                        return False
    
    # Check invalid entries (where i indices don't match in W)
    for i in range(n):
        for i2 in range(n):
            if i2 == i:
                continue
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i2
                    val = np.sum(U[a, :] * V[b, :] * W[c, :])
                    if abs(val) > 1e-9:
                        return False
    
    # Check invalid entries (where k indices don't match in V)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for k2 in range(p):
                    if k2 == k:
                        continue
                    a = i * m + j
                    b = j * p + k2
                    c = k * n + i
                    val = np.sum(U[a, :] * V[b, :] * W[c, :])
                    if abs(val) > 1e-9:
                        return False
    
    return True

def _strassen_2x2x2_block(n, m, p, offset_a, offset_b, offset_c):
    """
    Return (U, V, W) for a 2x2x2 Strassen block at given offsets.
    offset_a: starting index in A (i*m+j)
    offset_b: starting index in B (j*p+k)  
    offset_c: starting index in C (k*n+i)
    
    For a 2x2x2 block with local indices:
    A: local 0,1,2,3 -> global offset_a + [0,1,2,3]
    B: local 0,1,2,3 -> global offset_b + [0,1,2,3]
    C: local 0,1,2,3 -> global offset_c + [0,1,2,3]
    
    Strassen 2x2x2 decomposition:
    C00 = M1 - M4 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M5 - M7
    
    Where:
    M1 = (A00+A11)(B00+B11)
    M2 = (A10+A11)(B00)
    M3 = A00(B01-B11)
    M4 = A11(B10-B00)
    M5 = (A00+A01)B11
    M6 = (A10-A00)(B00+B01)  [unused in standard Strassen, kept for completeness]
    M7 = (A01-A11)(B10+B11)
    """
    U = np.zeros((n * m, 7))
    V = np.zeros((m * p, 7))
    W = np.zeros((p * n, 7))
    
    # Local A indices: 0=A00, 1=A01, 2=A10, 3=A11
    # Local B indices: 0=B00, 1=B01, 2=B10, 3=B11
    # Local C indices: 0=C00, 1=C01, 2=C10, 3=C11
    
    # M1 = (A00+A11)(B00+B11)
    # Contributes: +1 to C00, +1 to C11 (via C00=M1-..., C11=M1-...)
    U[offset_a + 0, 0] = 1
    U[offset_a + 3, 0] = 1
    V[offset_b + 0, 0] = 1
    V[offset_b + 3, 0] = 1
    W[offset_c + 0, 0] = 1   # C00 += M1
    W[offset_c + 3, 0] = 1   # C11 += M1
    
    # M2 = (A10+A11)(B00)
    # Contributes: +1 to C10, -1 to C11
    U[offset_a + 2, 1] = 1
    U[offset_a + 3, 1] = 1
    V[offset_b + 0, 1] = 1
    W[offset_c + 2, 1] = 1   # C10 += M2
    W[offset_c + 3, 1] = -1  # C11 -= M2
    
    # M3 = A00(B01-B11)
    # Contributes: +1 to C01
    U[offset_a + 0, 2] = 1
    V[offset_b + 1, 2] = 1
    V[offset_b + 3, 2] = -1
    W[offset_c + 1, 2] = 1   # C01 += M3
    
    # M4 = A11(B10-B00)
    # Contributes: -1 to C00, +1 to C10
    U[offset_a + 3, 3] = 1
    V[offset_b + 2, 3] = 1
    V[offset_b + 0, 3] = -1
    W[offset_c + 0, 3] = -1  # C00 -= M4
    W[offset_c + 2, 3] = 1   # C10 += M4
    
    # M5 = (A00+A01)B11
    # Contributes: +1 to C01, +1 to C11
    U[offset_a + 0, 4] = 1
    U[offset_a + 1, 4] = 1
    V[offset_b + 3, 4] = 1
    W[offset_c + 1, 4] = 1   # C01 += M5
    W[offset_c + 3, 4] = 1   # C11 += M5
    
    # M6 = (A10-A00)(B00+B01) -- not used in standard Strassen 2x2x2
    # Leave all zeros for this term
    
    # M7 = (A01-A11)(B10+B11)
    # Contributes: +1 to C00, -1 to C11
    U[offset_a + 1, 6] = 1
    U[offset_a + 3, 6] = -1
    V[offset_b + 2, 6] = 1
    V[offset_b + 3, 6] = 1
    W[offset_c + 0, 6] = 1   # C00 += M7
    W[offset_c + 3, 6] = -1  # C11 -= M7
    
    return U, V, W

def _dot_1x2x2_block(n, m, p, offset_a, offset_b, offset_c):
    """
    Return (U, V, W) for a 1x2x2 block (rank 2) at given offsets.
    A is 1x2, B is 2x2, C is 1x2.
    offset_a: starting index in A (j)
    offset_b: starting index in B (j*2+k)
    offset_c: starting index in C (k)
    """
    U = np.zeros((n * m, 2))
    V = np.zeros((m * p, 2))
    W = np.zeros((p * n, 2))
    
    # r=0: A[0] * (B[0,0], B[0,1])
    U[offset_a + 0, 0] = 1
    V[offset_b + 0, 0] = 1  # B[0,0]
    V[offset_b + 1, 0] = 1  # B[0,1]
    W[offset_c + 0, 0] = 1  # C[0]
    W[offset_c + 1, 0] = 1  # C[1]
    
    # r=1: A[1] * (B[1,0], B[1,1])
    U[offset_a + 1, 1] = 1
    V[offset_b + 2, 1] = 1  # B[1,0]
    V[offset_b + 3, 1] = 1  # B[1,1]
    W[offset_c + 0, 1] = 1  # C[0]
    W[offset_c + 1, 1] = 1  # C[1]
    
    return U, V, W

def _try_block_strassen():
    """
    Try block-based Strassen decomposition for 3x4x8.
    
    Strategy:
    - Split A (3x4) into A1 (2x4) and A2 (1x4)
    - Split B (4x8) into B1 (4x4) and B2 (4x4)
    - C = A*B = [A1*B1 | A1*B2; A2*B1 | A2*B2]
    
    For each 2x4x4 block, split into two 2x2x2 Strassen blocks.
    For each 1x4x4 block, split into two 1x2x2 dot product blocks.
    
    Total rank: 4 * 7 (Strassen) + 4 * 2 (dot) = 28 + 8 = 36
    """
    n, m, p = DIMENSIONS  # 3, 4, 8
    
    U_total = np.zeros((n * m, 0))
    V_total = np.zeros((m * p, 0))
    W_total = np.zeros((p * n, 0))
    
    def _add_block(U_block, V_block, W_block):
        nonlocal U_total, V_total, W_total
        U_total = np.hstack([U_total, U_block])
        V_total = np.hstack([V_total, V_block])
        W_total = np.hstack([W_total, W_block])
    
    # Block structure for 3x4x8:
    # A: 3x4 (rows 0-2, cols 0-3)
    # B: 4x8 (rows 0-3, cols 0-7)
    # C: 3x8 (rows 0-2, cols 0-7)
    #
    # C[0:2, 0:4] = A[0:2, 0:4] * B[0:4, 0:4]  (2x4x4)
    # C[0:2, 4:8] = A[0:2, 0:4] * B[0:4, 4:8]  (2x4x4)
    # C[2:3, 0:4] = A[2:3, 0:4] * B[0:4, 0:4]  (1x4x4)
    # C[2:3, 4:8] = A[2:3, 0:4] * B[0:4, 4:8]  (1x4x4)
    
    # C[0:2, 0:4]: A[0:2, 0:4] * B[0:4, 0:4]
    # Split A[0:2, 0:4] into A[0:2, 0:2] and A[0:2, 2:4]
    # Split B[0:4, 0:4] into B[0:2, 0:2] and B[2:4, 0:2]
    
    # Part 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # A offset: 0 (i=0,1; j=0,1)
    # B offset: 0 (j=0,1; k=0,1)
    # C offset: 0 (k=0,1; i=0,1)
    U_b, V_b, W_b = _strassen_2x2x2_block(n, m, p, 0, 0, 0)
    _add_block(U_b, V_b, W_b)
    
    # Part 2: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    # A offset: 2 (i=0,1; j=2,3)
    # B offset: 4 (j=2,3; k=0,1)
    # C offset: 0 (k=0,1; i=0,1)
    U_b, V_b, W_b = _strassen_2x2x2_block(n, m, p, 2, 4, 0)
    _add_block(U_b, V_b, W_b)
    
    # C[0:2, 4:8]: A[0:2, 0:4] * B[0:4, 4:8]
    # Part 1: A[0:2, 0:2] * B[0:2, 4:6] -> C[0:2, 4:6]
    # A offset: 0
    # B offset: 16 (j=0,1; k=4,5)
    # C offset: 4 (k=4,5; i=0,1)
    U_b, V_b, W_b = _strassen_2x2x2_block(n, m, p, 0, 16, 4)
    _add_block(U_b, V_b, W_b)
    
    # Part 2: A[0:2, 2:4] * B[2:4, 4:6] -> C[0:2, 4:6]
    # A offset: 2
    # B offset: 20 (j=2,3; k=4,5)
    # C offset: 4
    U_b, V_b, W_b = _strassen_2x2x2_block(n, m, p, 2, 20, 4)
    _add_block(U_b, V_b, W_b)
    
    # C[2:3, 0:4]: A[2:3, 0:4] * B[0:4, 0:4]
    # Part 1: A[2:3, 0:2] * B[0:2, 0:2] -> C[2:3, 0:2]
    # A offset: 8 (i=2; j=0,1)
    # B offset: 0 (j=0,1; k=0,1)
    # C offset: 6 (k=0,1; i=2)
    U_b, V_b, W_b = _dot_1x2x2_block(n, m, p, 8, 0, 6)
    _add_block(U_b, V_b, W_b)
    
    # Part 2: A[2:3, 2:4] * B[2:4, 0:2] -> C[2:3, 0:2]
    # A offset: 10 (i=2; j=2,3)
    # B offset: 4 (j=2,3; k=0,1)
    # C offset: 6
    U_b, V_b, W_b = _dot_1x2x2_block(n, m, p, 10, 4, 6)
    _add_block(U_b, V_b, W_b)
    
    # C[2:3, 4:8]: A[2:3, 0:4] * B[0:4, 4:8]
    # Part 1: A[2:3, 0:2] * B[0:2, 4:6] -> C[2:3, 4:6]
    # A offset: 8
    # B offset: 16
    # C offset: 8 (k=4,5; i=2)
    U_b, V_b, W_b = _dot_1x2x2_block(n, m, p, 8, 16, 8)
    _add_block(U_b, V_b, W_b)
    
    # Part 2: A[2:3, 2:4] * B[2:4, 4:6] -> C[2:3, 4:6]
    # A offset: 10
    # B offset: 20
    # C offset: 8
    U_b, V_b, W_b = _dot_1x2x2_block(n, m, p, 10, 20, 8)
    _add_block(U_b, V_b, W_b)
    
    # Convert to list format
    U_list = U_total.tolist()
    V_list = V_total.tolist()
    W_list = W_total.tolist()
    
    result = {"U": U_list, "V": V_list, "W": W_list}
    
    if _verify(U_list, V_list, W_list):
        return result, U_total.shape[1]
    else:
        return None, None

def _try_improved_blocks():
    """
    Try alternative block decompositions that might yield lower rank.
    
    Split A as (3x2 | 3x2) and B as (2x8; 2x8)
    - C = A_left * B_top + A_right * B_bottom
    - Each 3x2x8 block has rank 6 (schoolbook for that sub-problem)
    - Total rank: 2 * 6 = 12
    """
    n, m, p = DIMENSIONS  # 3, 4, 8
    
    def _dot_3x2x8_block(offset_a, offset_b, offset_c):
        """
        Return (U, V, W) for a 3x2x8 block (rank 6).
        A is 3x2, B is 2x8, C is 3x8.
        offset_a: starting index in A (i*m+j) -> for this block, i in {0,1,2}, j in {0,1}
        offset_b: starting index in B (j*p+k) -> for this block, j in {0,1}, k in {0,...,7}
        offset_c: starting index in C (k*n+i) -> for this block, k in {0,...,7}, i in {0,1,2}
        """
        U = np.zeros((n * m, 6))
        V = np.zeros((m * p, 6))
        W = np.zeros((p * n, 6))
        
        for i in range(3):
            for j in range(2):
                r = i * 2 + j
                # A index: offset_a + i*m + j
                U[offset_a + i * m + j, r] = 1
                # B indices: offset_b + j*p + k for k=0,...,7
                for k in range(8):
                    V[offset_b + j * p + k, r] = 1
                # C indices: offset_c + k*n + i for k=0,...,7
                for k in range(8):
                    W[offset_c + k * n + i, r] = 1
        
        return U, V, W
    
    U_total = np.zeros((n * m, 0))
    V_total = np.zeros((m * p, 0))
    W_total = np.zeros((p * n, 0))
    
    def _add_block(U_block, V_block, W_block):
        nonlocal U_total, V_total, W_total
        U_total = np.hstack([U_total, U_block])
        V_total = np.hstack([V_total, V_block])
        W_total = np.hstack([W_total, W_block])
    
    # C = A_left * B_top + A_right * B_bottom
    # A_left: A[0:3, 0:2] (3x2), offset_a = 0 (i=0,1,2; j=0,1)
    # A_right: A[0:3, 2:4] (3x2), offset_a = 2 (i=0,1,2; j=2,3)
    # B_top: B[0:2, 0:8] (2x8), offset_b = 0 (j=0,1; k=0,...,7)
    # B_bottom: B[2:4, 0:8] (2x8), offset_b = 16 (j=2,3; k=0,...,7)
    # C: C[0:3, 0:8] (3x8), offset_c = 0
    
    # Part 1: A_left * B_top -> C
    U_b, V_b, W_b = _dot_3x2x8_block(0, 0, 0)
    _add_block(U_b, V_b, W_b)
    
    # Part 2: A_right * B_bottom -> C
    U_b, V_b, W_b = _dot_3x2x8_block(2, 16, 0)
    _add_block(U_b, V_b, W_b)
    
    U_list = U_total.tolist()
    V_list = V_total.tolist()
    W_list = W_total.tolist()
    
    result = {"U": U_list, "V": V_list, "W": W_list}
    
    if _verify(U_list, V_list, W_list):
        return result, U_total.shape[1]
    else:
        return None, None

def _try_hybrid_blocks():
    """
    Try a hybrid approach combining Strassen for 2x2x2 blocks with
    efficient handling of the 3rd row.
    
    Split A as (2x4; 1x4) and B as (4x4 | 4x4).
    Use Strassen for 2x4x4 blocks, and a more efficient method for 1x4x4.
    
    For 1x4x4: A is 1x4, B is 4x4, C is 1x4
    C[k] = sum_j A[j] * B[j,k]
    This is rank 4 (each output is a dot product).
    But can we do better? Probably not for exact decomposition.
    
    So total: 2 * 14 (Strassen for 2x4x4) + 2 * 4 (dot for 1x4x4) = 28 + 8 = 36
    Same as before.
    """
    # This is essentially the same as _try_block_strassen, so skip.
    return None, None

def construct():
    n, m, p = DIMENSIONS
    
    # Try various block decompositions and return the best (lowest rank) valid one
    best_result = None
    best_rank = n * m * p
    
    # Try block Strassen
    try:
        if _remaining_time() > 10:
            result, rank = _try_block_strassen()
            if result is not None and rank < best_rank:
                best_result = result
                best_rank = rank
    except Exception:
        pass
    
    # Try improved blocks
    try:
        if _remaining_time() > 10:
            result, rank = _try_improved_blocks()
            if result is not None and rank < best_rank:
                best_result = result
                best_rank = rank
    except Exception:
        pass
    
    # Fallback to schoolbook
    if best_result is None:
        return _schoolbook()
    
    return best_result
# EVOLVE-BLOCK-END
