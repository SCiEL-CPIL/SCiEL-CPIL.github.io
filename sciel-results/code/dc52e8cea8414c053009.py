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
    """Verify that the decomposition is correct by checking all tensor entries."""
    n, m, p = DIMENSIONS
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    
    # Build the expected tensor
    # T[i*m+j, j*p+k, k*n+i] = 1, all else 0
    # Check all entries
    for i in range(n):
        for j in range(m):
            a = i * m + j
            for j2 in range(m):
                for k in range(p):
                    b = j2 * p + k
                    for i2 in range(n):
                        for k2 in range(p):
                            c = k2 * n + i2
                            expected = 1.0 if (j2 == j and i2 == i and k2 == k) else 0.0
                            val = np.dot(U_arr[a, :], V_arr[b, :] * W_arr[c, :])
                            if abs(val - expected) > 1e-6:
                                return False
    return True

def _strassen_2x2x2(offset_a, offset_b, offset_c, n, m, p):
    """
    Return (U, V, W) for a 2x2x2 block at given offsets using Strassen.
    
    offset_a: offset in A index space (i*m + j)
    offset_b: offset in B index space (j*p + k)
    offset_c: offset in C index space (k*n + i)
    
    For a 2x2x2 block:
    - A has 4 entries: indices 0,1,2,3 (local)
    - B has 4 entries: indices 0,1,2,3 (local)
    - C has 4 entries: indices 0,1,2,3 (local)
    
    Strassen decomposition:
    M1 = (A00+A11)(B00+B11)
    M2 = (A10+A11)B00
    M3 = A00(B01-B11)
    M4 = A11(B10-B00)
    M5 = (A00+A01)B11
    M6 = (A10-A00)(B00+B01)
    M7 = (A01-A11)(B10+B11)
    
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1  # A00
    U[3, 0] = 1  # A11
    V[0, 0] = 1  # B00
    V[3, 0] = 1  # B11
    W[0, 0] = 1  # C00
    W[3, 0] = 1  # C11
    
    # M2 = (A10+A11)B00
    U[2, 1] = 1  # A10
    U[3, 1] = 1  # A11
    V[0, 1] = 1  # B00
    W[0, 1] = -1  # C00
    W[2, 1] = 1  # C10
    W[3, 1] = -1  # C11
    
    # M3 = A00(B01-B11)
    U[0, 2] = 1  # A00
    V[1, 2] = 1  # B01
    V[3, 2] = -1  # -B11
    W[1, 2] = 1  # C01
    W[3, 2] = 1  # C11
    
    # M4 = A11(B10-B00)
    U[3, 3] = 1  # A11
    V[2, 3] = 1  # B10
    V[0, 3] = -1  # -B00
    W[0, 3] = 1  # C00
    W[2, 3] = 1  # C10
    
    # M5 = (A00+A01)B11
    U[0, 4] = 1  # A00
    U[1, 4] = 1  # A01
    V[3, 4] = 1  # B11
    W[1, 4] = 1  # C01
    W[3, 4] = -1  # C11
    
    # M6 = (A10-A00)(B00+B01)
    U[2, 5] = 1  # A10
    U[0, 5] = -1  # -A00
    V[0, 5] = 1  # B00
    V[1, 5] = 1  # B01
    W[3, 5] = 1  # C11
    
    # M7 = (A01-A11)(B10+B11)
    U[1, 6] = 1  # A01
    U[3, 6] = -1  # -A11
    V[2, 6] = 1  # B10
    V[3, 6] = 1  # B11
    W[0, 6] = 1  # C00
    
    # Now map to full-size matrices with offsets
    U_full = np.zeros((n*m, 7))
    V_full = np.zeros((m*p, 7))
    W_full = np.zeros((p*n, 7))
    
    # A local index -> global: offset_a + local
    # For 2x2x2, A local indices are 0,1,2,3 corresponding to (i,j) = (0,0),(0,1),(1,0),(1,1)
    # In the full matrix, these map to:
    # (0,0) -> i=0, j=0 -> global index = offset_a
    # (0,1) -> i=0, j=1 -> global index = offset_a + 1
    # (1,0) -> i=1, j=0 -> global index = offset_a + m  (since i increases by 1, j resets)
    # (1,1) -> i=1, j=1 -> global index = offset_a + m + 1
    
    # Wait, I need to be more careful. The offset_a should already account for the row offset.
    # Let me redefine: offset_a is the base index for A[0,0] in this block.
    # Then:
    # A[0,0] -> offset_a
    # A[0,1] -> offset_a + 1
    # A[1,0] -> offset_a + m
    # A[1,1] -> offset_a + m + 1
    
    a_map = [offset_a, offset_a + 1, offset_a + m, offset_a + m + 1]
    
    # For B: offset_b is the base index for B[0,0] in this block.
    # B local indices: 0,1,2,3 corresponding to (j,k) = (0,0),(0,1),(1,0),(1,1)
    # B[0,0] -> offset_b
    # B[0,1] -> offset_b + 1
    # B[1,0] -> offset_b + p
    # B[1,1] -> offset_b + p + 1
    
    b_map = [offset_b, offset_b + 1, offset_b + p, offset_b + p + 1]
    
    # For C: offset_c is the base index for C[0,0] in this block.
    # C local indices: 0,1,2,3 corresponding to (k,i) = (0,0),(0,1),(1,0),(1,1)
    # But the W convention is W[k*n + i], so:
    # C[0,0] -> k=0, i=0 -> offset_c
    # C[0,1] -> k=0, i=1 -> offset_c + 1
    # C[1,0] -> k=1, i=0 -> offset_c + n
    # C[1,1] -> k=1, i=1 -> offset_c + n + 1
    
    c_map = [offset_c, offset_c + 1, offset_c + n, offset_c + n + 1]
    
    for r in range(7):
        for local_idx in range(4):
            if U[local_idx, r] != 0:
                global_idx = a_map[local_idx]
                if 0 <= global_idx < n*m:
                    U_full[global_idx, r] = U[local_idx, r]
            if V[local_idx, r] != 0:
                global_idx = b_map[local_idx]
                if 0 <= global_idx < m*p:
                    V_full[global_idx, r] = V[local_idx, r]
            if W[local_idx, r] != 0:
                global_idx = c_map[local_idx]
                if 0 <= global_idx < p*n:
                    W_full[global_idx, r] = W[local_idx, r]
    
    return U_full, V_full, W_full

def _dot_1x2x2(offset_a, offset_b, offset_c, n, m, p):
    """
    Return (U, V, W) for a 1x2x2 block at given offsets.
    
    1x2x2: A is 1x2, B is 2x2, C is 1x2
    C[k] = A[0]B[0,k] + A[1]B[1,k]
    
    Rank 2 decomposition:
    r=0: A[0] * (B[0,0], B[0,1]) -> (C[0], C[1])
    r=1: A[1] * (B[1,0], B[1,1]) -> (C[0], C[1])
    """
    U = np.zeros((2, 2))
    V = np.zeros((4, 2))
    W = np.zeros((2, 2))
    
    # r=0: A[0] * B[0,:]
    U[0, 0] = 1
    V[0, 0] = 1  # B[0,0]
    V[1, 0] = 1  # B[0,1]
    W[0, 0] = 1  # C[0]
    W[1, 0] = 1  # C[1]
    
    # r=1: A[1] * B[1,:]
    U[1, 1] = 1
    V[2, 1] = 1  # B[1,0]
    V[3, 1] = 1  # B[1,1]
    W[0, 1] = 1  # C[0]
    W[1, 1] = 1  # C[1]
    
    U_full = np.zeros((n*m, 2))
    V_full = np.zeros((m*p, 2))
    W_full = np.zeros((p*n, 2))
    
    # A local: 0,1 -> A[0,0], A[0,1]
    # offset_a is A[0,0]
    a_map = [offset_a, offset_a + 1]
    
    # B local: 0,1,2,3 -> B[0,0], B[0,1], B[1,0], B[1,1]
    # offset_b is B[0,0]
    b_map = [offset_b, offset_b + 1, offset_b + p, offset_b + p + 1]
    
    # C local: 0,1 -> C[0,0], C[0,1]
    # offset_c is C[0,0]
    c_map = [offset_c, offset_c + 1]
    
    for r in range(2):
        for local_idx in range(2):
            if U[local_idx, r] != 0:
                global_idx = a_map[local_idx]
                if 0 <= global_idx < n*m:
                    U_full[global_idx, r] = U[local_idx, r]
        for local_idx in range(4):
            if V[local_idx, r] != 0:
                global_idx = b_map[local_idx]
                if 0 <= global_idx < m*p:
                    V_full[global_idx, r] = V[local_idx, r]
        for local_idx in range(2):
            if W[local_idx, r] != 0:
                global_idx = c_map[local_idx]
                if 0 <= global_idx < p*n:
                    W_full[global_idx, r] = W[local_idx, r]
    
    return U_full, V_full, W_full

def _try_strassen_blocks():
    """
    Construct a block-based Strassen decomposition for 3x4x8.
    
    Strategy:
    - Split A (3x4) into A1 (2x4) and A2 (1x4)
    - Split B (4x8) into B1 (4x4) and B2 (4x4)
    - C = A*B has blocks:
      C[0:2, 0:4] = A1 * B1  (2x4x4)
      C[0:2, 4:8] = A1 * B2  (2x4x4)
      C[2:3, 0:4] = A2 * B1  (1x4x4)
      C[2:3, 4:8] = A2 * B2  (1x4x4)
    
    For each 2x4x4 block:
      Split A1 (2x4) into A1a (2x2) and A1b (2x2)
      Split B (4x4) into two 4x2 blocks
      Each 2x2x2 uses Strassen (rank 7)
      Total: 14 per 2x4x4 block
    
    For each 1x4x4 block:
      Split A2 (1x4) into two 1x2 blocks
      Split B (4x4) into two 4x2 blocks
      Each 1x2x2 has rank 2
      Total: 4 per 1x4x4 block
    
    Total rank: 2*14 + 2*4 = 28 + 8 = 36
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
    
    # Block C[0:2, 0:4] = A[0:2, :] * B[:, 0:4]
    # Split into:
    #   A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    #   A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    
    # Part 1: A[0:2, 0:2] * B[0:2, 0:2]
    # A offset: i=0, j=0 -> 0*4+0 = 0
    # B offset: j=0, k=0 -> 0*8+0 = 0
    # C offset: k=0, i=0 -> 0*3+0 = 0
    U_b, V_b, W_b = _strassen_2x2x2(0, 0, 0, n, m, p)
    _add_block(U_b, V_b, W_b)
    
    # Part 2: A[0:2, 2:4] * B[2:4, 0:2]
    # A offset: i=0, j=2 -> 0*4+2 = 2
    # B offset: j=2, k=0 -> 2*8+0 = 16
    # C offset: k=0, i=0 -> 0*3+0 = 0
    U_b, V_b, W_b = _strassen_2x2x2(2, 16, 0, n, m, p)
    _add_block(U_b, V_b, W_b)
    
    # Block C[0:2, 4:8] = A[0:2, :] * B[:, 4:8]
    # Split into:
    #   A[0:2, 0:2] * B[0:2, 4:6] -> C[0:2, 4:6]
    #   A[0:2, 2:4] * B[2:4, 4:6] -> C[0:2, 4:6]
    
    # Part 1: A[0:2, 0:2] * B[0:2, 4:6]
    # A offset: 0
    # B offset: j=0, k=4 -> 0*8+4 = 4
    # C offset: k=4, i=0 -> 4*3+0 = 12
    U_b, V_b, W_b = _strassen_2x2x2(0, 4, 12, n, m, p)
    _add_block(U_b, V_b, W_b)
    
    # Part 2: A[0:2, 2:4] * B[2:4, 4:6]
    # A offset: 2
    # B offset: j=2, k=4 -> 2*8+4 = 20
    # C offset: k=4, i=0 -> 12
    U_b, V_b, W_b = _strassen_2x2x2(2, 20, 12, n, m, p)
    _add_block(U_b, V_b, W_b)
    
    # Block C[2:3, 0:4] = A[2:3, :] * B[:, 0:4]
    # Split into:
    #   A[2:3, 0:2] * B[0:2, 0:2] -> C[2:3, 0:2]
    #   A[2:3, 2:4] * B[2:4, 0:2] -> C[2:3, 0:2]
    
    # Part 1: A[2:3, 0:2] * B[0:2, 0:2]
    # A offset: i=2, j=0 -> 2*4+0 = 8
    # B offset: j=0, k=0 -> 0
    # C offset: k=0, i=2 -> 0*3+2 = 2
    U_b, V_b, W_b = _dot_1x2x2(8, 0, 2, n, m, p)
    _add_block(U_b, V_b, W_b)
    
    # Part 2: A[2:3, 2:4] * B[2:4, 0:2]
    # A offset: i=2, j=2 -> 2*4+2 = 10
    # B offset: j=2, k=0 -> 16
    # C offset: k=0, i=2 -> 2
    U_b, V_b, W_b = _dot_1x2x2(10, 16, 2, n, m, p)
    _add_block(U_b, V_b, W_b)
    
    # Block C[2:3, 4:8] = A[2:3, :] * B[:, 4:8]
    # Split into:
    #   A[2:3, 0:2] * B[0:2, 4:6] -> C[2:3, 4:6]
    #   A[2:3, 2:4] * B[2:4, 4:6] -> C[2:3, 4:6]
    
    # Part 1: A[2:3, 0:2] * B[0:2, 4:6]
    # A offset: 8
    # B offset: j=0, k=4 -> 4
    # C offset: k=4, i=2 -> 4*3+2 = 14
    U_b, V_b, W_b = _dot_1x2x2(8, 4, 14, n, m, p)
    _add_block(U_b, V_b, W_b)
    
    # Part 2: A[2:3, 2:4] * B[2:4, 4:6]
    # A offset: 10
    # B offset: j=2, k=4 -> 20
    # C offset: k=4, i=2 -> 14
    U_b, V_b, W_b = _dot_1x2x2(10, 20, 14, n, m, p)
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

def _coppersmith_4x4x4():
    """
    Return (U, V, W) for 4x4x4 matrix multiplication using Coppersmith's rank-49 decomposition.
    
    The 4x4x4 multiplication tensor can be decomposed into 49 rank-1 terms.
    This is based on the work of Coppersmith et al. (1987).
    
    For the 4x4x4 case, we use a recursive Strassen-like approach:
    Split 4x4 into four 2x2 blocks.
    Use Strassen's 7 multiplications for each pair of 2x2 blocks.
    
    Actually, let's use a simpler approach: decompose 4x4x4 as a tensor product
    of two 2x2x2 tensors. The 4x4x4 tensor is isomorphic to (2x2x2) ⊗ (2x2x2),
    so its rank is at most 7 * 7 = 49.
    """
    n4, m4, p4 = 4, 4, 4
    
    # Build the 4x4x4 decomposition as a tensor product of two 2x2x2 Strassen decompositions
    # First, get the 2x2x2 Strassen decomposition
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # M1 = (A00+A11)(B00+B11)
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1
    
    # M2 = (A10+A11)B00
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[0, 1] = -1; W2[2, 1] = 1; W2[3, 1] = -1
    
    # M3 = A00(B01-B11)
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[1, 2] = 1; W2[3, 2] = 1
    
    # M4 = A11(B10-B00)
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1; W2[2, 3] = 1
    
    # M5 = (A00+A01)B11
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[1, 4] = 1; W2[3, 4] = -1
    
    # M6 = (A10-A00)(B00+B01)
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1
    
    # Now build the 4x4x4 decomposition using tensor product
    # The 4x4x4 tensor T[i*m+j, j*p+k, k*n+l] where i,j,k,l in {0,1}
    # We can view this as a tensor product of two 2x2x2 tensors
    
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    # For each pair of 2x2x2 Strassen terms (r1, r2), we get a 4x4x4 term
    for r1 in range(7):
        for r2 in range(7):
            r4 = r1 * 7 + r2
            
            # A index: (i1, j1) for first 2x2, (i2, j2) for second 2x2
            # Local A index in 4x4: i4 = i1*2 + i2, j4 = j1*2 + j2
            # Flat index: i4*4 + j4 = (i1*2+i2)*4 + (j1*2+j2)
            
            # B index: (j1, k1) for first, (j2, k2) for second
            # Flat index: (j1*2+j2)*4 + (k1*2+k2)
            
            # C index: (k1, i1) for first, (k2, i2) for second  
            # Wait, need to be careful with the W convention: W[k*n+i]
            # For 4x4x4: W[k*4+i]
            # Local C index: k4 = k1*2 + k2, i4 = i1*2 + i2
            # Flat index: k4*4 + i4 = (k1*2+k2)*4 + (i1*2+i2)
            
            for a1 in range(4):
                if U2[a1, r1] != 0:
                    i1 = a1 // 2
                    j1 = a1 % 2
                    for a2 in range(4):
                        if U2[a2, r2] != 0:
                            i2 = a2 // 2
                            j2 = a2 % 2
                            a4 = (i1*2 + i2) * 4 + (j1*2 + j2)
                            U4[a4, r4] += U2[a1, r1] * U2[a2, r2]
            
            for b1 in range(4):
                if V2[b1, r1] != 0:
                    j1 = b1 // 2
                    k1 = b1 % 2
                    for b2 in range(4):
                        if V2[b2, r2] != 0:
                            j2 = b2 // 2
                            k2 = b2 % 2
                            b4 = (j1*2 + j2) * 4 + (k1*2 + k2)
                            V4[b4, r4] += V2[b1, r1] * V2[b2, r2]
            
            for c1 in range(4):
                if W2[c1, r1] != 0:
                    k1 = c1 // 2
                    i1 = c1 % 2
                    for c2 in range(4):
                        if W2[c2, r2] != 0:
                            k2 = c2 // 2
                            i2 = c2 % 2
                            c4 = (k1*2 + k2) * 4 + (i1*2 + i2)
                            W4[c4, r4] += W2[c1, r1] * W2[c2, r2]
    
    return U4, V4, W4

def _try_padding_projection():
    """
    Embed 3x4x8 into 4x4x8, decompose as (4x4x4) ⊗ (1x1x2), project down.
    
    The 4x4x8 tensor can be seen as two 4x4x4 tensors:
    - First half: B columns 0-3
    - Second half: B columns 4-7
    
    So 4x4x8 = (4x4x4) ⊕ (4x4x4), rank at most 49 + 49 = 98.
    
    Then we project from 4x4x8 to 3x4x8 by:
    - Zeroing out row 3 of A (i=3)
    - Zeroing out row 3 of C (i=3)
    
    This gives a valid 3x4x8 decomposition with rank 98.
    """
    n, m, p = DIMENSIONS  # 3, 4, 8
    
    # Get the 4x4x4 decomposition
    U4, V4, W4 = _coppersmith_4x4x4()
    
    # Build 4x4x8 as two copies of 4x4x4
    # First 4x4x4: B columns 0-3, C columns 0-3
    # Second 4x4x4: B columns 4-7, C columns 4-7
    
    U_4x4x8 = np.zeros((16, 98))
    V_4x4x8 = np.zeros((32, 98))
    W_4x4x8 = np.zeros((32, 98))
    
    # First half: B[:, 0:4], C[:, 0:4]
    U_4x4x8[:, 0:49] = U4
    V_4x4x8[0:16, 0:49] = V4  # B[0:4, 0:4]
    W_4x4x8[0:16, 0:49] = W4  # C[0:4, 0:4]
    
    # Second half: B[:, 4:8], C[:, 4:8]
    U_4x4x8[:, 49:98] = U4
    V_4x4x8[16:32, 49:98] = V4  # B[0:4, 4:8]
    W_4x4x8[16:32, 49:98] = W4  # C[4:8, 0:4]
    
    # Now project to 3x4x8
    # A: 3x4 -> keep rows 0-2 (i=0,1,2), drop row 3 (i=3)
    # B: 4x8 -> keep all rows (j=0,1,2,3), keep all columns (k=0..7)
    # C: 8x3 -> keep rows 0-7 (k=0..7), keep columns 0-2 (i=0,1,2), drop column 3 (i=3)
    
    # A projection: from 4x4 to 3x4
    # A index: i*4 + j, where i in {0,1,2}, j in {0,1,2,3}
    # Keep: i=0,1,2 -> indices 0-3, 4-7, 8-11
    a_keep = []
    for i in range(3):
        for j in range(4):
            a_keep.append(i * 4 + j)
    
    U_3x4x8 = U_4x4x8[a_keep, :]
    
    # B projection: from 4x8 to 4x8 (keep all)
    # B index: j*8 + k, where j in {0,1,2,3}, k in {0..7}
    # Keep all
    b_keep = list(range(32))
    V_3x4x8 = V_4x4x8[b_keep, :]
    
    # C projection: from 8x4 to 8x3
    # C index: k*4 + i, where k in {0..7}, i in {0,1,2}
    # Keep: i=0,1,2
    c_keep = []
    for k in range(8):
        for i in range(3):
            c_keep.append(k * 4 + i)
    
    W_3x4x8 = W_4x4x8[c_keep, :]
    
    # Convert to lists
    U_list = U_3x4x8.tolist()
    V_list = V_3x4x8.tolist()
    W_list = W_3x4x8.tolist()
    
    result = {"U": U_list, "V": V_list, "W": W_list}
    
    if _verify(U_list, V_list, W_list):
        return result, U_3x4x8.shape[1]
    else:
        return None, None

def construct():
    n, m, p = DIMENSIONS
    
    best_result = None
    best_rank = n * m * p
    
    # Try the block Strassen decomposition
    try:
        if _remaining_time() > 10:
            result, rank = _try_strassen_blocks()
            if rank < best_rank:
                best_rank = rank
                best_result = result
    except Exception:
        pass
    
    # Try padding and projection
    try:
        if _remaining_time() > 10:
            result, rank = _try_padding_projection()
            if result is not None and rank < best_rank:
                best_rank = rank
                best_result = result
    except Exception:
        pass
    
    if best_result is not None:
        return best_result
    
    # Fallback to schoolbook
    return _schoolbook()
# EVOLVE-BLOCK-END
