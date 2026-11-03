import time
import numpy as np

DIMENSIONS = [3, 5, 5]

def _schoolbook(n, m, p):
    """Return (U, V, W) for schoolbook decomposition of n x m times m x p."""
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
    return U, V, W

def _strassen_2x2x2():
    """Return (U, V, W) for Strassen decomposition of 2x2 matrix multiplication.
    U has shape (4, 7), V has shape (4, 7), W has shape (4, 7).
    Convention: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c] with n=m=p=2.
    W[k*2+i, c] -> W[i + 2*k, c]
    """
    n = m = p = 2
    rank = 7
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen factors:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B10 - B00)
    # M4 = A11(B11 - B01)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # U factors (for A):
    # A00 at (0,0), A01 at (0,1), A10 at (1,0), A11 at (1,1)
    # U index: i*m + j
    U[0, 0] = 1  # A00 in M1
    U[0, 2] = 1  # A00 in M3
    U[0, 4] = 1  # A00 in M5
    U[0, 5] = -1 # A00 in M6 (negative)
    
    U[1, 4] = 1  # A01 in M5
    U[1, 6] = 1  # A01 in M7
    
    U[2, 1] = 1  # A10 in M2
    U[2, 5] = 1  # A10 in M6
    
    U[3, 0] = 1  # A11 in M1
    U[3, 1] = 1  # A11 in M2
    U[3, 3] = 1  # A11 in M4
    U[3, 6] = -1 # A11 in M7 (negative)
    
    # V factors (for B):
    # B00 at (0,0), B01 at (0,1), B10 at (1,0), B11 at (1,1)
    # V index: j*p + k
    V[0, 0] = 1  # B00 in M1
    V[0, 1] = 1  # B00 in M2
    V[0, 2] = -1 # B00 in M3 (negative)
    V[0, 5] = 1  # B00 in M6
    
    V[1, 3] = -1 # B01 in M4 (negative)
    V[1, 5] = 1  # B01 in M6
    
    V[2, 2] = 1  # B10 in M3
    V[2, 6] = 1  # B10 in M7
    
    V[3, 0] = 1  # B11 in M1
    V[3, 3] = 1  # B11 in M4
    V[3, 4] = 1  # B11 in M5
    V[3, 6] = 1  # B11 in M7
    
    # W factors (for C):
    # C00 at (0,0), C01 at (0,1), C10 at (1,0), C11 at (1,1)
    # W index: k*n + i = k*2 + i
    # C00: W[0*2+0, :] = W[0, :]
    # C01: W[0*2+1, :] = W[1, :]
    # C10: W[1*2+0, :] = W[2, :]
    # C11: W[1*2+1, :] = W[3, :]
    
    # C00 = M1 + M4 - M5 + M7
    W[0, 0] = 1  # M1
    W[0, 3] = 1  # M4
    W[0, 4] = -1 # M5
    W[0, 6] = 1  # M7
    
    # C01 = M3 + M5
    W[1, 2] = 1  # M3
    W[1, 4] = 1  # M5
    
    # C10 = M2 + M4
    W[2, 1] = 1  # M2
    W[2, 3] = 1  # M4
    
    # C11 = M1 - M2 + M3 + M6
    W[3, 0] = 1  # M1
    W[3, 1] = -1 # M2
    W[3, 2] = 1  # M3
    W[3, 5] = 1  # M6
    
    return U, V, W

def _combine_block_decomposition(blocks, n_total, m_total, p_total):
    """
    Combine block decomposition factors.
    blocks: list of (n_i, m_i, p_i, U_i, V_i, W_i)
    Each block corresponds to a sub-matrix multiplication.
    """
    total_rank = sum(b[3].shape[1] for b in blocks)
    U = np.zeros((n_total * m_total, total_rank))
    V = np.zeros((m_total * p_total, total_rank))
    W = np.zeros((p_total * n_total, total_rank))
    
    col_offset = 0
    row_offset_U = 0  # i offset in n dimension
    col_offset_U = 0  # j offset in m dimension
    row_offset_V = 0  # j offset in m dimension
    col_offset_V = 0  # k offset in p dimension
    row_offset_W = 0  # k offset in p dimension
    col_offset_W = 0  # i offset in n dimension
    
    for (n_i, m_i, p_i, U_i, V_i, W_i) in blocks:
        r_i = U_i.shape[1]
        # Place U_i
        U[row_offset_U * m_total + col_offset_U : (row_offset_U + n_i) * m_total + col_offset_U + m_i, 
          col_offset : col_offset + r_i] = U_i
        # Place V_i
        V[row_offset_V * p_total + col_offset_V : (row_offset_V + m_i) * p_total + col_offset_V + p_i, 
          col_offset : col_offset + r_i] = V_i
        # Place W_i: W index is k*n_total + i
        # For sub-block, k ranges 0..p_i-1, i ranges 0..n_i-1
        # Global k = k + k_offset, global i = i + i_offset
        # W global index = (k + k_offset) * n_total + (i + i_offset)
        # W_i local index = k * n_i + i
        # We need to map local (k, i) to global
        for k in range(p_i):
            for i in range(n_i):
                local_idx = k * n_i + i
                global_idx = (row_offset_W + k) * n_total + (col_offset_W + i)
                W[global_idx, col_offset : col_offset + r_i] = W_i[local_idx, :]
        
        col_offset += r_i
        row_offset_U += n_i
        col_offset_U += m_i
        row_offset_V += m_i
        col_offset_V += p_i
        row_offset_W += p_i
        col_offset_W += n_i
    
    return U, V, W

def _verify(U, V, W, n, m, p):
    """Verify that sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = delta(i,k) for all i,j,k."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for c in range(rank):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                expected = 1 if i == k else 0
                if abs(val - expected) > 1e-9:
                    return False
    return True

def _try_block_split_3_2():
    """
    Split n=3, m=5, p=5 as:
    - Split m into 3+2, p into 3+2
    - Block (3,3,3): use Strassen-like or schoolbook for 3x3x3
    - Other blocks: schoolbook
    
    Actually, let's try:
    n=3, m=3+2, p=3+2
    Blocks:
    1. n=3, m=3, p=3
    2. n=3, m=3, p=2
    3. n=3, m=2, p=3
    4. n=3, m=2, p=2
    """
    n = 3
    m1, m2 = 3, 2
    p1, p2 = 3, 2
    
    # For the 3x3x3 block, we can try to use a known rank-23 decomposition
    # For now, let's use schoolbook for all and see if we can improve
    
    # Let's just use schoolbook for all blocks to get a valid baseline
    # Then we can try to replace the 3x3x3 block
    
    blocks = []
    
    # Block 1: 3x3x3
    U1, V1, W1 = _schoolbook(3, 3, 3)
    blocks.append((3, 3, 3, U1, V1, W1))
    
    # Block 2: 3x3x2
    U2, V2, W2 = _schoolbook(3, 3, 2)
    blocks.append((3, 3, 2, U2, V2, W2))
    
    # Block 3: 3x2x3
    U3, V3, W3 = _schoolbook(3, 2, 3)
    blocks.append((3, 2, 3, U3, V3, W3))
    
    # Block 4: 3x2x2
    U4, V4, W4 = _schoolbook(3, 2, 2)
    blocks.append((3, 2, 2, U4, V4, W4))
    
    U, V, W = _combine_block_decomposition(blocks, n, m1+m2, p1+p2)
    
    # Verify
    if not _verify(U, V, W, n, m1+m2, p1+p2):
        return None
    
    rank = U.shape[1]
    return U, V, W, rank

def construct():
    start_time = time.time()
    deadline = start_time + 100  # Leave 20s margin for safety
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    U_school, V_school, W_school = _schoolbook(n, m, p)
    best_rank = U_school.shape[1]
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    
    # Try block decomposition
    try:
        result = _try_block_split_3_2()
        if result is not None:
            U, V, W, rank = result
            if rank < best_rank:
                best_rank = rank
                best_result = {"U": U, "V": V, "W": W}
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline:
        return best_result
    
    return best_result

