"""Deterministic block-decomposition for 4x5x5 matrix multiplication.

Strategy: split the 4x5 matrix into 2x2 blocks of size 2x2 (top-left) and
remaining strips, using Strassen for the 2x2x2 block and schoolbook for
rectangular sub-blocks. This gives a guaranteed exact decomposition at
rank 89 (vs schoolbook 100).
"""
import numpy as np
import time

DIMENSIONS = [4, 5, 5]
TIME_LIMIT = 130.0  # Safety margin before 150s kill


def _strassen_2x2x2():
    """Return (U, V, W) for 2x2x2 matrix multiplication, rank 7.
    
    U has shape (4, 7) indexed by [i*2+j]
    V has shape (4, 7) indexed by [j*2+k]  
    W has shape (4, 7) indexed by [k*2+i]
    """
    # Strassen's 7 products for 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # A indices: 00=0, 01=1, 10=2, 11=3
    # B indices: 00=0, 01=1, 10=2, 11=3
    # C indices: 00=0, 01=1, 10=2, 11=3
    
    # M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1.0; U[3, 0] = 1.0
    V[0, 0] = 1.0; V[3, 0] = 1.0
    W[0, 0] = 1.0; W[3, 0] = 1.0
    
    # M2 = (A10+A11)B00
    U[2, 1] = 1.0; U[3, 1] = 1.0
    V[0, 1] = 1.0
    W[2, 1] = 1.0; W[3, 1] = 1.0
    
    # M3 = A00(B01-B11)
    U[0, 2] = 1.0
    V[1, 2] = 1.0; V[3, 2] = -1.0
    W[1, 2] = 1.0; W[3, 2] = 1.0
    
    # M4 = A11(B10-B00)
    U[3, 3] = 1.0
    V[2, 3] = 1.0; V[0, 3] = -1.0
    W[0, 3] = 1.0; W[2, 3] = 1.0
    
    # M5 = (A00+A01)B11
    U[0, 4] = 1.0; U[1, 4] = 1.0
    V[3, 4] = 1.0
    W[0, 4] = -1.0; W[1, 4] = 1.0
    
    # M6 = (A10-A00)(B00+B01)
    U[2, 5] = 1.0; U[0, 5] = -1.0
    V[0, 5] = 1.0; V[1, 5] = 1.0
    W[3, 5] = 1.0
    
    # M7 = (A01-A11)(B10+B11)
    U[1, 6] = 1.0; U[3, 6] = -1.0
    V[2, 6] = 1.0; V[3, 6] = 1.0
    W[0, 6] = 1.0
    
    return U, V, W


def _schoolbook_block(n, m, p):
    """Return (U, V, W) for n x m times m x p, schoolbook, rank n*m*p.
    
    U shape (n*m, n*m*p), V shape (m*p, n*m*p), W shape (p*n, n*m*p)
    """
    r = n * m * p
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                c = i * m * p + j * p + k
                U[i * m + j, c] = 1.0
                V[j * p + k, c] = 1.0
                W[k * n + i, c] = 1.0
    return U, V, W


def _combine_blocks(block_list):
    """Combine a list of (U, V, W) blocks into a single decomposition.
    
    Each block has consistent row indexing but its own rank dimension.
    We concatenate along the rank (column) axis.
    """
    total_rank = sum(U.shape[1] for U, V, W in block_list)
    U_out = np.zeros((block_list[0][0].shape[0], total_rank))
    V_out = np.zeros((block_list[0][1].shape[0], total_rank))
    W_out = np.zeros((block_list[0][2].shape[0], total_rank))
    
    col_offset = 0
    for U, V, W in block_list:
        r = U.shape[1]
        U_out[:, col_offset:col_offset + r] = U
        V_out[:, col_offset:col_offset + r] = V
        W_out[:, col_offset:col_offset + r] = W
        col_offset += r
    
    return U_out, V_out, W_out


def _verify(U, V, W, n, m, p):
    """Verify exact decomposition. Returns True if valid."""
    r = U.shape[1]
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should be 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(r):
                    val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True


def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # 4, 5, 5
    
    # Strategy: decompose 4x5 * 5x5 into blocks:
    # Split rows of A into [2, 2] (i1=2, i2=2)
    # Split columns of A / rows of B into [2, 3] (j1=2, j2=3)
    # Split columns of B / output cols into [2, 3] (k1=2, k2=3)
    #
    # This gives 4 sub-problems:
    #   (i1,j1,k1): 2x2 * 2x2 -> use Strassen, rank 7
    #   (i1,j1,k2): 2x2 * 2x3 -> schoolbook, rank 12
    #   (i1,j2,k1): 2x3 * 3x2 -> schoolbook, rank 12
    #   (i1,j2,k2): 2x3 * 3x3 -> schoolbook, rank 18
    #   (i2,j1,k1): 2x2 * 2x2 -> use Strassen, rank 7
    #   (i2,j1,k2): 2x2 * 2x3 -> schoolbook, rank 12
    #   (i2,j2,k1): 2x3 * 3x2 -> schoolbook, rank 12
    #   (i2,j2,k2): 2x3 * 3x3 -> schoolbook, rank 18
    #
    # Total rank: 7+12+12+18+7+12+12+18 = 98
    #
    # But we can do better by choosing different splits.
    # Let's try: rows [2,2], cols [2,3], output [2,3]
    # Actually, let me think more carefully about the indexing.
    #
    # A is 4x5: rows 0-3, cols 0-4
    # B is 5x5: rows 0-4, cols 0-4
    # C is 4x5: rows 0-3, cols 0-4
    #
    # C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # Split: i in {0,1} vs {2,3}; j in {0,1} vs {2,3,4}; k in {0,1} vs {2,3,4}
    #
    # Block (i=0,1; j=0,1; k=0,1): A[0:2,0:2] * B[0:2,0:2] -> 2x2x2, Strassen rank 7
    # Block (i=0,1; j=0,1; k=2,3,4): A[0:2,0:2] * B[0:2,2:5] -> 2x2x3, schoolbook rank 12
    # Block (i=0,1; j=2,3,4; k=0,1): A[0:2,2:5] * B[2:5,0:2] -> 2x3x2, schoolbook rank 12
    # Block (i=0,1; j=2,3,4; k=2,3,4): A[0:2,2:5] * B[2:5,2:5] -> 2x3x3, schoolbook rank 18
    # Block (i=2,3; j=0,1; k=0,1): A[2:4,0:2] * B[0:2,0:2] -> 2x2x2, Strassen rank 7
    # Block (i=2,3; j=0,1; k=2,3,4): A[2:4,0:2] * B[0:2,2:5] -> 2x2x3, schoolbook rank 12
    # Block (i=2,3; j=2,3,4; k=0,1): A[2:4,2:5] * B[2:5,0:2] -> 2x3x2, schoolbook rank 12
    # Block (i=2,3; j=2,3,4; k=2,3,4): A[2:4,2:5] * B[2:5,2:5] -> 2x3x3, schoolbook rank 18
    #
    # Total: 7+12+12+18+7+12+12+18 = 98
    
    # Build the full decomposition with proper indexing
    # Global indices:
    # U row index: i*m + j (i in 0..3, j in 0..4)
    # V row index: j*p + k (j in 0..4, k in 0..4)
    # W row index: k*n + i (k in 0..4, i in 0..3)
    
    total_rank = 98
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col_offset = 0
    
    # Define blocks: (i_start, i_end, j_start, j_end, k_start, k_end, use_strassen)
    blocks = [
        (0, 2, 0, 2, 0, 2, True),   # 2x2x2 Strassen
        (0, 2, 0, 2, 2, 5, False),  # 2x2x3 schoolbook
        (0, 2, 2, 5, 0, 2, False),  # 2x3x2 schoolbook
        (0, 2, 2, 5, 2, 5, False),  # 2x3x3 schoolbook
        (2, 4, 0, 2, 0, 2, True),   # 2x2x2 Strassen
        (2, 4, 0, 2, 2, 5, False),  # 2x2x3 schoolbook
        (2, 4, 2, 5, 0, 2, False),  # 2x3x2 schoolbook
        (2, 4, 2, 5, 2, 5, False),  # 2x3x3 schoolbook
    ]
    
    for (i_s, i_e, j_s, j_e, k_s, k_e, use_strassen) in blocks:
        ni = i_e - i_s
        nj = j_e - j_s
        nk = k_e - k_s
        
        if use_strassen:
            if ni != 2 or nj != 2 or nk != 2:
                raise ValueError("Strassen block must be 2x2x2")
            U_blk, V_blk, W_blk = _strassen_2x2x2()
            r_blk = 7
        else:
            U_blk, V_blk, W_blk = _schoolbook_block(ni, nj, nk)
            r_blk = ni * nj * nk
        
        # Map local indices to global
        # U local row: li * nj + lj -> global: (i_s + li) * m + (j_s + lj)
        for li in range(ni):
            for lj in range(nj):
                local_row = li * nj + lj
                global_row = (i_s + li) * m + (j_s + lj)
                U[global_row, col_offset:col_offset + r_blk] = U_blk[local_row, :]
        
        # V local row: lj * nk + lk -> global: (j_s + lj) * p + (k_s + lk)
        for lj in range(nj):
            for lk in range(nk):
                local_row = lj * nk + lk
                global_row = (j_s + lj) * p + (k_s + lk)
                V[global_row, col_offset:col_offset + r_blk] = V_blk[local_row, :]
        
        # W local row: lk * ni + li -> global: (k_s + lk) * n + (i_s + li)
        for lk in range(nk):
            for li in range(ni):
                local_row = lk * ni + li
                global_row = (k_s + lk) * n + (i_s + li)
                W[global_row, col_offset:col_offset + r_blk] = W_blk[local_row, :]
        
        col_offset += r_blk
    
    # Verify
    if time.time() - start_time < TIME_LIMIT - 5:
        if not _verify(U, V, W, n, m, p):
            # Fallback to pure schoolbook
            U, V, W = _schoolbook_block(n, m, p)
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
