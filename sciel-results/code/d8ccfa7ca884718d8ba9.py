"""Matrix multiplication 2x4 x 4x4 tensor decomposition (rank 28 via Strassen on 2x2 blocks)."""

import time
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START

def _build_strassen_rank28():
    """
    Builds the rank-28 Strassen decomposition for 2x4 * 4x4 matrix multiplication.
    
    Tensor convention: T[i,k,j] = 1 for all i in [0,n), k in [0,m), j in [0,p)
    Decomposition: T[i,k,j] = sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r]
    
    n=2, m=4, p=4.
    U shape: (8, 28), V shape: (16, 28), W shape: (8, 28).
    
    Split A (2x4) into two 2x2 column blocks: A_L (cols 0,1), A_R (cols 2,3).
    Split B (4x4) into four 2x2 blocks by rows and cols:
      B_TL (rows 0,1; cols 0,1), B_TR (rows 0,1; cols 2,3)
      B_BL (rows 2,3; cols 0,1), B_BR (rows 2,3; cols 2,3)
    
    C (2x4) splits into C_L (cols 0,1) and C_R (cols 2,3).
    
    C_L = A_L * B_TL + A_R * B_BL  (each product is 2x2 * 2x2, rank 7 each)
    C_R = A_L * B_TR + A_R * B_BR  (each product is 2x2 * 2x2, rank 7 each)
    
    Total rank = 4 * 7 = 28.
    """
    n, m, p = DIMENSIONS
    rank = 28
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Strassen 2x2 * 2x2 -> 2x2:
    # X = [[x00, x01], [x10, x11]], Y = [[y00, y01], [y10, y11]]
    # M1 = (x00+x11)*(y00+y11)
    # M2 = (x10+x11)*y00
    # M3 = x00*(y01-y11)
    # M4 = x11*(y10-y00)
    # M5 = (x00+x01)*y11
    # M6 = (x10-x00)*(y00+y01)
    # M7 = (x01-x11)*(y10+y11)
    # P00 = M1 + M4 - M5 + M7
    # P01 = M3 + M5
    # P10 = M2 + M4
    # P11 = M1 - M2 + M3 + M6
    
    # For each Strassen term s (0..6), define coefficients on X, Y, P sides.
    # X indices: 0=x00, 1=x01, 2=x10, 3=x11
    # Y indices: 0=y00, 1=y01, 2=y10, 3=y11
    # P indices: 0=p00, 1=p01, 2=p10, 3=p11
    
    U_coeffs = [
        {0: 1, 3: 1},       # M1: x00 + x11
        {2: 1, 3: 1},       # M2: x10 + x11
        {0: 1},             # M3: x00
        {3: 1},             # M4: x11
        {0: 1, 1: 1},       # M5: x00 + x01
        {2: 1, 0: -1},      # M6: x10 - x00
        {1: 1, 3: -1},      # M7: x01 - x11
    ]
    
    V_coeffs = [
        {0: 1, 3: 1},       # M1: y00 + y11
        {0: 1},             # M2: y00
        {1: 1, 3: -1},      # M3: y01 - y11
        {2: 1, 0: -1},      # M4: y10 - y00
        {3: 1},             # M5: y11
        {0: 1, 1: 1},       # M6: y00 + y01
        {2: 1, 3: 1},       # M7: y10 + y11
    ]
    
    W_coeffs = [
        {0: 1, 3: 1},       # M1: p00 + p11
        {2: 1, 3: -1},      # M2: p10 - p11
        {1: 1, 3: 1},       # M3: p01 + p11
        {0: 1, 2: 1},       # M4: p00 + p10
        {0: -1, 1: 1},      # M5: -p00 + p01
        {3: 1},             # M6: p11
        {0: 1},             # M7: p00
    ]
    
    # 4 blocks: (a_col_block, b_row_block, b_col_block, c_col_block)
    # a_col_block: 0 -> A cols [0,1], 1 -> A cols [2,3]
    # b_row_block: 0 -> B rows [0,1], 1 -> B rows [2,3]
    # b_col_block: 0 -> B cols [0,1], 1 -> B cols [2,3]
    # c_col_block: 0 -> C cols [0,1], 1 -> C cols [2,3]
    
    blocks = [
        (0, 0, 0, 0),  # A_L * B_TL -> C_L
        (1, 1, 0, 0),  # A_R * B_BL -> C_L
        (0, 0, 1, 1),  # A_L * B_TR -> C_R
        (1, 1, 1, 1),  # A_R * B_BR -> C_R
    ]
    
    col = 0
    for (a_blk, b_rblk, b_cblk, c_cblk) in blocks:
        for s in range(7):
            # U: A[i, col] where i in {0,1}, col = 2*a_blk + local_col
            # Local X index: 0=(i=0,c=0), 1=(i=0,c=1), 2=(i=1,c=0), 3=(i=1,c=1)
            for loc, coeff in U_coeffs[s].items():
                i = loc // 2
                c = loc % 2
                g_col = 2 * a_blk + c
                U[i * m + g_col, col] += coeff
            
            # V: B[row, col] where row = 2*b_rblk + local_row, col = 2*b_cblk + local_col
            # Local Y index: 0=(r=0,c=0), 1=(r=0,c=1), 2=(r=1,c=0), 3=(r=1,c=1)
            for loc, coeff in V_coeffs[s].items():
                r = loc // 2
                c = loc % 2
                g_row = 2 * b_rblk + r
                g_col = 2 * b_cblk + c
                V[g_row * p + g_col, col] += coeff
            
            # W: C[row, col] where row = local_row, col = 2*c_cblk + local_col
            # W index = j*n + i = g_col * n + g_row
            # Local P index: 0=(r=0,c=0), 1=(r=0,c=1), 2=(r=1,c=0), 3=(r=1,c=1)
            for loc, coeff in W_coeffs[s].items():
                r = loc // 2
                c = loc % 2
                g_row = r
                g_col = 2 * c_cblk + c
                W[g_col * n + g_row, col] += coeff
            
            col += 1
    
    return U, V, W


def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """
    Verify T[i,k,j] = sum_r U[i*m+k,r] * V[k*p+j,r] * W[j*n+i,r] == 1 for all i,k,j.
    """
    rank = U.shape[1]
    for i in range(n):
        for k in range(m):
            for j in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i * m + k, r] * V[k * p + j, r] * W[j * n + i, r]
                if abs(val - 1.0) > tol:
                    return False
    return True


def construct():
    """
    Return a rank-28 exact decomposition of the 2x4 * 4x4 matrix multiplication tensor.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # 10s margin before 120s limit
    
    # Build the rank-28 Strassen decomposition
    U, V, W = _build_strassen_rank28()
    
    # Verify correctness
    if not _verify_decomposition(U, V, W, n, m, p):
        raise ValueError("Rank 28 decomposition failed verification")
    
    elapsed = time.time() - start_time
    if elapsed > deadline - start_time:
        raise TimeoutError("Exceeded time budget")
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
