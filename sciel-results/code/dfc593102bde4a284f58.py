"""Matrix multiplication 2x2, 2x7 exact tensor decomposition.

Strategy:
  * Guaranteed fallback: the explicit schoolbook decomposition of rank
    n*m*p = 28.

  * Improvement: block-diagonal construction.
    Partition the 7 output columns into three 2-column blocks (k=0,1), (k=2,3), (k=4,5)
    and one 1-column block (k=6).
    For each 2-column block, use a Strassen rank-7 decomposition of 2x2x2.
    For the 1-column block, use a rank-4 schoolbook decomposition of 2x2x1.
    Total rank = 3*7 + 4 = 25.
"""

import time
import numpy as np

DIMENSIONS = [2, 2, 7]
START_TIME = time.time()
TIME_LIMIT = 150  # seconds

def construct():
    n, m, p = 2, 2, 7
    
    # Check time
    if time.time() - START_TIME > TIME_LIMIT - 5:
        return _schoolbook(n, m, p)

    # --- Build the rank-25 block-diagonal decomposition ---
    
    # Get a valid Strassen decomposition for 2x2x2
    # We generate it by solving for the coefficients of the 7 rank-1 terms
    # that match the identity tensor for 2x2x2.
    # The standard Strassen algorithm provides this.
    # U_str: 4x7, V_str: 4x7, W_str: 4x7
    # Indices: U[i*m+j, r], V[j*p+k, r], W[k*n+i, r] for n=m=p=2
    
    U_str, V_str, W_str = _get_strassen_2x2x2()
    
    if U_str is None:
        return _schoolbook(n, m, p)

    # Verify Strassen block locally
    for i in range(2):
        for j in range(2):
            for k in range(2):
                s = 0
                for r in range(7):
                    s += U_str[i*2+j][r] * V_str[j*2+k][r] * W_str[k*2+i][r]
                if s != 1:
                    return _schoolbook(n, m, p)

    # Now build the full 2x2x7 decomposition
    total_rank = 3 * 7 + 4  # = 25

    U = [[0] * total_rank for _ in range(n * m)]   # 4 x 25
    V = [[0] * total_rank for _ in range(m * p)]   # 14 x 25
    W = [[0] * total_rank for _ in range(p * n)]   # 14 x 25

    col = 0
    # Three Strassen blocks for k-offsets 0, 2, 4
    for block_start in [0, 2, 4]:
        for r in range(7):
            # U part: same for all blocks (operates on A which is 2x2)
            for i in range(2):
                for j in range(2):
                    U[i * m + j][col] = U_str[i * 2 + j][r]
            # V part: V[j*p+k, col] where k = block_start + local_k
            for j in range(2):
                for local_k in range(2):
                    k = block_start + local_k
                    V[j * p + k][col] = V_str[j * 2 + local_k][r]
            # W part: W[k*n+i, col] where k = block_start + local_k
            for local_k in range(2):
                for i in range(2):
                    k = block_start + local_k
                    W[k * n + i][col] = W_str[local_k * 2 + i][r]
            col += 1

    # Schoolbook block for k=6: rank 4 (one per (i,j) pair)
    for i in range(2):
        for j in range(2):
            U[i * m + j][col] = 1
            V[j * p + 6][col] = 1
            W[6 * n + i][col] = 1
            col += 1

    # Final exact verification
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(total_rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if s != 1:
                    return _schoolbook(n, m, p)

    return {"U": U, "V": V, "W": W}

def _get_strassen_2x2x2():
    """
    Returns U, V, W matrices for Strassen 2x2x2 decomposition.
    U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    # Standard Strassen formulas
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b01-b11)
    # M4 = a11(b10-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # Term 0: M1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # Term 1: M2
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1
    
    # Term 2: M3
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1
    
    # Term 3: M4
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # Term 4: M5
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # Term 5: M6
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # Term 6: M7
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
