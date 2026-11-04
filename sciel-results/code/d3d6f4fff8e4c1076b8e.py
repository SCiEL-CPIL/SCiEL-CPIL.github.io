import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-89 decomposition for 5x5 matrix multiplication using
    Strassen's algorithm on the 4x4 sub-block and schoolbook for edges.
    
    Rank calculation:
    4x4*4x4 via Strassen: 7 * 4 = 28
    4x1*1x4: 16
    4x4*4x1: 16
    4x1*1x1: 4
    1x4*4x4: 16
    1x1*1x4: 4
    1x4*4x1: 4
    1x1*1x1: 1
    Total: 89
    """
    start_time = time.time()
    deadline = 120.0
    safety_margin = 5.0
    
    n, m, p = DIMENSIONS
    nm = n * m  # 25
    mp = m * p  # 25
    pn = p * n  # 25
    
    terms = []
    
    def add_term(i, j, k, coeff=1.0):
        if coeff != 0:
            u_idx = i * m + j
            v_idx = j * p + k
            w_idx = k * n + i
            terms.append((u_idx, v_idx, w_idx, coeff))
    
    def add_2x2_product(i_start, j_start, k_start, sign=1.0):
        """Add a 2x2 schoolbook product starting at (i_start, j_start, k_start)."""
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    add_term(i_start + di, j_start + dj, k_start + dk, sign)
    
    # Strassen 4x4 decomposition for A[0:4,0:4] * B[0:4,0:4] -> C[0:4,0:4]
    # Blocks: A11=A[0:2,0:2], A12=A[0:2,2:4], A21=A[2:4,0:2], A22=A[2:4,2:4]
    #          B11=B[0:2,0:2], B12=B[0:2,2:4], B21=B[2:4,0:2], B22=B[2:4,2:4]
    
    # M1 = (A11+A22)(B11+B22)
    # Terms: A11*B11, A11*B22, A22*B11, A22*B22
    add_2x2_product(0, 0, 0, 1)   # A11*B11
    add_2x2_product(0, 2, 0, 1)   # A11*B22  -- wait, this is wrong
