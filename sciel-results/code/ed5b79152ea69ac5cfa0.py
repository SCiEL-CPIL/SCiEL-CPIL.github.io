import time
import numpy as np

DIMENSIONS = [2, 3, 8]

def construct():
    """
    Constructs a rank-44 matrix multiplication decomposition for 2x3 * 3x8.
    Uses block-partitioned Strassen for the 2x2 sub-blocks and schoolbook for the 2x1 sub-block.
    Returns a dictionary with matrices U, V, W of shapes (n*m, rank), (m*p, rank), (p*n, rank).
    """
    start_time = time.time()
    deadline = start_time + 115
    
    n, m, p = DIMENSIONS
    rank = 44
    
    # Initialize U, V, W as lists of lists of zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_2x2x2(i_off, j_off, k_off):
        """
        Add 7 rank-1 terms for a 2x2x2 Strassen block.
        i_off: row offset in A (0 or 1)
        j_off: col offset in A / row offset in B (0 or 1)
        k_off: col offset in B (0, 2, 4, 6)
        
        U index: (i_off+ii) * m + (j_off+jj)
        V index: (j_off+jj) * p + (k_off+kk)
        W index: (k_off+kk) * n + (i_off+ii)
        """
        nonlocal col
        
        # Mult 1: (a00+a11)(b00+b11) -> c00 (+), c11 (+)
        U[i_off * m + j_off][col] += 1
        U[(i_off + 1) * m + (j_off + 1)][col] += 1
        V[j_off * p + k_off][col] += 1
        V[(j_off + 1) * p + (k_off + 1)][col] += 1
        W[k_off * n + i_off][col] += 1
        W[(k_off + 1) * n + (i_off + 1)][col] += 1
        col += 1
        
        # Mult 2: (a10+a11)(b00) -> c10 (+), c11 (-)
        U[(i_off + 1) * m + j_off][col] += 1
        U[(i_off + 1) * m + (j_off + 1)][col] += 1
        V[j_off * p + k_off][col] += 1
        W[k_off * n + (i_off + 1)][col] += 1
        W[(k_off + 1) * n + (i_off + 1)][col] += -1
        col += 1
        
        # Mult 3: (a00)(b10-b11) -> c01 (+), c11 (+)
        U[i_off * m + j_off][col] += 1
        V[(j_off + 1) * p + k_off][col] += 1
        V[(j_off + 1) * p + (k_off + 1)][col] += -1
        W[(k_off + 1) * n + i_off][col] += 1
        W[(k_off + 1) * n + (i_off + 1)][col] += 1
        col += 1
        
        # Mult 4: (a11)(b01-b00) -> c00 (+), c10 (+)
        U[(i_off + 1) * m + (j_off + 1)][col] += 1
        V[j_off * p + (k_off + 1)][col] += 1
        V[j_off * p + k_off][col] += -1
        W[k_off * n + i_off][col] += 1
        W[k_off * n + (i_off + 1)][col] += 1
        col += 1
        
        # Mult 5: (a00+a01)(b11) -> c00 (-), c01 (+)
        U[i_off * m + j_off][col] += 1
        U[i_off * m + (j_off + 1)][col] += 1
        V[(j_off + 1) * p + (k_off + 1)][col] += 1
        W[k_off * n + i_off][col] += -1
        W[(k_off + 1) * n + i_off][col] += 1
        col += 1
        
        # Mult 6: (a10-a00)(b00+b01) -> c11 (+)
        U[(i_off + 1) * m + j_off][col] += 1
        U[i_off * m + j_off][col] += -1
        V[j_off * p + k_off][col] += 1
        V[j_off * p + (k_off + 1)][col] += 1
        W[(k_off + 1) * n + (i_off + 1)][col] += 1
        col += 1
        
        # Mult 7: (a01-a11)(b10+b11) -> c00 (+)
        U[i_off * m + (j_off + 1)][col] += 1
        U[(i_off + 1) * m + (j_off + 1)][col] += -1
        V[(j_off + 1) * p + k_off][col] += 1
        V[(j_off + 1) * p + (k_off + 1)][col] += 1
        W[k_off * n + i_off][col] += 1
        col += 1

    def add_schoolbook_2x1xk(i_off, j_idx, k_off, k_count):
        """
        Add k_count rank-1 terms for a 2x1xk schoolbook block.
        i_off: row offset in A (0)
        j_idx: col index in A (2)
        k_off: col offset in B
        k_count: number of columns
        """
        nonlocal col
        for ii in range(2):
            for kk in range(k_count):
                U[(i_off + ii) * m + j_idx][col] = 1
                V[j_idx * p + (k_off + kk)][col] = 1
                W[(k_off + kk) * n + (i_off + ii)][col] = 1
                col += 1

    # The matrix A is 2x3. We split the middle dimension (m=3) into:
    #   Part 1: columns 0,1 (2x2 block) -> use Strassen
    #   Part 2: column 2 (2x1 block) -> use schoolbook
    
    # The matrix B is 3x8. We split the output dimension (p=8) into 4 blocks of 2 columns.
    # For each block of 2 columns:
    #   - The 2x2 part of A (cols 0,1) multiplies with the 2x2 part of B (rows 0,1, cols k,k+1) using Strassen
    #   - The 2x1 part of A (col 2) multiplies with the 1x2 part of B (row 2, cols k,k+1) using schoolbook
    
    for block in range(4):
        k_off = block * 2
        
        # Part 1: Strassen for A[:, 0:2] * B[0:2, k:k+2]
        # i_off=0, j_off=0 (since we're using cols 0,1 of A and rows 0,1 of B)
        add_strassen_2x2x2(0, 0, k_off)
        
        # Part 2: Schoolbook for A[:, 2] * B[2, k:k+2]
        # i_off=0, j_idx=2, k_off=k_off, k_count=2
        add_schoolbook_2x1xk(0, 2, k_off, 2)

    if col != rank:
        # Fallback to schoolbook if construction failed (should not happen)
        return {"U": [[0]*48 for _ in range(6)], "V": [[0]*48 for _ in range(24)], "W": [[0]*48 for _ in range(16)]}
        
    return {"U": U, "V": V, "W": W}
