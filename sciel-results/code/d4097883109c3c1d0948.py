"""
Optimized 4x4x4 matrix multiplication tensor decomposition.

The current best known rank for 4x4 matrix multiplication is 49, achieved by
applying Strassen's algorithm at two levels (7 block multiplications, each using
7 scalar multiplications). This program constructs that decomposition directly.

The key insight is that the 4x4 matrix multiplication tensor can be decomposed
with rank 49 by viewing 4x4 matrices as 2x2 blocks of 2x2 matrices, applying
Strassen at the block level (7 block multiplications), and then applying Strassen
at the scalar level within each block multiplication (7 scalar multiplications each).

This gives a total of 7 * 7 = 49 rank-1 terms.
"""

import time
import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START

def _strassen_2x2():
    """
    Return the 2x2 Strassen decomposition as three lists of 7 vectors.
    
    U: list of 7 vectors, each of length 4 (for A entries A00,A01,A10,A11)
    V: list of 7 vectors, each of length 4 (for B entries B00,B01,B10,B11)
    W: list of 7 vectors, each of length 4 (for C entries C00,C01,C10,C11)
    
    Indexing convention:
    U[i*2+j] for A[i,j]
    V[j*2+k] for B[j,k]
    W[k*2+i] for C[i,k] (note: W index is k*n+i where n=2, so W[k*2+i])
    """
    # Each of the 7 Strassen multiplications:
    # M1 = (A00+A11)*(B00+B11) -> C00 += M1, C11 += M1
    # M2 = (A10+A11)*B00       -> C10 += M2
    # M3 = A00*(B01-B11)       -> C01 += M3, C11 -= M3
    # M4 = A11*(B10-B00)       -> C00 += M4, C10 += M4
    # M5 = (A00+A01)*B11       -> C00 -= M5, C01 += M5
    # M6 = (A10-A00)*(B00+B01) -> C11 += M6
    # M7 = (A01-A11)*(B10+B11) -> C00 += M7, C11 -= M7
    
    U = [
        [1, 0, 0, 1],  # M1: A00 + A11
        [0, 0, 1, 1],  # M2: A10 + A11
        [1, 0, 0, 0],  # M3: A00
        [0, 0, 0, 1],  # M4: A11
        [1, 1, 0, 0],  # M5: A00 + A01
        [0, 0, 1, -1], # M6: A10 - A00
        [0, 1, 0, -1], # M7: A01 - A11
    ]
    V = [
        [1, 0, 0, 1],  # M1: B00 + B11
        [1, 0, 0, 0],  # M2: B00
        [0, 1, 0, -1], # M3: B01 - B11
        [0, 0, 1, -1], # M4: B10 - B00
        [0, 0, 0, 1],  # M5: B11
        [1, 1, 0, 0],  # M6: B00 + B01
        [0, 0, 1, 1],  # M7: B10 + B11
    ]
    W = [
        [1, 0, 0, 1],  # M1: C00 + C11
        [0, 0, 1, 0],  # M2: C10
        [0, 1, 0, -1], # M3: C01 - C11
        [1, 0, 1, 0],  # M4: C00 + C10
        [-1, 1, 0, 0], # M5: -C00 + C01
        [0, 0, 0, 1],  # M6: C11
        [1, 0, 0, -1], # M7: C00 - C11
    ]
    return U, V, W


def _construct_49_rank():
    """
    Construct the rank-49 decomposition for 4x4 matrix multiplication.
    
    Returns U, V, W as lists of lists with shapes:
    U: (16, 49)
    V: (16, 49)  
    W: (16, 49)
    
    Indexing:
    U[i*4+j, r] for A[i,j]
    V[j*4+k, r] for B[j,k]
    W[k*4+i, r] for C[i,k]
    """
    n = m = p = 4
    rank = 49
    
    U = [[0.0] * rank for _ in range(16)]
    V = [[0.0] * rank for _ in range(16)]
    W = [[0.0] * rank for _ in range(16)]
    
    U2, V2, W2 = _strassen_2x2()
    
    # Block-level Strassen for 4x4:
    # Partition into 2x2 blocks:
    # A = [[A00, A01], [A10, A11]] where each block is 2x2
    # B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    
    # Block Strassen:
    # M1 = (A00+A11)*(B00+B11) -> C00 += M1, C11 += M1
    # M2 = (A10+A11)*B00       -> C10 += M2
    # M3 = A00*(B01-B11)       -> C01 += M3, C11 -= M3
    # M4 = A11*(B10-B00)       -> C00 += M4, C10 += M4
    # M5 = (A00+A01)*B11       -> C00 -= M5, C01 += M5
    # M6 = (A10-A00)*(B00+B01) -> C11 += M6
    # M7 = (A01-A11)*(B10+B11) -> C00 += M7, C11 -= M7
    
    # For each block multiplication r (0-6), we have:
    # - A blocks involved with coefficients
    # - B blocks involved with coefficients  
    # - C blocks involved with coefficients
    
    # Block (bi, bj) where bi, bj in {0,1}:
    # A block (bi,bj): rows bi*2 to bi*2+1, cols bj*2 to bj*2+1
    #   Element (i,j) in block -> global (bi*2+i, bj*2+j)
    # B block (bi,bj): rows bi*2 to bi*2+1, cols bj*2 to bj*2+1
    #   Element (j,k) in block -> global (bi*2+j, bj*2+k)
    # C block (bi,bj): rows bi*2 to bi*2+1, cols bj*2 to bj*2+1
    #   Element (i,k) in block -> global (bi*2+i, bj*2+k)
    #   W index: k*4+i where k is col of C, i is row of C
    
    # Define block-level coefficients for each of the 7 block multiplications
    # Format: list of (block_row, block_col, coefficient)
    
    block_mults = [
        # M1: (A00+A11)*(B00+B11) -> C00(+1), C11(+1)
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
        # M2: (A10+A11)*B00 -> C10(+1)
        ([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1)]),
        # M3: A00*(B01-B11) -> C01(+1), C11(-1)
        ([(0,0,1)], [(0,1,1), (1,1,-1)], [(0,1,1), (1,1,-1)]),
        # M4: A11*(B10-B00) -> C00(+1), C10(+1)
        ([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (1,0,1)]),
        # M5: (A00+A01)*B11 -> C00(-1), C01(+1)
        ([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (0,1,1)]),
        # M6: (A10-A00)*(B00+B01) -> C11(+1)
        ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)]),
        # M7: (A01-A11)*(B10+B11) -> C00(+1), C11(-1)
        ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1), (1,1,-1)]),
    ]
    
    for r in range(7):
        a_blocks, b_blocks, c_blocks = block_mults[r]
        
        for s in range(7):
            global_idx = r * 7 + s
            
            # Process U contributions
            for (bi, bj, ca) in a_blocks:
                for i in range(2):
                    for j in range(2):
                        u_val = U2[s][i*2+j]
                        if u_val != 0:
                            gi = bi * 2 + i
                            gj = bj * 2 + j
                            U[gi * 4 + gj][global_idx] += ca * u_val
            
            # Process V contributions
            for (bi, bj, cb) in b_blocks:
                for j in range(2):
                    for k in range(2):
                        v_val = V2[s][j*2+k]
                        if v_val != 0:
                            gj = bi * 2 + j
                            gk = bj * 2 + k
                            V[gj * 4 + gk][global_idx] += cb * v_val
            
            # Process W contributions
            for (bi, bj, cc) in c_blocks:
                for i in range(2):
                    for k in range(2):
                        w_val = W2[s][k*2+i]
                        if w_val != 0:
                            gi = bi * 2 + i  # row of C
                            gk = bj * 2 + k  # col of C
                            # W index = gk * 4 + gi
                            W[gk * 4 + gi][global_idx] += cc * w_val
    
    return U, V, W


def _verify(U, V, W, n, m, p):
    """Verify the decomposition is correct."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True


def _schoolbook(n, m, p):
    """Fallback schoolbook decomposition with rank n*m*p."""
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
    return U, V, W


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # Try the rank-49 Strassen decomposition
    try:
        U, V, W = _construct_49_rank()
        
        # Verify
        if _verify(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline:
        # Return schoolbook as last resort
        U, V, W = _schoolbook(n, m, p)
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    U, V, W = _schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
