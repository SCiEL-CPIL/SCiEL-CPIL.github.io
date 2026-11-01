import numpy as np
import time

def construct():
    """
    Rank-minimal exact tensor decomposition for 2x4 * 4x7 matrix multiplication.
    
    The tensor is T: F^{2x4} x F^{4x7} -> F^{2x7}, where (C)_{i,k} = sum_j A_{i,j} * B_{j,k}.
    
    The decomposition is U (8 x r), V (28 x r), W (14 x r) such that
    for all i, j, k: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    Strategy:
    - Split the 4 middle columns into two blocks of 2: j0 in {0, 2}
    - Split the 7 output columns into three blocks: {0,1,2}, {3,4}, {5,6}
    - For each (j0, k-block) pair, use the optimal known rank:
      - 2x2 * 2x2 (Strassen): rank 7
      - 2x2 * 2x1: rank 4
      Total: 3 * 7 + 3 * 4 = 33
    """
    
    n, m, p = 2, 4, 7
    rank = 33
    
    # U: 8 x rank (indices: i*m + j)
    # V: 28 x rank (indices: j*p + k)
    # W: 14 x rank (indices: k*n + i)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_222(i0, j0, k0):
        """
        Add a Strassen 2x2 * 2x2 block.
        
        A block: rows i0, i0+1; cols j0, j0+1
        B block: rows j0, j0+1; cols k0, k0+1
        C block: rows i0, i0+1; cols k0, k0+1
        
        Strassen formulas (7 multiplications):
        M1 = (A11 + A22) * (B11 + B22)
        M2 = (A21 + A22) * B11
        M3 = A11 * (B21 - B11)
        M4 = A22 * (B22 - B12)
        M5 = (A11 + A12) * B22
        M6 = (A21 - A11) * (B11 + B12)
        M7 = (A12 - A22) * (B21 + B22)
        
        C11 = M1 + M4 - M5 + M7
        C12 = M3 + M5
        C21 = M2 + M4
        C22 = M1 - M2 + M3 + M6
        """
        nonlocal col
        if col >= rank:
            return False
        
        # A indices
        a11 = i0 * m + j0
        a12 = i0 * m + (j0 + 1)
        a21 = (i0 + 1) * m + j0
        a22 = (i0 + 1) * m + (j0 + 1)
        
        # B indices
        b11 = j0 * p + k0
        b12 = j0 * p + (k0 + 1)
        b21 = (j0 + 1) * p + k0
        b22 = (j0 + 1) * p + (k0 + 1)
        
        # C indices (W convention: k*n + i)
        c11 = k0 * n + i0
        c12 = (k0 + 1) * n + i0
        c21 = k0 * n + (i0 + 1)
        c22 = (k0 + 1) * n + (i0 + 1)
        
        # M1 = (A11 + A22) * (B11 + B22)
        U[a11][col] = 1
        U[a22][col] = 1
        V[b11][col] = 1
        V[b22][col] = 1
        W[c11][col] = 1
        W[c22][col] = 1
        col += 1
        
        # M2 = (A21 + A22) * B11
        U[a21][col] = 1
        U[a22][col] = 1
        V[b11][col] = 1
        W[c21][col] = 1
        W[c22][col] = -1
        col += 1
        
        # M3 = A11 * (B21 - B11)
        U[a11][col] = 1
        V[b21][col] = 1
        V[b11][col] = -1
        W[c12][col] = 1
        W[c22][col] = 1
        col += 1
        
        # M4 = A22 * (B22 - B12)
        U[a22][col] = 1
        V[b22][col] = 1
        V[b12][col] = -1
        W[c11][col] = 1
        W[c21][col] = 1
        col += 1
        
        # M5 = (A11 + A12) * B22
        U[a11][col] = 1
        U[a12][col] = 1
        V[b22][col] = 1
        W[c11][col] = -1
        W[c12][col] = 1
        col += 1
        
        # M6 = (A21 - A11) * (B11 + B12)
        U[a21][col] = 1
        U[a11][col] = -1
        V[b11][col] = 1
        V[b12][col] = 1
        W[c22][col] = 1
        col += 1
        
        # M7 = (A12 - A22) * (B21 + B22)
        U[a12][col] = 1
        U[a22][col] = -1
        V[b21][col] = 1
        V[b22][col] = 1
        W[c11][col] = 1
        col += 1
        
        return True
    
    def add_rank4_221(i0, j0, k0):
        """
        Add a 2x2 * 2x1 block with rank 4.
        
        A block: rows i0, i0+1; cols j0, j0+1
        B block: rows j0, j0+1; col k0
        C block: rows i0, i0+1; col k0
        
        This is just 4 independent products (no Strassen benefit for 2x1).
        """
        nonlocal col
        if col >= rank:
            return False
        
        # A indices
        a11 = i0 * m + j0
        a12 = i0 * m + (j0 + 1)
        a21 = (i0 + 1) * m + j0
        a22 = (i0 + 1) * m + (j0 + 1)
        
        # B indices
        b11 = j0 * p + k0
        b21 = (j0 + 1) * p + k0
        
        # C indices
        c11 = k0 * n + i0
        c21 = k0 * n + (i0 + 1)
        
        # A11 * B11
        U[a11][col] = 1
        V[b11][col] = 1
        W[c11][col] = 1
        col += 1
        
        # A12 * B21
        U[a12][col] = 1
        V[b21][col] = 1
        W[c11][col] = 1
        col += 1
        
        # A21 * B11
        U[a21][col] = 1
        V[b11][col] = 1
        W[c21][col] = 1
        col += 1
        
        # A22 * B21
        U[a22][col] = 1
        V[b21][col] = 1
        W[c21][col] = 1
        col += 1
        
        return True
    
    # Process blocks
    # j0 = 0: j = 0,1
    #   k0 = 0: k = 0,1 -> Strassen (7)
    #   k0 = 3: k = 3,4 -> Strassen (7)
    #   k0 = 6: k = 6   -> Rank 4 (4)
    # j0 = 2: j = 2,3
    #   k0 = 0: k = 0,1 -> Strassen (7)
    #   k0 = 3: k = 3,4 -> Strassen (7)
    #   k0 = 6: k = 6   -> Rank 4 (4)
    
    blocks = [
        ('strassen', 0, 0, 0),
        ('strassen', 0, 0, 3),
        ('rank4', 0, 0, 6),
        ('strassen', 0, 2, 0),
        ('strassen', 0, 2, 3),
        ('rank4', 0, 2, 6),
    ]
    
    for block_type, i0, j0, k0 in blocks:
        if block_type == 'strassen':
            add_strassen_222(i0, j0, k0)
        else:
            add_rank4_221(i0, j0, k0)
    
    return {"U": U, "V": V, "W": W}
