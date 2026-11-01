import numpy as np
import time

DIMENSIONS = [2, 4, 7]

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 2x4 * 4x7 matrix multiplication.
    Score = 1/rank. We use a block decomposition with 6 Strassen 2x2x2 blocks
    and 2 schoolbook 2x2x1 blocks, giving rank 50.
    
    Interface:
        U: shape (n*m, rank) = (8, 50)
        V: shape (m*p, rank) = (28, 50)
        W: shape (p*n, rank) = (14, 50)
    """
    n, m, p = DIMENSIONS
    # n=2, m=4, p=7
    
    # Target rank: 6*7 + 2*4 = 42 + 8 = 50
    R = 50
    
    # Initialize factor matrices
    U = np.zeros((n * m, R), dtype=np.float64)
    V = np.zeros((m * p, R), dtype=np.float64)
    W = np.zeros((p * n, R), dtype=np.float64)
    
    col = 0
    
    def add_strassen_222(i0, j0, k0):
        """
        Adds a 2x2x2 Strassen block.
        A block: rows i0, i0+1; cols j0, j0+1
        B block: rows j0, j0+1; cols k0, k0+1
        C block: rows i0, i0+1; cols k0, k0+1
        """
        nonlocal col
        if col >= R:
            return False
            
        # A indices
        a = i0 * m + j0
        b = i0 * m + (j0 + 1)
        c = (i0 + 1) * m + j0
        d = (i0 + 1) * m + (j0 + 1)
        
        # B indices
        e = j0 * p + k0
        f = j0 * p + (k0 + 1)
        g = (j0 + 1) * p + k0
        h = (j0 + 1) * p + (k0 + 1)
        
        # C indices (Trace convention: W[k*n + i, r])
        # C[i, k]
        c00 = k0 * n + i0
        c01 = (k0 + 1) * n + i0
        c10 = k0 * n + (i0 + 1)
        c11 = (k0 + 1) * n + (i0 + 1)
        
        # M1 = (a+d)(e+h) -> C00 += 1, C11 += 1
        U[a, col] += 1
        U[d, col] += 1
        V[e, col] += 1
        V[h, col] += 1
        W[c00, col] += 1
        W[c11, col] += 1
        col += 1
        
        # M2 = (c+d)e -> C10 += 1, C11 -= 1
        U[c, col] += 1
        U[d, col] += 1
        V[e, col] += 1
        W[c10, col] += 1
        W[c11, col] -= 1
        col += 1
        
        # M3 = a(f-h) -> C01 += 1, C11 += 1
        U[a, col] += 1
        V[f, col] += 1
        V[h, col] -= 1
        W[c01, col] += 1
        W[c11, col] += 1
        col += 1
        
        # M4 = d(g-e) -> C00 += 1, C10 += 1
        U[d, col] += 1
        V[g, col] += 1
        V[e, col] -= 1
        W[c00, col] += 1
        W[c10, col] += 1
        col += 1
        
        # M5 = (a+b)h -> C00 -= 1, C01 += 1
        U[a, col] += 1
        U[b, col] += 1
        V[h, col] += 1
        W[c00, col] -= 1
        W[c01, col] += 1
        col += 1
        
        # M6 = (c-a)(e+f) -> C11 += 1
        U[c, col] += 1
        U[a, col] -= 1
        V[e, col] += 1
        V[f, col] += 1
        W[c11, col] += 1
        col += 1
        
        # M7 = (b-d)(g+h) -> C00 += 1
        U[b, col] += 1
        U[d, col] -= 1
        V[g, col] += 1
        V[h, col] += 1
        W[c00, col] += 1
        col += 1
        
        return True

    def add_schoolbook_221(i0, j0, k0):
        """
        Adds a 2x2x1 schoolbook block.
        A block: rows i0, i0+1; cols j0, j0+1
        B block: rows j0, j0+1; cols k0
        C block: rows i0, i0+1; cols k0
        """
        nonlocal col
        if col >= R:
            return False
            
        a = i0 * m + j0
        b = i0 * m + (j0 + 1)
        c = (i0 + 1) * m + j0
        d = (i0 + 1) * m + (j0 + 1)
        
        e = j0 * p + k0
        g = (j0 + 1) * p + k0
        
        c00 = k0 * n + i0
        c10 = k0 * n + (i0 + 1)
        
        # P1 = a*e
        U[a, col] += 1
        V[e, col] += 1
        W[c00, col] += 1
        col += 1
        
        # P2 = b*g
        U[b, col] += 1
        V[g, col] += 1
        W[c00, col] += 1
        col += 1
        
        # P3 = c*e
        U[c, col] += 1
        V[e, col] += 1
        W[c10, col] += 1
        col += 1
        
        # P4 = d*g
        U[d, col] += 1
        V[g, col] += 1
        W[c10, col] += 1
        col += 1
        
        return True

    # Structure:
    # A is 2x4. Split into A_left (cols 0,1) and A_right (cols 2,3).
    # B is 4x7. Split into B_top (rows 0,1) and B_bot (rows 2,3).
    # C = A_left @ B_top + A_right @ B_bot
    
    # For A_left @ B_top:
    # B_top is 2x7. Split B_top into cols 0-1, 2-3, 4-5, 6.
    # Blocks: (0,0), (0,1), (0,2) are 2x2x2 Strassen. (0,3) is 2x2x1 Schoolbook.
    
    # For A_right @ B_bot:
    # B_bot is 2x7. Split B_bot into cols 0-1, 2-3, 4-5, 6.
    # Blocks: (1,0), (1,1), (1,2) are 2x2x2 Strassen. (1,3) is 2x2x1 Schoolbook.
    
    # Generate blocks for A_left @ B_top
    for k0 in [0, 2, 4]:
        add_strassen_222(0, 0, k0)
    
    # Last column of B_top is k=6 (single col)
    # This is a 2x2x1 block. A rows 0,1 cols 0,1. B rows 0,1 col 6.
    add_schoolbook_221(0, 0, 6)
    
    # Generate blocks for A_right @ B_bot
    # A_right: rows 0,1, cols 2,3. i0=0, j0=2.
    # B_bot: rows 2,3, cols k. j0=2.
    for k0 in [0, 2, 4]:
        add_strassen_222(0, 2, k0)
    
    # Last column of B_bot is k=6
    # i0=0, j0=2, k0=6.
    add_schoolbook_221(0, 2, 6)

    return {"U": U, "V": V, "W": W}
