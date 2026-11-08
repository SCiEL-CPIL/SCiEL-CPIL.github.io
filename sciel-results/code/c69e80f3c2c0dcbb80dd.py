"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 7
    deadline = time.time() + 140  # safety margin
    
    # Decompose (2,4,7) as (2,4,4) + (2,4,3)
    # (2,4,4): split middle 4 into 2+2, use Strassen for each (2,2,2)
    # (2,4,3): split middle 4 into 2+2, use schoolbook for each (2,2,3)
    
    # We'll build U, V, W matrices with total rank = 7 + 7 + 12 + 12 = 38
    rank = 38
    U = [[0] * rank for _ in range(n * m)]  # 8 x 38
    V = [[0] * rank for _ in range(m * p)]  # 28 x 38
    W = [[0] * rank for _ in range(p * n)]  # 14 x 38
    
    col = 0
    
    # Helper: add Strassen (2,2,2) block
    # A is 2x2 at rows [i0:i0+2, j0:j0+2] of A (2x4)
    # B is 2x2 at rows [j0:j0+2, k0:k0+2] of B (4x7)
    # C is 2x2 at rows [k0:k0+2, i0:i0+2] of C (7x2)
    def add_strassen(i0, j0, k0, col_start):
        nonlocal col
        # Strassen formulas for 2x2x2:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B11-B21)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # A indices: A[i, j] where i in [i0, i0+2), j in [j0, j0+2)
        # In U: row = i*m + j = i*4 + j
        # B indices: B[j, k] where j in [j0, j0+2), k in [k0, k0+2)
        # In V: row = j*p + k = j*7 + k
        # C indices: C[k, i] where k in [k0, k0+2), i in [i0, i0+2)
        # In W: row = k*n + i = k*2 + i (since n=2)
        
        a11 = (i0, j0)
        a12 = (i0, j0+1)
        a21 = (i0+1, j0)
        a22 = (i0+1, j0+1)
        
        b11 = (j0, k0)
        b12 = (j0, k0+1)
        b21 = (j0+1, k0)
        b22 = (j0+1, k0+1)
        
        c11 = (k0, i0)
        c12 = (k0, i0+1)
        c21 = (k0+1, i0)
        c22 = (k0+1, i0+1)
        
        def u_row(i, j):
            return i * m + j
        
        def v_row(j, k):
            return j * p + k
        
        def w_row(k, i):
            return k * n + i
        
        # M1 = (A11+A22)(B11+B22)
        U[u_row(*a11)][col] = 1
        U[u_row(*a22)][col] = 1
        V[v_row(*b11)][col] = 1
        V[v_row(*b22)][col] = 1
        W[w_row(*c11)][col] = 1
        W[w_row(*c22)][col] = -1
        W[w_row(*c12)][col] = 1
        W[w_row(*c21)][col] = 1
        col += 1
        
        # M2 = (A21+A22)B11
        U[u_row(*a21)][col] = 1
        U[u_row(*a22)][col] = 1
        V[v_row(*b11)][col] = 1
        W[w_row(*c21)][col] = 1
        W[w_row(*c22)][col] = -1
        col += 1
        
        # M3 = A11(B21-B22)
        U[u_row(*a11)][col] = 1
        V[v_row(*b21)][col] = 1
        V[v_row(*b22)][col] = -1
        W[w_row(*c12)][col] = 1
        W[w_row(*c22)][col] = 1
        col += 1
        
        # M4 = A22(B11-B21)
        U[u_row(*a22)][col] = 1
        V[v_row(*b11)][col] = 1
        V[v_row(*b21)][col] = -1
        W[w_row(*c11)][col] = 1
        W[w_row(*c21)][col] = 1
        col += 1
        
        # M5 = (A11+A12)B22
        U[u_row(*a11)][col] = 1
        U[u_row(*a12)][col] = 1
        V[v_row(*b22)][col] = 1
        W[w_row(*c11)][col] = -1
        W[w_row(*c12)][col] = 1
        col += 1
        
        # M6 = (A21-A11)(B11+B12)
        U[u_row(*a21)][col] = 1
        U[u_row(*a11)][col] = -1
        V[v_row(*b11)][col] = 1
        V[v_row(*b12)][col] = 1
        W[w_row(*c22)][col] = 1
        col += 1
        
        # M7 = (A12-A22)(B21+B22)
        U[u_row(*a12)][col] = 1
        U[u_row(*a22)][col] = -1
        V[v_row(*b21)][col] = 1
        V[v_row(*b22)][col] = 1
        W[w_row(*c11)][col] = 1
        W[w_row(*c12)][col] = -1
        col += 1
        
        return col
    
    # Helper: add schoolbook (2,2,3) block
    def add_schoolbook_223(i0, j0, k0, col_start):
        nonlocal col
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    U[(i0+i) * m + (j0+j)][col] = 1
                    V[(j0+j) * p + (k0+k)][col] = 1
                    W[(k0+k) * n + (i0+i)][col] = 1
                    col += 1
        return col
    
    # (2,4,4) part: k from 0 to 3
    # Split into two (2,2,2) blocks: k0=0 and k0=2
    # Block 1: i0=0, j0=0, k0=0
    col = add_strassen(0, 0, 0, col)
    # Block 2: i0=0, j0=2, k0=2
    col = add_strassen(0, 2, 2, col)
    
    # (2,4,3) part: k from 4 to 6
    # Split into two (2,2,3) blocks: j0=0 and j0=2
    # Block 1: i0=0, j0=0, k0=4
    col = add_schoolbook_223(0, 0, 4, col)
    # Block 2: i0=0, j0=2, k0=4
    col = add_schoolbook_223(0, 2, 4, col)
    
    # Verify col == rank
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
