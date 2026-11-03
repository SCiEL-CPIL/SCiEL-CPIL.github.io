"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank 36 decomposition for 2x4 * 4x5 matrix multiplication.
    
    Partition:
    - n=2 (rows of A, cols of C)
    - m=4 (cols of A, rows of B), split into m1=2, m2=2
    - p=5 (cols of B, rows of C), split into p1=2, p2=3
    
    Four blocks:
    1. A[0:2,0:2] * B[0:2,0:2] -> Strassen rank 7
    2. A[0:2,2:4] * B[2:4,0:2] -> Strassen rank 7
    3. A[0:2,0:2] * B[0:2,2:5] -> Rank 11 (7+4)
    4. A[0:2,2:4] * B[2:4,2:5] -> Rank 11 (7+4)
    
    Total rank = 7+7+11+11 = 36
    """
    n, m, p = DIMENSIONS  # 2, 4, 5
    rank = 36
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    def add_strassen(i_off, j_off, k_off):
        """Add a 2x2x2 Strassen decomposition block at offsets."""
        nonlocal col
        # Local indices: i in 0..1, j in 0..1, k in 0..1
        # Global flat indices:
        # U: (i_off+i)*m + (j_off+j)
        # V: (j_off+j)*p + (k_off+k)
        # W: (k_off+k)*n + (i_off+i)
        
        # M1: (a00+a11)(b00+b11)
        U[(i_off+0)*m + (j_off+0)][col] = 1
        U[(i_off+1)*m + (j_off+1)][col] = 1
        V[(j_off+0)*p + (k_off+0)][col] = 1
        V[(j_off+1)*p + (k_off+1)][col] = 1
        W[(k_off+0)*n + (i_off+0)][col] = 1
        W[(k_off+1)*n + (i_off+1)][col] = 1
        col += 1
        
        # M2: (a10+a11)*b00
        U[(i_off+1)*m + (j_off+0)][col] = 1
        U[(i_off+1)*m + (j_off+1)][col] = 1
        V[(j_off+0)*p + (k_off+0)][col] = 1
        W[(k_off+0)*n + (i_off+1)][col] = 1
        W[(k_off+1)*n + (i_off+1)][col] = -1
        col += 1
        
        # M3: a00*(b01-b11)
        U[(i_off+0)*m + (j_off+0)][col] = 1
        V[(j_off+0)*p + (k_off+1)][col] = 1
        V[(j_off+1)*p + (k_off+1)][col] = -1
        W[(k_off+1)*n + (i_off+0)][col] = 1
        W[(k_off+1)*n + (i_off+1)][col] = 1
        col += 1
        
        # M4: a11*(b10-b00)
        U[(i_off+1)*m + (j_off+1)][col] = 1
        V[(j_off+1)*p + (k_off+0)][col] = 1
        V[(j_off+0)*p + (k_off+0)][col] = -1
        W[(k_off+0)*n + (i_off+0)][col] = 1
        W[(k_off+0)*n + (i_off+1)][col] = 1
        col += 1
        
        # M5: (a00+a01)*b11
        U[(i_off+0)*m + (j_off+0)][col] = 1
        U[(i_off+0)*m + (j_off+1)][col] = 1
        V[(j_off+1)*p + (k_off+1)][col] = 1
        W[(k_off+0)*n + (i_off+0)][col] = -1
        W[(k_off+1)*n + (i_off+0)][col] = 1
        col += 1
        
        # M6: (a10-a00)*(b00+b01)
        U[(i_off+1)*m + (j_off+0)][col] = 1
        U[(i_off+0)*m + (j_off+0)][col] = -1
        V[(j_off+0)*p + (k_off+0)][col] = 1
        V[(j_off+0)*p + (k_off+1)][col] = 1
        W[(k_off+1)*n + (i_off+1)][col] = 1
        col += 1
        
        # M7: (a01-a11)*(b10+b11)
        U[(i_off+0)*m + (j_off+1)][col] = 1
        U[(i_off+1)*m + (j_off+1)][col] = -1
        V[(j_off+1)*p + (k_off+0)][col] = 1
        V[(j_off+1)*p + (k_off+1)][col] = 1
        W[(k_off+0)*n + (i_off+0)][col] = 1
        col += 1
    
    def add_2x2x3(i_off, j_off, k_off):
        """
        Add a 2x2x3 block decomposition.
        Uses Strassen (7) for first 2 columns + schoolbook (4) for 3rd column.
        Local k indices: 0, 1 use Strassen, k=2 uses 4 rank-1 terms.
        """
        nonlocal col
        # Strassen part for k_local = 0, 1
        add_strassen(i_off, j_off, k_off)
        
        # Schoolbook part for k_local = 2
        # C[k_off+2, i_off+0] = A[0,j0]*B[j0,k2] + A[1,j0]*B[j1,k2]
        # C[k_off+2, i_off+1] = A[0,j1]*B[j0,k2] + A[1,j1]*B[j1,k2]
        
        # Factor 1: A[i_off+0, j_off+0] * B[j_off+0, k_off+2]
        U[(i_off+0)*m + (j_off+0)][col] = 1
        V[(j_off+0)*p + (k_off+2)][col] = 1
        W[(k_off+2)*n + (i_off+0)][col] = 1
        col += 1
        
        # Factor 2: A[i_off+1, j_off+0] * B[j_off+1, k_off+2]
        U[(i_off+1)*m + (j_off+0)][col] = 1
        V[(j_off+1)*p + (k_off+2)][col] = 1
        W[(k_off+2)*n + (i_off+0)][col] = 1
        col += 1
        
        # Factor 3: A[i_off+0, j_off+1] * B[j_off+0, k_off+2]
        U[(i_off+0)*m + (j_off+1)][col] = 1
        V[(j_off+0)*p + (k_off+2)][col] = 1
        W[(k_off+2)*n + (i_off+1)][col] = 1
        col += 1
        
        # Factor 4: A[i_off+1, j_off+1] * B[j_off+1, k_off+2]
        U[(i_off+1)*m + (j_off+1)][col] = 1
        V[(j_off+1)*p + (k_off+2)][col] = 1
        W[(k_off+2)*n + (i_off+1)][col] = 1
        col += 1
    
    # Block 1: i=0..1, j=0..1, k=0..1 (top-left 2x2x2)
    add_strassen(0, 0, 0)
    
    # Block 2: i=0..1, j=2..3, k=0..1 (top-right 2x2x2)
    add_strassen(0, 2, 0)
    
    # Block 3: i=0..1, j=0..1, k=2..4 (bottom-left 2x2x3)
    add_2x2x3(0, 0, 2)
    
    # Block 4: i=0..1, j=2..3, k=2..4 (bottom-right 2x2x3)
    add_2x2x3(0, 2, 2)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
