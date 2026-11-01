"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 2]
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-7 exact tensor decomposition for 2x2 matrix multiplication.
    Uses Strassen's algorithm decomposition (optimal rank for 2x2).
    Verified optimal by Landsberg (1994).
    """
    n, m, p = DIMENSIONS  # 2, 2, 2

    # Strassen's 7 products:
    # M1 = (A00+A11)(B00+B11) -> C00+=M1, C11+=M1
    # M2 = (A10+A11)B00       -> C10+=M2, C11-=M2
    # M3 = A00(B01-B11)       -> C01+=M3, C11+=M3
    # M4 = A11(B10-B00)       -> C00+=M4, C10+=M4
    # M5 = (A00+A01)B11       -> C00-=M5, C01+=M5
    # M6 = (A10-A00)(B00+B01) -> C11+=M6
    # M7 = (A01-A11)(B10+B11) -> C00+=M7

    # U[i*m+j, c]: coefficient of A[i,j] in product c
    # Rows: A00, A01, A10, A11
    U_data = [
        [1, 0, 1, 0, 1, -1, 0],  # A[0,0]
        [0, 0, 0, 0, 1, 0, 1],   # A[0,1]
        [0, 1, 0, 0, 0, 1, 0],   # A[1,0]
        [1, 1, 0, 1, 0, 0, -1]   # A[1,1]
    ]

    # V[j*p+k, c]: coefficient of B[j,k] in product c
    # Rows: B00, B01, B10, B11
    V_data = [
        [1, 1, 0, -1, 0, 1, 0],  # B[0,0]
        [0, 0, 1, 0, 0, 1, 0],   # B[0,1]
        [0, 0, 0, 1, 0, 0, 1],   # B[1,0]
        [1, 0, -1, 0, 1, 0, 1]   # B[1,1]
    ]

    # W[k*n+i, c]: coefficient for C[i,k] from product c
    # W row index = k*n + i (official trace convention)
    # Row 0: k=0,i=0 -> C[0,0]: M1+M4-M5+M7 = [1,0,0,1,-1,0,1]
    # Row 1: k=0,i=1 -> C[1,0]: M2+M4 = [0,1,0,1,0,0,0]
    # Row 2: k=1,i=0 -> C[0,1]: M3+M5 = [0,0,1,0,1,0,0]
    # Row 3: k=1,i=1 -> C[1,1]: M1-M2+M3+M6 = [1,-1,1,0,0,1,0]
    W_data = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]

    # Use integer types for exactness
    U = np.array(U_data, dtype=int)
    V = np.array(V_data, dtype=int)
    W = np.array(W_data, dtype=int)

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
