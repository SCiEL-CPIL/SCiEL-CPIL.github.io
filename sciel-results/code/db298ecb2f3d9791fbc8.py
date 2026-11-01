import time

DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-7 exact tensor decomposition for 2x2 matrix multiplication.
    Uses Strassen's algorithm decomposition, which is rank-minimal for 2x2 matrices.
    
    The decomposition is defined by matrices U, V, W such that:
    T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    For 2x2 matrix multiplication, the minimal rank is 7.
    """
    # Safety margin, though computation is instant
    start_time = time.time()
    deadline = start_time + 10.0
    
    n, m, p = DIMENSIONS  # 2, 2, 2
    rank = 7

    # Strassen's 7 products coefficients
    # U rows: A00 (i=0,j=0), A01 (i=0,j=1), A10 (i=1,j=0), A11 (i=1,j=1)
    U_data = [
        [1, 0, 1, 0, 1, -1, 0],  # A00
        [0, 0, 0, 0, 1, 0, 1],   # A01
        [0, 1, 0, 0, 0, 1, 0],   # A10
        [1, 1, 0, 1, 0, 0, -1]   # A11
    ]

    # V rows: B00 (j=0,k=0), B01 (j=0,k=1), B10 (j=1,k=0), B11 (j=1,k=1)
    V_data = [
        [1, 1, 0, -1, 0, 1, 0],  # B00
        [0, 0, 1, 0, 0, 1, 0],   # B01
        [0, 0, 0, 1, 0, 0, 1],   # B10
        [1, 0, -1, 0, 1, 0, 1]   # B11
    ]

    # W rows indexed by k*n + i:
    # Index 0: k=0, i=0 -> C00
    # Index 1: k=0, i=1 -> C10
    # Index 2: k=1, i=0 -> C01
    # Index 3: k=1, i=1 -> C11
    
    # Formulas:
    # C00 = P1 + P4 - P5 + P7
    # C10 = P2 + P4
    # C01 = P3 + P5
    # C11 = P1 - P2 + P3 + P6
    
    W_data = [
        [1, 0, 0, 1, -1, 0, 1],  # C00
        [0, 1, 0, 1, 0, 0, 0],   # C10
        [0, 0, 1, 0, 1, 0, 0],   # C01
        [1, -1, 1, 0, 0, 1, 0]   # C11
    ]

    # Check deadline (should never trigger)
    if time.time() > deadline:
        pass

    # Return lists directly as per interface specification
    return {"U": U_data, "V": V_data, "W": W_data}
# EVOLVE-BLOCK-END
