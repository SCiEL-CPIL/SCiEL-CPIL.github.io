import numpy as np
import time

DIMENSIONS = [2, 3, 10]
TIME_LIMIT = 110  # seconds
_START_TIME = time.time()

def _time_remaining():
    return TIME_LIMIT - (time.time() - _START_TIME)

def construct():
    """
    Returns a rank-55 exact tensor decomposition for 2x3 by 3x10 matrix multiplication.
    Strategy:
    Split the 10 columns of B into 5 blocks of 2 columns.
    For each block, split A into A_L (2x2) and A_R (2x1).
    Split the block of B into B_L (2x2, top 2 rows) and B_R (1x2, bottom row).
    C_block = A_L B_L + A_R B_R.
    Rank(A_L B_L) = 7 (Strassen).
    Rank(A_R B_R) = 4 (Outer product of 2x1 and 1x2).
    Total rank per block = 11.
    Total rank = 5 * 11 = 55.
    """
    if _time_remaining() < 5:
        return _construct_default()
    
    n, m, p = DIMENSIONS
    rank = 55

    # U: (n*m, rank) = (6, 55)
    # V: (m*p, rank) = (30, 55)
    # W: (p*n, rank) = (20, 55)
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    # A indices: i in {0,1}, j in {0,1,2} -> U index i*m + j
    a00, a01, a10, a11 = 0*m+0, 0*m+1, 1*m+0, 1*m+1
    a02, a12 = 0*m+2, 1*m+2

    # Iterate over 5 blocks of B columns
    for g in range(5):
        k0 = 2 * g
        k1 = 2 * g + 1
        
        # B indices: j in {0,1,2}, k in {k0, k1} -> V index j*p + k
        b00, b01 = 0*p+k0, 0*p+k1
        b10, b11 = 1*p+k0, 1*p+k1
        b20, b21 = 2*p+k0, 2*p+k1
        
        # C indices: i in {0,1}, k in {k0, k1} -> W index k*n + i
        c00, c01 = k0*n+0, k1*n+0
        c10, c11 = k0*n+1, k1*n+1
        
        # --- Part 1: Strassen for A_L (2x2) * B_L (2x2) ---
        # M1: (A00+A11)(B00+B11)
        U[a00, col] = 1; U[a11, col] = 1
        V[b00, col] = 1; V[b11, col] = 1
        W[c00, col] = 1; W[c11, col] = 1
        col += 1

        # M2: (A10+A11)B00
        U[a10, col] = 1; U[a11, col] = 1
        V[b00, col] = 1
        W[c10, col] = 1; W[c11, col] = -1
        col += 1

        # M3: A00(B01-B11)
        U[a00, col] = 1
        V[b01, col] = 1; V[b11, col] = -1
        W[c01, col] = 1; W[c11, col] = 1
        col += 1

        # M4: A11(B10-B00)
        U[a11, col] = 1
        V[b10, col] = 1; V[b00, col] = -1
        W[c00, col] = 1; W[c10, col] = 1
        col += 1

        # M5: (A00+A01)B11
        U[a00, col] = 1; U[a01, col] = 1
        V[b11, col] = 1
        W[c00, col] = -1; W[c01, col] = 1
        col += 1

        # M6: (A10-A00)(B00+B01)
        U[a10, col] = 1; U[a00, col] = -1
        V[b00, col] = 1; V[b01, col] = 1
        W[c11, col] = 1
        col += 1

        # M7: (A01-A11)(B10+B11)
        U[a01, col] = 1; U[a11, col] = -1
        V[b10, col] = 1; V[b11, col] = 1
        W[c00, col] = 1
        col += 1
        
        # --- Part 2: Schoolbook for A_R (2x1) * B_R (1x2) ---
        # P1: A02 * B20
        U[a02, col] = 1
        V[b20, col] = 1
        W[c00, col] = 1
        col += 1

        # P2: A02 * B21
        U[a02, col] = 1
        V[b21, col] = 1
        W[c01, col] = 1
        col += 1

        # P3: A12 * B20
        U[a12, col] = 1
        V[b20, col] = 1
        W[c10, col] = 1
        col += 1

        # P4: A12 * B21
        U[a12, col] = 1
        V[b21, col] = 1
        W[c11, col] = 1
        col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _construct_default():
    """Fallback: construct the rank-55 decomposition without time checks."""
    n, m, p = DIMENSIONS
    rank = 55

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    a00, a01, a10, a11 = 0, 1, 3, 4
    a02, a12 = 2, 5

    for g in range(5):
        k0 = 2 * g
        k1 = 2 * g + 1
        b00, b01 = 0*p+k0, 0*p+k1
        b10, b11 = 1*p+k0, 1*p+k1
        b20, b21 = 2*p+k0, 2*p+k1
        c00, c01 = k0*n+0, k1*n+0
        c10, c11 = k0*n+1, k1*n+1

        U[a00][col] = 1; U[a11][col] = 1
        V[b00][col] = 1; V[b11][col] = 1
        W[c00][col] = 1; W[c11][col] = 1
        col += 1

        U[a10][col] = 1; U[a11][col] = 1
        V[b00][col] = 1
        W[c10][col] = 1; W[c11][col] = -1
        col += 1

        U[a00][col] = 1
        V[b01][col] = 1; V[b11][col] = -1
        W[c01][col] = 1; W[c11][col] = 1
        col += 1

        U[a11][col] = 1
        V[b10][col] = 1; V[b00][col] = -1
        W[c00][col] = 1; W[c10][col] = 1
        col += 1

        U[a00][col] = 1; U[a01][col] = 1
        V[b11][col] = 1
        W[c00][col] = -1; W[c01][col] = 1
        col += 1

        U[a10][col] = 1; U[a00][col] = -1
        V[b00][col] = 1; V[b01][col] = 1
        W[c11][col] = 1
        col += 1

        U[a01][col] = 1; U[a11][col] = -1
        V[b10][col] = 1; V[b11][col] = 1
        W[c00][col] = 1
        col += 1

        U[a02][col] = 1; V[b20][col] = 1; W[c00][col] = 1
        col += 1
        U[a02][col] = 1; V[b21][col] = 1; W[c01][col] = 1
        col += 1
        U[a12][col] = 1; V[b20][col] = 1; W[c10][col] = 1
        col += 1
        U[a12][col] = 1; V[b21][col] = 1; W[c11][col] = 1
        col += 1

    return {"U": U, "V": V, "W": W}
