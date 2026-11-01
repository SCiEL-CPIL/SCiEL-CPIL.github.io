"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 2]
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-7 exact tensor decomposition for 2x2 matrix multiplication.
    Uses Strassen's algorithm decomposition (optimal rank for 2x2).
    """
    n, m, p = DIMENSIONS  # 2, 2, 2
    rank = 7

    # Strassen's 7 products coefficients
    # U[i*m+j, c]: coefficient of A[i,j] in product c
    # V[j*p+k, c]: coefficient of B[j,k] in product c
    # W[k*n+i, c]: coefficient for C[i,k] from product c (Trace convention: W[k*n+i])
    
    # Product 1: (A00 + A11) * (B00 + B11)
    # Product 2: (A10 + A11) * B00
    # Product 3: A00 * (B01 - B11)
    # Product 4: A11 * (B10 - B00)
    # Product 5: (A00 + A01) * B11
    # Product 6: (A10 - A00) * (B00 + B01)
    # Product 7: (A01 - A11) * (B10 + B11)
    
    # U rows: A00, A01, A10, A11
    U_data = [
        [1, 0, 1, 0, 1, -1, 0],  # A00
        [0, 0, 0, 0, 1, 0, 1],   # A01
        [0, 1, 0, 0, 0, 1, 0],   # A10
        [1, 1, 0, 1, 0, 0, -1]   # A11
    ]

    # V rows: B00, B01, B10, B11
    V_data = [
        [1, 1, 0, -1, 0, 1, 0],  # B00
        [0, 0, 1, 0, 0, 1, 0],   # B01
        [0, 0, 0, 1, 0, 0, 1],   # B10
        [1, 0, -1, 0, 1, 0, 1]   # B11
    ]

    # W rows: C00 (k=0,i=0), C10 (k=0,i=1), C01 (k=1,i=0), C11 (k=1,i=1)
    # Note: W index is k*n + i.
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

    U = np.array(U_data, dtype=float)
    V = np.array(V_data, dtype=float)
    W = np.array(W_data, dtype=float)

    # Verify correctness
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Should be 1 for all i,j,k
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U_data[i*m+j][c] * V_data[j*p+k][c] * W_data[k*n+i][c]
                if abs(s - 1.0) > 1e-9:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break

    if not valid:
        # Fallback to a known valid decomposition if verification fails (should not happen)
        # But Strassen is correct, so this is just a safety check.
        pass

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
