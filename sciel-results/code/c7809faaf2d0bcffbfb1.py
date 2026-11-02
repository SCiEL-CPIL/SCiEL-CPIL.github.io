"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 2]
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-7 exact tensor decomposition for 2x2 matrix multiplication.
    Uses Strassen's algorithm decomposition.
    """
    n, m, p = DIMENSIONS  # 2, 2, 2
    
    # Strassen's 7 products:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)

    # U[i*m+j, c]: coefficient of A[i,j] in product c
    # Rows: A[0,0], A[0,1], A[1,0], A[1,1]
    U_data = [
        [1, 0, 1, 0, 1, -1, 0],   # A[0,0]
        [0, 0, 0, 0, 1, 0, 1],    # A[0,1]
        [0, 1, 0, 0, 0, 1, 0],    # A[1,0]
        [1, 1, 0, 1, 0, 0, -1]    # A[1,1]
    ]

    # V[j*p+k, c]: coefficient of B[j,k] in product c
    # Rows: B[0,0], B[0,1], B[1,0], B[1,1]
    V_data = [
        [1, 1, 0, -1, 0, 1, 0],   # B[0,0]
        [0, 0, 1, 0, 0, 1, 0],    # B[0,1]
        [0, 0, 0, 1, 0, 0, 1],    # B[1,0]
        [1, 0, -1, 0, 1, 0, 1]    # B[1,1]
    ]

    # W[k*n+i, c]: coefficient for C[i,k] from product c
    # Index k*n + i:
    # Row 0: k=0, i=0 -> C[0,0]
    # Row 1: k=0, i=1 -> C[1,0]
    # Row 2: k=1, i=0 -> C[0,1]
    # Row 3: k=1, i=1 -> C[1,1]
    
    # C[0,0] = M1 - M5 + M7 + M4
    # C[1,0] = M2 + M4
    # C[0,1] = M3 + M5
    # C[1,1] = M1 - M2 + M3 + M6
    
    W_data = [
        [1, 0, 0, 1, -1, 0, 1],   # C[0,0]
        [0, 1, 0, 1, 0, 0, 0],    # C[1,0]
        [0, 0, 1, 0, 1, 0, 0],    # C[0,1]
        [1, -1, 1, 0, 0, 1, 0]    # C[1,1]
    ]

    U = np.array(U_data, dtype=int)
    V = np.array(V_data, dtype=int)
    W = np.array(W_data, dtype=int)

    # Verify correctness: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    # This is the tensor identity for matrix multiplication.
    is_valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for c in range(7):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if val != 1:
                    is_valid = False
                    break
            if not is_valid:
                break
        if not is_valid:
            break

    if is_valid:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to rank 8 (schoolbook) if verification fails (should not happen)
        rank = 8
        U_fb = np.zeros((n*m, rank), dtype=int)
        V_fb = np.zeros((m*p, rank), dtype=int)
        W_fb = np.zeros((p*n, rank), dtype=int)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j, col] = 1
                    V_fb[j*p+k, col] = 1
                    W_fb[k*n+i, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
# EVOLVE-BLOCK-END
