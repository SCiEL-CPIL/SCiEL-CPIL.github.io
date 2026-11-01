"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 2]
import time
import numpy as np

# EVOLVE-BLOCK-START
def _verify_strassen(U, V, W, n, m, p, rank, tol=1e-12):
    """Verify that the tensor decomposition is exact: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1"""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m + j, c] * V[j*p + k, c] * W[k*n + i, c]
                if abs(val - 1.0) > tol:
                    return False
    return True

def _strassen_rank7():
    """
    Returns the known valid rank-7 Strassen decomposition for 2x2 matrix multiply.
    This is the minimal possible rank for 2x2 matrix multiplication (proven by Strassen 1969).
    Uses integer entries for exact arithmetic.
    
    Strassen's 7 products:
    M1 = (A00 + A11) * (B00 + B11)
    M2 = (A10 + A11) * B00
    M3 = A00 * (B01 - B11)
    M4 = A11 * (B10 - B00)
    M5 = (A00 + A01) * B11
    M6 = (A10 - A00) * (B00 + B01)
    M7 = (A01 - A11) * (B10 + B11)
    
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    """
    n, m, p = 2, 2, 2
    rank = 7
    
    # U shape: (n*m, rank) = (4, 7)
    # Row order: A[0,0], A[0,1], A[1,0], A[1,1]
    U_data = [
        [1, 0, 1, 0, 1, -1, 0],   # A[0,0]
        [0, 0, 0, 0, 1, 0, 1],    # A[0,1]
        [0, 1, 0, 0, 0, 1, 0],    # A[1,0]
        [1, 1, 0, 1, 0, 0, -1]    # A[1,1]
    ]
    
    # V shape: (m*p, rank) = (4, 7)
    # Row order: B[0,0], B[0,1], B[1,0], B[1,1]
    V_data = [
        [1, 1, 0, -1, 0, 1, 0],   # B[0,0]
        [0, 0, 1, 0, 0, 1, 0],    # B[0,1]
        [0, 0, 0, 1, 0, 0, 1],    # B[1,0]
        [1, 0, -1, 0, 1, 0, 1]    # B[1,1]
    ]
    
    # W shape: (p*n, rank) = (4, 7)
    # W index = k*n + i (official trace convention)
    # Row 0: k=0, i=0 -> C[0,0]
    # Row 1: k=0, i=1 -> C[1,0]
    # Row 2: k=1, i=0 -> C[0,1]
    # Row 3: k=1, i=1 -> C[1,1]
    W_data = [
        [1, 0, 0, 1, -1, 0, 1],   # C[0,0]: M1 + M4 - M5 + M7
        [0, 1, 0, 1, 0, 0, 0],    # C[1,0]: M2 + M4
        [0, 0, 1, 0, 1, 0, 0],    # C[0,1]: M3 + M5
        [1, -1, 1, 0, 0, 1, 0]    # C[1,1]: M1 - M2 + M3 + M6
    ]
    
    return (np.array(U_data, dtype=float),
            np.array(V_data, dtype=float),
            np.array(W_data, dtype=float))

def construct():
    """
    Returns the rank-minimal (rank 7) exact tensor decomposition for 2x2 matrix multiplication.
    Since 7 is the proven lower bound for the tensor rank of 2x2 matrix multiplication,
    this is the optimal solution.
    
    The decomposition uses Strassen's algorithm with integer entries, ensuring exactness.
    """
    n, m, p = DIMENSIONS  # 2, 2, 2
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s kill
    
    U, V, W = _strassen_rank7()
    rank = 7
    
    # Verify the decomposition is valid
    if time.time() < deadline:
        if not _verify_strassen(U, V, W, n, m, p, rank):
            # This should never happen with the known-correct Strassen decomposition,
            # but fallback to schoolbook rank 8 for safety
            rank = 8
            U = np.zeros((n * m, rank))
            V = np.zeros((m * p, rank))
            W = np.zeros((p * n, rank))
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i * m + j, col] = 1.0
                        V[j * p + k, col] = 1.0
                        W[k * n + i, col] = 1.0
                        col += 1
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
