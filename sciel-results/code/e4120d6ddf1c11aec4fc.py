"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def _strassen_decomposition(n, m, p):
    """Construct the 7-term Strassen decomposition for 2x2 matrix multiplication.
    
    Output factor W follows the official trace convention W[k*n+i, r].
    U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
    
    The matrix product C = A @ B is decomposed as:
    C = sum_r U[:, r] * W[:, r] * V[:, r]
    where the tensor T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    """
    rank = 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Index conventions:
    # U[i*m + j, r]: A[i,j] coefficient in r-th product
    # V[j*p + k, r]: B[j,k] coefficient in r-th product  
    # W[k*n + i, r]: C[i,k] coefficient in r-th product (trace convention W[k*n+i])

    # P1 = (A00 + A11)(B00 + B11)
    U[0][0] = 1   # A[0,0]
    U[3][0] = 1   # A[1,1]
    V[0][0] = 1   # B[0,0]
    V[3][0] = 1   # B[1,1]
    W[0][0] = 1   # C[0,0] -> k=0, i=0
    W[3][0] = 1   # C[1,1] -> k=1, i=1

    # P2 = (A10 + A11)B00
    U[2][1] = 1   # A[1,0]
    U[3][1] = 1   # A[1,1]
    V[0][1] = 1   # B[0,0]
    W[1][1] = 1   # C[1,0] -> k=0, i=1
    W[3][1] = -1  # C[1,1] -> k=1, i=1

    # P3 = A00(B01 - B11)
    U[0][2] = 1   # A[0,0]
    V[1][2] = 1   # B[0,1]
    V[3][2] = -1  # B[1,1]
    W[2][2] = 1   # C[0,1] -> k=1, i=0
    W[3][2] = 1   # C[1,1] -> k=1, i=1

    # P4 = A11(B10 - B00)
    U[3][3] = 1   # A[1,1]
    V[2][3] = 1   # B[1,0]
    V[0][3] = -1  # B[0,0]
    W[0][3] = 1   # C[0,0] -> k=0, i=0
    W[1][3] = 1   # C[1,0] -> k=0, i=1

    # P5 = (A00 + A01)B11
    U[0][4] = 1   # A[0,0]
    U[1][4] = 1   # A[0,1]
    V[3][4] = 1   # B[1,1]
    W[0][4] = -1  # C[0,0] -> k=0, i=0
    W[2][4] = 1   # C[0,1] -> k=1, i=0

    # P6 = (A10 - A00)(B00 + B01)
    U[2][5] = 1   # A[1,0]
    U[0][5] = -1  # A[0,0]
    V[0][5] = 1   # B[0,0]
    V[1][5] = 1   # B[0,1]
    W[3][5] = 1   # C[1,1] -> k=1, i=1

    # P7 = (A01 - A11)(B10 + B11)
    U[1][6] = 1   # A[0,1]
    U[3][6] = -1  # A[1,1]
    V[2][6] = 1   # B[1,0]
    V[3][6] = 1   # B[1,1]
    W[0][6] = 1   # C[0,0] -> k=0, i=0

    return {"U": U, "V": V, "W": W}

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Directly return the known rank-7 Strassen decomposition
    result = _strassen_decomposition(n, m, p)
    
    # Safety margin check (should complete nearly instantly)
    if time.time() - start_time > 140:
        raise RuntimeError("Time limit exceeded")
    
    return result
# EVOLVE-BLOCK-END
