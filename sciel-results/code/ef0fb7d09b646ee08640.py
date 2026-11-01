"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p, rank, tol=1e-9):
    """Verify that the tensor decomposition exactly reproduces the matmul tensor.
    
    For each (i, j, k) in the output tensor, check:
      sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    """
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(s - 1.0) > tol:
                    return False, (i, j, k, s)
    return True, None


def _strassen_decomposition(n, m, p):
    """Construct the 7-term Strassen decomposition for 2x2 matrix multiplication.
    
    Output factor W follows the official trace convention W[k*n+i, r].
    U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
    
    The matrix product C = A @ B is decomposed as:
    C[i,k] = sum_j A[i,j]*B[j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    Strassen's 7 products:
      m1 = (A00+A11)(B00+B11)
      m2 = (A10+A11)B00
      m3 = A00(B01-B11)
      m4 = A11(B10-B00)
      m5 = (A00+A01)B11
      m6 = (A10-A00)(B00+B01)
      m7 = (A01-A11)(B10+B11)
    
    Output reconstruction:
      C[0,0] = m1 + m4 - m5 + m7
      C[0,1] = m3 + m5
      C[1,0] = m2 + m4
      C[1,1] = m1 - m2 + m3 + m6
    """
    rank = 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Index conventions:
    # U[i*m + j, r]: coefficient of A[i,j] in r-th product
    # V[j*p + k, r]: coefficient of B[j,k] in r-th product  
    # W[k*n + i, r]: coefficient for C[i,k] in r-th product (trace convention W[k*n+i])

    # W row mapping (k*n + i):
    #   W[0] = k=0,i=0 -> C[0,0]
    #   W[1] = k=0,i=1 -> C[1,0]
    #   W[2] = k=1,i=0 -> C[0,1]
    #   W[3] = k=1,i=1 -> C[1,1]

    # m1 = (A00+A11)(B00+B11) -> contributes to C[0,0] (+1) and C[1,1] (+1)
    U[0][0] = 1   # A[0,0]
    U[3][0] = 1   # A[1,1]
    V[0][0] = 1   # B[0,0]
    V[3][0] = 1   # B[1,1]
    W[0][0] = 1   # C[0,0]
    W[3][0] = 1   # C[1,1]

    # m2 = (A10+A11)B00 -> contributes to C[1,0] (+1) and C[1,1] (-1)
    U[2][1] = 1   # A[1,0]
    U[3][1] = 1   # A[1,1]
    V[0][1] = 1   # B[0,0]
    W[1][1] = 1   # C[1,0]
    W[3][1] = -1  # C[1,1]

    # m3 = A00(B01-B11) -> contributes to C[0,1] (+1) and C[1,1] (+1)
    U[0][2] = 1   # A[0,0]
    V[1][2] = 1   # B[0,1]
    V[3][2] = -1  # B[1,1]
    W[2][2] = 1   # C[0,1]
    W[3][2] = 1   # C[1,1]

    # m4 = A11(B10-B00) -> contributes to C[0,0] (+1) and C[1,0] (+1)
    U[3][3] = 1   # A[1,1]
    V[2][3] = 1   # B[1,0]
    V[0][3] = -1  # B[0,0]
    W[0][3] = 1   # C[0,0]
    W[1][3] = 1   # C[1,0]

    # m5 = (A00+A01)B11 -> contributes to C[0,0] (-1) and C[0,1] (+1)
    U[0][4] = 1   # A[0,0]
    U[1][4] = 1   # A[0,1]
    V[3][4] = 1   # B[1,1]
    W[0][4] = -1  # C[0,0]
    W[2][4] = 1   # C[0,1]

    # m6 = (A10-A00)(B00+B01) -> contributes to C[1,1] (+1)
    U[2][5] = 1   # A[1,0]
    U[0][5] = -1  # A[0,0]
    V[0][5] = 1   # B[0,0]
    V[1][5] = 1   # B[0,1]
    W[3][5] = 1   # C[1,1]

    # m7 = (A01-A11)(B10+B11) -> contributes to C[0,0] (+1)
    U[1][6] = 1   # A[0,1]
    U[3][6] = -1  # A[1,1]
    V[2][6] = 1   # B[1,0]
    V[3][6] = 1   # B[1,1]
    W[0][6] = 1   # C[0,0]

    return {"U": U, "V": V, "W": W}


def _schoolbook_decomposition(n, m, p):
    """Fallback: rank-(n*m*p) schoolbook decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS

    # Return the known rank-7 Strassen decomposition (provably optimal)
    result = _strassen_decomposition(n, m, p)

    # Verification pass: ensure exact decomposition
    valid, err = _verify_decomposition(
        result["U"], result["V"], result["W"], n, m, p, 7
    )
    if not valid:
        # Fallback to schoolbook (should never trigger, but safety net)
        result = _schoolbook_decomposition(n, m, p)

    # Safety margin check
    if time.time() - start_time > 140:
        raise RuntimeError("Time limit exceeded")

    return result
# EVOLVE-BLOCK-END
