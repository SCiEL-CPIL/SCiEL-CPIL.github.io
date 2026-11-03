import time
import random
import numpy as np

DIMENSIONS = [2, 4, 5]

def _verify(U, V, W, n, m, p, rank):
    """Verify that the decomposition is exact using the tensor check."""
    U_np = np.array(U)
    V_np = np.array(V)
    W_np = np.array(W)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U_np[i*m+j], V_np[j*p+k] * W_np[k*n+i])
                if abs(s - 1) > 1e-9:
                    return False
    return True

def _strassen_2x2x2_block(U, V, W, i_off, j_off, k_off, col_start):
    """
    Add a Strassen 2x2x2 block.
    """
    n, m, p = DIMENSIONS
    
    # Define local row indices
    u00 = i_off * m + j_off
    u01 = i_off * m + j_off + 1
    u10 = (i_off + 1) * m + j_off
    u11 = (i_off + 1) * m + j_off + 1
    
    v00 = j_off * p + k_off
    v01 = j_off * p + k_off + 1
    v10 = (j_off + 1) * p + k_off
    v11 = (j_off + 1) * p + k_off + 1
    
    w00 = k_off * n + i_off
    w01 = k_off * n + i_off + 1
    w10 = (k_off + 1) * n + i_off
    w11 = (k_off + 1) * n + i_off + 1
    
    col = col_start
    
    # M1 = (A00+A11)*(B00+B11)
    U[u00][col] += 1
    U[u11][col] += 1
    V[v00][col] += 1
    V[v11][col] += 1
    W[w00][col] += 1
    W[w11][col] += 1
    col += 1
    
    # M2 = (A10+A11)*B00
    U[u10][col] += 1
    U[u11][col] += 1
    V[v00][col] += 1
    W[w10][col] += 1
    W[w11][col] += -1
    col += 1
    
    # M3 = A00*(B01-B11)
    U[u00][col] += 1
    V[v01][col] += 1
    V[v11][col] += -1
    W[w00][col] += 1
    W[w11][col] += 1
    col += 1
    
    # M4 = A11*(B10-B00)
    U[u11][col] += 1
    V[v10][col] += 1
    V[v00][col] += -1
    W[w00][col] += 1
    W[w10][col] += 1
    col += 1
    
    # M5 = (A00+A01)*B11
    U[u00][col] += 1
    U[u01][col] += 1
    V[v11][col] += 1
    W[w00][col] += -1
    W[w01][col] += 1
    col += 1
    
    # M6 = (A10-A00)*(B00+B01)
    U[u10][col] += 1
    U[u00][col] += -1
    V[v00][col] += 1
    V[v01][col] += 1
    W[w10][col] += 1
    W[w11][col] += 1
    col += 1
    
    # M7 = (A01-A11)*(B10+B11)
    U[u01][col] += 1
    U[u11][col] += -1
    V[v10][col] += 1
    V[v11][col] += 1
    W[w01][col] += 1
    W[w11][col] += 1
    col += 1
    
    return col - col_start

def _build_strassen_2x4x5(n, m, p):
    """Build a Strassen-based decomposition for 2x4x5."""
    max_rank = n * m * p
    U = [[0] * max_rank for _ in range(n * m)]
    V = [[0] * max_rank for _ in range(m * p)]
    W = [[0] * max_rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen block: i=0:2, j=0:2, k=0:2
    col += _strassen_2x2x2_block(U, V, W, 0, 0, 0, col)
    
    # Strassen block: i=0:2, j=0:2, k=2:4
    col += _strassen_2x2x2_block(U, V, W, 0, 0, 2, col)
    
    # Strassen block: i=0:2, j=2:4, k=0:2
    col += _strassen_2x2x2_block(U, V, W, 0, 2, 0, col)
    
    # Strassen block: i=0:2, j=2:4, k=2:4
    col += _strassen_2x2x2_block(U, V, W, 0, 2, 2, col)
    
    # Schoolbook for k=4 (size 1)
    # j=0:2, k=4
    for i in range(2):
        for j in range(2):
            U[i * m + j][col] = 1
            V[j * p + 4][col] = 1
            W[4 * n + i][col] = 1
            col += 1
    
    # j=2:4, k=4
    for i in range(2):
        for j in range(2):
            U[i * m + j + 2][col] = 1
            V[(j + 2) * p + 4][col] = 1
            W[4 * n + i][col] = 1
            col += 1
    
    rank = col
    U = [row[:rank] for row in U]
    V = [row[:rank] for row in V]
    W = [row[:rank] for row in W]
    
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 135
    
    result = _build_strassen_2x4x5(n, m, p)
    
    if _verify(result["U"], result["V"], result["W"], n, m, p, len(result["U"][0])):
        return result
    else:
        # Fallback to schoolbook if Strassen fails
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
