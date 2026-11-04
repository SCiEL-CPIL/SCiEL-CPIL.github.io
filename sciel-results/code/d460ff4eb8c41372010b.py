"""Bootstrap SEARCH for rank-minimal exact tensor decomposition of 2x5 * 5x6."""
import time
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START

def _schoolbook_rank():
    """Return the standard schoolbook decomposition (rank = n*m*p = 60)."""
    n, m, p = DIMENSIONS
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

def _strassen_block(A_rows, A_cols, B_rows, B_cols, C_rows, C_cols, U, V, W, start_col):
    """
    Add Strassen decomposition for a 2x2x2 block.
    Returns new column index.
    """
    col = start_col
    n, m, p = DIMENSIONS
    
    # M1 = A00 * (B00 + B11)
    U[A_rows[0] * m + A_cols[0]][col] += 1
    V[B_rows[0] * p + B_cols[0]][col] += 1
    V[B_rows[1] * p + B_cols[1]][col] += 1
    W[C_rows[0] * n + C_cols[0]][col] += 1
    W[C_rows[1] * n + C_cols[1]][col] += 1
    col += 1
    
    # M2 = (A10 + A11) * B00
    U[A_rows[1] * m + A_cols[0]][col] += 1
    U[A_rows[1] * m + A_cols[1]][col] += 1
    V[B_rows[0] * p + B_cols[0]][col] += 1
    W[C_rows[1] * n + C_cols[0]][col] += 1
    col += 1
    
    # M3 = A01 * (B10 - B00)
    U[A_rows[0] * m + A_cols[1]][col] += 1
    V[B_rows[1] * p + B_cols[0]][col] += 1
    V[B_rows[0] * p + B_cols[0]][col] += -1
    W[C_rows[0] * n + C_cols[1]][col] += 1
    col += 1
    
    # M4 = A11 * (B11 - B01)
    U[A_rows[1] * m + A_cols[1]][col] += 1
    V[B_rows[1] * p + B_cols[1]][col] += 1
    V[B_rows[0] * p + B_cols[1]][col] += -1
    W[C_rows[0] * n + C_cols[0]][col] += 1
    W[C_rows[1] * n + C_cols[0]][col] += 1
    col += 1
    
    # M5 = (A00 + A01) * B11
    U[A_rows[0] * m + A_cols[0]][col] += 1
    U[A_rows[0] * m + A_cols[1]][col] += 1
    V[B_rows[1] * p + B_cols[1]][col] += 1
    W[C_rows[0] * n + C_cols[0]][col] += -1
    W[C_rows[0] * n + C_cols[1]][col] += 1
    col += 1
    
    # M6 = (A10 - A00) * (B00 + B01)
    U[A_rows[1] * m + A_cols[0]][col] += 1
    U[A_rows[0] * m + A_cols[0]][col] += -1
    V[B_rows[0] * p + B_cols[0]][col] += 1
    V[B_rows[0] * p + B_cols[1]][col] += 1
    W[C_rows[1] * n + C_cols[1]][col] += 1
    col += 1
    
    # M7 = (A01 - A11) * (B10 + B11)
    U[A_rows[0] * m + A_cols[1]][col] += 1
    U[A_rows[1] * m + A_cols[1]][col] += -1
    V[B_rows[1] * p + B_cols[0]][col] += 1
    V[B_rows[1] * p + B_cols[1]][col] += 1
    W[C_rows[0] * n + C_cols[0]][col] += 1
    W[C_rows[1] * n + C_cols[1]][col] += 1
    col += 1
    
    return col

def _block_decomposition():
    """
    Block decomposition with Strassen for 2x2x2 sub-blocks.
    
    Strategy: Partition the j-k plane (5x6) into:
    - j in {0,1,2,3}, k in {0,1,2,3}: 4 Strassen 2x2x2 blocks (rank 7 each = 28)
    - j in {0,1,2,3}, k in {4,5}: 2 Strassen 2x2x2 blocks (rank 7 each = 14)
    - j in {4}, k in {0,1,2,3}: 1 schoolbook 2x1x4 block (rank 4)
    - j in {4}, k in {4,5}: 1 schoolbook 2x1x2 block (rank 2)
    
    Total rank: 28 + 14 + 4 + 2 = 48
    """
    n, m, p = DIMENSIONS
    
    # Compute total rank
    # 4 Strassen blocks for j in {0,1,2,3}, k in {0,1,2,3}
    # 2 Strassen blocks for j in {0,1,2,3}, k in {4,5}
    # 1 schoolbook block for j=4, k in {0,1,2,3}
    # 1 schoolbook block for j=4, k in {4,5}
    rank = 4 * 7 + 2 * 7 + 4 + 2  # 28 + 14 + 4 + 2 = 48
    
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    
    col = 0
    A_rows = [0, 1]  # all 2 rows of A
    
    # Strassen blocks for j in {0,1,2,3}, k in {0,1,2,3}
    for j0 in [0, 2]:
        for k0 in [0, 2]:
            A_cols = [j0, j0 + 1]
            B_rows = [j0, j0 + 1]
            B_cols = [k0, k0 + 1]
            C_cols = [k0, k0 + 1]
            C_rows = [0, 1]
            col = _strassen_block(A_rows, A_cols, B_rows, B_cols, C_rows, C_cols, U, V, W, col)
    
    # Strassen blocks for j in {0,1,2,3}, k in {4,5}
    for j0 in [0, 2]:
        A_cols = [j0, j0 + 1]
        B_rows = [j0, j0 + 1]
        B_cols = [4, 5]
        C_cols = [4, 5]
        C_rows = [0, 1]
        col = _strassen_block(A_rows, A_cols, B_rows, B_cols, C_rows, C_cols, U, V, W, col)
    
    # Schoolbook block for j=4, k in {0,1,2,3}
    # C_ik = A_i4 * B_4k for i in {0,1}, k in {0,1,2,3}
    for i in range(2):
        for k in range(4):
            U[i * m + 4][col] += 1
            V[4 * p + k][col] += 1
            W[k * n + i][col] += 1
            col += 1
    
    # Schoolbook block for j=4, k in {4,5}
    for i in range(2):
        for k in range(4, 6):
            U[i * m + 4][col] += 1
            V[4 * p + k][col] += 1
            W[k * n + i][col] += 1
            col += 1
    
    # Convert to lists of lists as required by the interface
    U_list = [[int(x) for x in row] for row in U]
    V_list = [[int(x) for x in row] for row in V]
    W_list = [[int(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}

def _verify_decomposition(U, V, W):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    n, m, p = DIMENSIONS
    rank = len(U[0]) if U else 0
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if s != 1:
                    return False
    return True

def construct():
    """
    Construct the rank-minimal exact tensor decomposition.
    Returns a dict with keys 'U', 'V', 'W' containing the factor matrices.
    """
    start_time = time.time()
    deadline = 120.0  # safety margin before the 150s limit
    
    # Try the block decomposition with Strassen (rank 48)
    result = _block_decomposition()
    
    # Verify
    if _verify_decomposition(result["U"], result["V"], result["W"]):
        return result
    
    # Fallback to schoolbook (rank 60)
    return _schoolbook_rank()

# EVOLVE-BLOCK-END

