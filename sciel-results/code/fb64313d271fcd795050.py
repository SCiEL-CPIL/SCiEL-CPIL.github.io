"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _verify(U, V, W, n, m, p, rank):
    """Verify that the decomposition is exact using the tensor check."""
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should equal 1 for all i,j,k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1) > 1e-9:
                    return False
    return True

def _try_rank(rank, n, m, p, deadline, max_iters=20000):
    """Try to find a rank-r decomposition with integer/half-integer entries."""
    vals = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Sparse random initialization: each row has a few non-zero entries
        nnz_per_row = max(1, rank // 4)
        
        U = [[0.0] * rank for _ in range(n*m)]
        V = [[0.0] * rank for _ in range(m*p)]
        W = [[0.0] * rank for _ in range(p*n)]
        
        for row in range(n*m):
            cols = random.sample(range(rank), min(nnz_per_row, rank))
            for col in cols:
                U[row][col] = random.choice(vals)
        
        for row in range(m*p):
            cols = random.sample(range(rank), min(nnz_per_row, rank))
            for col in cols:
                V[row][col] = random.choice(vals)
        
        for row in range(p*n):
            cols = random.sample(range(rank), min(nnz_per_row, rank))
            for col in cols:
                W[row][col] = random.choice(vals)
        
        # Fast check using numpy
        U_np = np.array(U)
        V_np = np.array(V)
        W_np = np.array(W)
        
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U_np[i*m+j], V_np[j*p+k] * W_np[k*n+i])
                    if abs(s - 1) > 1e-9:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            return U, V, W
    
    return None

def _schoolbook(n, m, p):
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

def _strassen_2x2x2_block(U, V, W, i_off, j_off, k_off, col_start):
    """
    Add a Strassen 2x2x2 block.
    i_off: offset in i dimension (0 or 1)
    j_off: offset in j dimension (0 or 1)
    k_off: offset in k dimension (0 or 1)
    col_start: starting column index
    Returns: number of columns used (7)
    
    U rows are indexed by i*m + j
    V rows are indexed by j*p + k
    W rows are indexed by k*n + i
    
    For a 2x2x2 block at offsets (i_off, j_off, k_off):
    Local indices:
    A[0,0] -> U[i_off*m + j_off, col]
    A[0,1] -> U[i_off*m + j_off+1, col]
    A[1,0] -> U[(i_off+1)*m + j_off, col]
    A[1,1] -> U[(i_off+1)*m + j_off+1, col]
    
    B[0,0] -> V[j_off*p + k_off, col]
    B[0,1] -> V[j_off*p + k_off+1, col]
    B[1,0] -> V[(j_off+1)*p + k_off, col]
    B[1,1] -> V[(j_off+1)*p + k_off+1, col]
    
    C[0,0] -> W[k_off*n + i_off, col]
    C[0,1] -> W[k_off*n + i_off+1, col]
    C[1,0] -> W[(k_off+1)*n + i_off, col]
    C[1,1] -> W[(k_off+1)*n + i_off+1, col]
    """
    n, m, p = DIMENSIONS
    
    # Define local row indices
    u00 = i_off * m + j_off       # A[0,0]
    u01 = i_off * m + j_off + 1   # A[0,1]
    u10 = (i_off + 1) * m + j_off       # A[1,0]
    u11 = (i_off + 1) * m + j_off + 1   # A[1,1]
    
    v00 = j_off * p + k_off       # B[0,0]
    v01 = j_off * p + k_off + 1   # B[0,1]
    v10 = (j_off + 1) * p + k_off       # B[1,0]
    v11 = (j_off + 1) * p + k_off + 1   # B[1,1]
    
    w00 = k_off * n + i_off       # C[0,0]
    w01 = k_off * n + i_off + 1   # C[0,1]
    w10 = (k_off + 1) * n + i_off       # C[1,0]
    w11 = (k_off + 1) * n + i_off + 1   # C[1,1]
    
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

def _strassen_decomposition(n, m, p):
    """
    Build a Strassen-based decomposition for n x m x p multiplication.
    For 2x4x5, we can use:
    - Split j dimension (size 4) into two blocks of size 2
    - Split k dimension (size 5) into blocks of size 2, 2, 1
    - Use Strassen for 2x2x2 blocks and schoolbook for remaining
    """
    rank = 0
    U = [[0] * 100 for _ in range(n * m)]  # Pre-allocate large enough
    V = [[0] * 100 for _ in range(m * p)]
    W = [[0] * 100 for _ in range(p * n)]
    
    # We'll build the decomposition by processing blocks
    # For 2x4x5:
    # i has size 2 (one block of 2)
    # j has size 4 (two blocks of 2)
    # k has size 5 (two blocks of 2, one block of 1)
    
    col = 0
    # Block (i=0:2, j=0:2, k=0:2) - Strassen 2x2x2
    col += _strassen_2x2x2_block(U, V, W, 0, 0, 0, col)
    
    # Block (i=0:2, j=0:2, k=2:4) - Strassen 2x2x2
    col += _strassen_2x2x2_block(U, V, W, 0, 0, 2, col)
    
    # Block (i=0:2, j=2:4, k=0:2) - Strassen 2x2x2
    col += _strassen_2x2x2_block(U, V, W, 0, 2, 0, col)
    
    # Block (i=0:2, j=2:4, k=2:4) - Strassen 2x2x2
    col += _strassen_2x2x2_block(U, V, W, 0, 2, 2, col)
    
    # Block (i=0:2, j=0:2, k=4:5) - schoolbook 2x2x1 (4 products)
    for j in range(2):
        for k in range(1):
            for i in range(2):
                U[i * m + j][col] = 1
                V[j * p + k + 4][col] = 1
                W[(k + 4) * n + i][col] = 1
                col += 1
    
    # Block (i=0:2, j=2:4, k=4:5) - schoolbook 2x2x1 (4 products)
    for j in range(2):
        for k in range(1):
            for i in range(2):
                U[i * m + j + 2][col] = 1
                V[(j + 2) * p + k + 4][col] = 1
                W[(k + 4) * n + i][col] = 1
                col += 1
    
    rank = col
    # Trim arrays to actual rank
    U = [row[:rank] for row in U]
    V = [row[:rank] for row in V]
    W = [row[:rank] for row in W]
    
    return {"U": U, "V": "V", "W": W}

def _build_strassen_2x4x5(n, m, p):
    """Build a Strassen-based decomposition for 2x4x5."""
    # Initialize with zeros
    max_rank = n * m * p
    U = [[0] * max_rank for _ in range(n * m)]
    V = [[0] * max_rank for _ in range(m * p)]
    W = [[0] * max_rank for _ in range(p * n)]
    
    col = 0
    
    # Use Strassen for 2x2x2 sub-blocks
    # j dimension: 0-1, 2-3
    # k dimension: 0-1, 2-3, 4
    
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
    deadline = start_time + 135  # Safety margin before 150s kill
    
    # Start with schoolbook as fallback
    best_result = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try Strassen-based decomposition first
    strassen_result = _build_strassen_2x4x5(n, m, p)
    strassen_rank = len(strassen_result["U"][0])
    
    # Verify Strassen result
    if _verify(strassen_result["U"], strassen_result["V"], strassen_result["W"], n, m, p, strassen_rank):
        if strassen_rank < best_rank:
            best_result = strassen_result
            best_rank = strassen_rank
    else:
        # Strassen failed verification, try to fix or fall back
        pass
    
    # Try random search for even lower ranks
    for rank in range(max(18, best_rank - 5), best_rank):
        if time.time() > deadline:
            break
        
        remaining = deadline - time.time()
        if remaining < 2:
            break
        
        iters = min(50000, max(5000, int(remaining * 300)))
        
        result = _try_rank(rank, n, m, p, deadline, max_iters=iters)
        if result is not None:
            U, V, W = result
            best_result = {"U": U, "V": V, "W": W}
            best_rank = rank
            break
    
    return best_result
# EVOLVE-BLOCK-END
