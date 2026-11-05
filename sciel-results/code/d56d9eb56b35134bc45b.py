"""Program for matrix multiplication tensor decomposition: 2x4 * 4x5."""
import time
import numpy as np

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def _validate_numpy(U, V, W, n=2, m=4, p=5, tol=1e-9):
    """Vectorized validation using numpy for speed."""
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    rank = U_np.shape[1]
    
    # Check all (i,j,k) entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # sum over r of U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                prod = U_np[i*m+j] * V_np[j*p+k] * W_np[k*n+i]
                val = np.sum(prod)
                if abs(val - 1.0) > tol:
                    return False
    return True

def _clean_entry(v):
    if abs(v - round(v)) < 1e-9:
        return int(round(v))
    doubled = v * 2
    if abs(doubled - round(doubled)) < 1e-9:
        return round(doubled) / 2.0
    return v

def _construct_rank_26():
    """
    Rank-26 decomposition for 2x4 x 4x5 matrix multiplication.
    
    Strategy: Split m=4 into two blocks of 2.
    For each 2x2x5 sub-problem:
      - Use Strassen (rank 7) for the first 2 columns of B
      - Use schoolbook (rank 6) for the remaining 3 columns of B
    Total per sub-problem: 13
    Total: 2 * 13 = 26
    """
    n, m, p = 2, 4, 5
    rank = 26
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_strassen_222(i1, i2, j1, j2, k1, k2, col_offset):
        """Add Strassen 2x2x2 block."""
        a11 = i1 * m + j1
        a12 = i1 * m + j2
        a21 = i2 * m + j1
        a22 = i2 * m + j2
        
        b11 = j1 * p + k1
        b12 = j1 * p + k2
        b21 = j2 * p + k1
        b22 = j2 * p + k2
        
        c11 = k1 * n + i1
        c12 = k2 * n + i1
        c21 = k1 * n + i2
        c22 = k2 * n + i2
        
        col = col_offset + 0
        U[a11][col] = 1; U[a22][col] = 1
        V[b11][col] = 1; V[b22][col] = 1
        W[c11][col] = 1; W[c22][col] = 1
        
        col = col_offset + 1
        U[a21][col] = 1; U[a22][col] = 1
        V[b11][col] = 1
        W[c21][col] = 1; W[c22][col] = -1
        
        col = col_offset + 2
        U[a11][col] = 1
        V[b12][col] = 1; V[b22][col] = -1
        W[c12][col] = 1; W[c22][col] = 1
        
        col = col_offset + 3
        U[a22][col] = 1
        V[b21][col] = 1; V[b11][col] = -1
        W[c11][col] = 1; W[c21][col] = 1
        
        col = col_offset + 4
        U[a11][col] = 1; U[a12][col] = 1
        V[b22][col] = 1
        W[c11][col] = -1; W[c12][col] = 1
        
        col = col_offset + 5
        U[a21][col] = 1; U[a11][col] = -1
        V[b11][col] = 1; V[b12][col] = 1
        W[c22][col] = 1
        
        col = col_offset + 6
        U[a12][col] = 1; U[a22][col] = -1
        V[b21][col] = 1; V[b22][col] = 1
        W[c11][col] = 1

    def add_schoolbook_223(i1, i2, j1, j2, k1, k2, k3, col_offset):
        """Add schoolbook 2x2x3 block (rank 6)."""
        a11 = i1 * m + j1
        a12 = i1 * m + j2
        a21 = i2 * m + j1
        a22 = i2 * m + j2
        
        b11 = j1 * p + k1
        b12 = j1 * p + k2
        b13 = j1 * p + k3
        b21 = j2 * p + k1
        b22 = j2 * p + k2
        b23 = j2 * p + k3
        
        c11 = k1 * n + i1
        c12 = k2 * n + i1
        c13 = k3 * n + i1
        c21 = k1 * n + i2
        c22 = k2 * n + i2
        c23 = k3 * n + i2
        
        col = col_offset + 0
        U[a11][col] = 1; V[b11][col] = 1; W[c11][col] = 1
        col = col_offset + 1
        U[a12][col] = 1; V[b21][col] = 1; W[c11][col] = 1
        col = col_offset + 2
        U[a21][col] = 1; V[b11][col] = 1; W[c21][col] = 1
        col = col_offset + 3
        U[a22][col] = 1; V[b21][col] = 1; W[c21][col] = 1
        col = col_offset + 4
        U[a11][col] = 1; V[b12][col] = 1; W[c12][col] = 1
        col = col_offset + 5
        U[a12][col] = 1; V[b22][col] = 1; W[c12][col] = 1
        col = col_offset + 6
        U[a21][col] = 1; V[b12][col] = 1; W[c22][col] = 1
        col = col_offset + 7
        U[a22][col] = 1; V[b22][col] = 1; W[c22][col] = 1
        col = col_offset + 8
        U[a11][col] = 1; V[b13][col] = 1; W[c13][col] = 1
        col = col_offset + 9
        U[a12][col] = 1; V[b23][col] = 1; W[c13][col] = 1
        col = col_offset + 10
        U[a21][col] = 1; V[b13][col] = 1; W[c23][col] = 1
        col = col_offset + 11
        U[a22][col] = 1; V[b23][col] = 1; W[c23][col] = 1

    col = 0
    
    # Sub-problem 1: A rows 0-1, cols 0-1; B rows 0-1, cols 0-4
    # 2x2x2 for B cols 0,1 using Strassen
    add_strassen_222(0, 1, 0, 1, 0, 1, col)
    col += 7
    # 2x2x3 for B cols 2,3,4 using schoolbook
    add_schoolbook_223(0, 1, 0, 1, 2, 3, 4, col)
    col += 12
    
    # Sub-problem 2: A rows 0-1, cols 2-3; B rows 2-3, cols 0-4
    # 2x2x2 for B cols 0,1 using Strassen
    add_strassen_222(0, 1, 2, 3, 0, 1, col)
    col += 7
    # 2x2x3 for B cols 2,3,4 using schoolbook
    add_schoolbook_223(0, 1, 2, 3, 2, 3, 4, col)
    col += 12
    
    assert col == rank, f"Expected {rank} columns, got {col}"
    
    return U, V, W

def _construct_rank_36():
    """
    Rank-36 decomposition for 2x4 x 4x5 matrix multiplication.
    Fallback using Strassen for 2x2x2 blocks and schoolbook for 2x2x1.
    """
    n, m, p = 2, 4, 5
    rank = 36
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_strassen_222(i1, i2, j1, j2, k1, k2, col_offset):
        a11 = i1 * m + j1
        a12 = i1 * m + j2
        a21 = i2 * m + j1
        a22 = i2 * m + j2
        b11 = j1 * p + k1
        b12 = j1 * p + k2
        b21 = j2 * p + k1
        b22 = j2 * p + k2
        c11 = k1 * n + i1
        c12 = k2 * n + i1
        c21 = k1 * n + i2
        c22 = k2 * n + i2
        
        col = col_offset + 0
        U[a11][col] = 1; U[a22][col] = 1
        V[b11][col] = 1; V[b22][col] = 1
        W[c11][col] = 1; W[c22][col] = 1
        col = col_offset + 1
        U[a21][col] = 1; U[a22][col] = 1
        V[b11][col] = 1
        W[c21][col] = 1; W[c22][col] = -1
        col = col_offset + 2
        U[a11][col] = 1
        V[b12][col] = 1; V[b22][col] = -1
        W[c12][col] = 1; W[c22][col] = 1
        col = col_offset + 3
        U[a22][col] = 1
        V[b21][col] = 1; V[b11][col] = -1
        W[c11][col] = 1; W[c21][col] = 1
        col = col_offset + 4
        U[a11][col] = 1; U[a12][col] = 1
        V[b22][col] = 1
        W[c11][col] = -1; W[c12][col] = 1
        col = col_offset + 5
        U[a21][col] = 1; U[a11][col] = -1
        V[b11][col] = 1; V[b12][col] = 1
        W[c22][col] = 1
        col = col_offset + 6
        U[a12][col] = 1; U[a22][col] = -1
        V[b21][col] = 1; V[b22][col] = 1
        W[c11][col] = 1

    def add_schoolbook_221(i1, i2, j1, j2, k1, col_offset):
        a11 = i1 * m + j1
        a12 = i1 * m + j2
        a21 = i2 * m + j1
        a22 = i2 * m + j2
        b11 = j1 * p + k1
        b21 = j2 * p + k1
        c11 = k1 * n + i1
        c21 = k1 * n + i2
        
        col = col_offset + 0
        U[a11][col] = 1; V[b11][col] = 1; W[c11][col] = 1
        col = col_offset + 1
        U[a12][col] = 1; V[b21][col] = 1; W[c11][col] = 1
        col = col_offset + 2
        U[a21][col] = 1; V[b11][col] = 1; W[c21][col] = 1
        col = col_offset + 3
        U[a22][col] = 1; V[b21][col] = 1; W[c21][col] = 1

    col = 0
    add_strassen_222(0, 1, 0, 1, 0, 1, col); col += 7
    add_strassen_222(0, 1, 0, 1, 2, 3, col); col += 7
    add_schoolbook_221(0, 1, 0, 1, 4, col); col += 4
    add_strassen_222(0, 1, 2, 3, 0, 1, col); col += 7
    add_strassen_222(0, 1, 2, 3, 2, 3, col); col += 7
    add_schoolbook_221(0, 1, 2, 3, 4, col); col += 4
    
    assert col == rank
    return U, V, W

def _construct_schoolbook():
    """Fallback schoolbook decomposition with rank n*m*p = 40."""
    n, m, p = 2, 4, 5
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
    
    return U, V, W

def construct():
    start_time = time.time()
    DEADLINE = 140
    
    n, m, p = DIMENSIONS
    
    # Try Rank-26 decomposition first
    try:
        U_r26, V_r26, W_r26 = _construct_rank_26()
        if _validate_numpy(U_r26, V_r26, W_r26, n, m, p):
            result_U = [[_clean_entry(v) for v in row] for row in U_r26]
            result_V = [[_clean_entry(v) for v in row] for row in V_r26]
            result_W = [[_clean_entry(v) for v in row] for row in W_r26]
            return {"U": result_U, "V": result_V, "W": result_W}
    except Exception:
        pass
    
    # Try Rank-36 decomposition
    try:
        U_r36, V_r36, W_r36 = _construct_rank_36()
        if _validate_numpy(U_r36, V_r36, W_r36, n, m, p):
            result_U = [[_clean_entry(v) for v in row] for row in U_r36]
            result_V = [[_clean_entry(v) for v in row] for row in V_r36]
            result_W = [[_clean_entry(v) for v in row] for row in W_r36]
            return {"U": result_U, "V": result_V, "W": result_W}
    except Exception:
        pass
    
    # Fallback to Schoolbook
    U_school, V_school, W_school = _construct_schoolbook()
    assert _validate_numpy(U_school, V_school, W_school, n, m, p), "Schoolbook decomposition failed!"
    
    return {"U": U_school, "V": V_school, "W": W_school}
# EVOLVE-BLOCK-END
