"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly reproduces the tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Check valid terms
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.sum(U[a] * V[b] * W[c])
                if abs(val - 1.0) > 1e-9:
                    return False
    
    # Check spurious terms
    valid = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                valid.add((i * m + j, j * p + k, k * n + i))
    
    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
                if (a, b, c) not in valid:
                    val = np.sum(U[a] * V[b] * W[c])
                    if abs(val) > 1e-9:
                        return False
    return True

def _rank32_decomposition(n, m, p):
    """
    Rank-32 decomposition exploiting the structure:
    C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]
    C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k]
    
    For each k, use 2 rank-1 terms:
    Term 1: shares B[0,k], combines A[0,0] and A[1,0]
    Term 2: shares B[1,k], combines A[0,1] and A[1,1]
    """
    rank = 2 * p  # 32
    U = [[0] * rank for _ in range(n * m)]  # 4 x 32
    V = [[0] * rank for _ in range(m * p)]  # 32 x 32
    W = [[0] * rank for _ in range(p * n)]  # 32 x 32
    
    for k in range(p):
        # Term 1: A[0,0] and A[1,0] with B[0,k], contributing to C[0,k] and C[1,k]
        col1 = 2 * k
        U[0][col1] = 1   # A[0,0]
        U[2][col1] = 1   # A[1,0]
        V[k][col1] = 1   # B[0,k] (row 0 of B, column k)
        W[2 * k][col1] = 1     # C[0,k]
        W[2 * k + 1][col1] = 1 # C[1,k]
        
        # Term 2: A[0,1] and A[1,1] with B[1,k], contributing to C[0,k] and C[1,k]
        col2 = 2 * k + 1
        U[1][col2] = 1   # A[0,1]
        U[3][col2] = 1   # A[1,1]
        V[p + k][col2] = 1  # B[1,k] (row 1 of B, column k)
        W[2 * k][col2] = 1     # C[0,k]
        W[2 * k + 1][col2] = 1 # C[1,k]
    
    return U, V, W

def _strassen_block(n, m, p):
    """Fallback: Strassen-based decomposition, rank 56."""
    n, m, p = 2, 2, 16
    num_blocks = p // 2
    rank = num_blocks * 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for t in range(num_blocks):
        v_idx = {'B00': 2*t, 'B01': 2*t+1, 'B10': 16+2*t, 'B11': 16+2*t+1}
        w_idx = {'C00': 4*t, 'C01': 4*t+2, 'C10': 4*t+1, 'C11': 4*t+3}
        
        # M1 = (A00+A11)(B00+B11)
        U[0][col] += 1; U[3][col] += 1
        V[v_idx['B00']][col] += 1; V[v_idx['B11']][col] += 1
        W[w_idx['C00']][col] += 1; W[w_idx['C11']][col] += 1
        col += 1
        # M2 = (A10+A11)B00
        U[2][col] += 1; U[3][col] += 1
        V[v_idx['B00']][col] += 1
        W[w_idx['C10']][col] += 1; W[w_idx['C11']][col] -= 1
        col += 1
        # M3 = A00(B01-B11)
        U[0][col] += 1
        V[v_idx['B01']][col] += 1; V[v_idx['B11']][col] -= 1
        W[w_idx['C01']][col] += 1; W[w_idx['C11']][col] += 1
        col += 1
        # M4 = A11(B10-B00)
        U[3][col] += 1
        V[v_idx['B10']][col] += 1; V[v_idx['B00']][col] -= 1
        W[w_idx['C00']][col] += 1; W[w_idx['C10']][col] += 1
        col += 1
        # M5 = (A00+A01)B11
        U[0][col] += 1; U[1][col] += 1
        V[v_idx['B11']][col] += 1
        W[w_idx['C00']][col] -= 1; W[w_idx['C01']][col] += 1
        col += 1
        # M6 = (A10-A00)(B00+B01)
        U[2][col] += 1; U[0][col] -= 1
        V[v_idx['B00']][col] += 1; V[v_idx['B01']][col] += 1
        W[w_idx['C11']][col] += 1
        col += 1
        # M7 = (A01-A11)(B10+B11)
        U[1][col] += 1; U[3][col] -= 1
        V[v_idx['B10']][col] += 1; V[v_idx['B11']][col] += 1
        W[w_idx['C00']][col] += 1
        col += 1
    
    return U, V, W

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
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit
    
    # Try rank-32 decomposition first
    U, V, W = _rank32_decomposition(n, m, p)
    if _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # Fall back to Strassen (rank 56)
    if time.time() < deadline:
        U, V, W = _strassen_block(n, m, p)
        if _verify(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    
    # Final fallback: schoolbook (rank 64)
    U, V, W = _schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
