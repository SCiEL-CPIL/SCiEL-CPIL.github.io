"""Matrix multiplication 2x3x8 - rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 3, 8]

def _verify(U, V, W, tol=1e-9):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Check target entries
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = 0.0
                for r in range(rank):
                    val += U_arr[a, r] * V_arr[b, r] * W_arr[c, r]
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    
    # Check non-target entries are zero
    target_set = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target_set.add((i*m+j, j*p+k, k*n+i))
    
    non_target_err = 0.0
    for a in range(n*m):
        for b in range(m*p):
            for c in range(p*n):
                if (a, b, c) not in target_set:
                    val = 0.0
                    for r in range(rank):
                        val += U_arr[a, r] * V_arr[b, r] * W_arr[c, r]
                    err = abs(val)
                    if err > non_target_err:
                        non_target_err = err
    
    return max_err <= tol and non_target_err <= tol

def _build_rank44():
    """
    Build explicit rank-44 decomposition for 2x3 * 3x8.
    
    Split B (3x8) into 4 blocks of 3x2.
    For each block, decompose 2x3 * 3x2 as:
      - Strassen rank-7 for A[0:2,0:2] * B[0:2,0:2] (the 2x2 core)
      - 4 direct rank-1 terms for A[0:2,2] * B[2,0:2] (the extra column/row)
    Total: 4 * (7 + 4) = 44
    """
    n, m, p = DIMENSIONS  # 2, 3, 8
    rank = 44
    
    U = [[0] * rank for _ in range(n * m)]  # (6, 44)
    V = [[0] * rank for _ in range(m * p)]  # (24, 44)
    W = [[0] * rank for _ in range(p * n)]  # (16, 44)
    
    col = 0
    
    for b in range(4):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # --- Strassen part: A_core (2x2) * B_core (2x2) ---
        # A_core uses A[i, j] for i in {0,1}, j in {0,1}
        # B_core uses B[j, k] for j in {0,1}, k in {k0, k1}
        # C_core[i, k] = sum_{j=0}^{1} A[i,j] * B[j,k]
        
        # M1 = (A[0,0] + A[1,1]) * (B[0,k0] + B[1,k1])
        U[0 * m + 0][col] = 1   # A[0,0]
        U[1 * m + 1][col] = 1   # A[1,1]
        V[0 * p + k0][col] = 1  # B[0,k0]
        V[1 * p + k1][col] = 1  # B[1,k1]
        W[k0 * n + 0][col] = 1  # C[0,k0] += ...
        W[k1 * n + 1][col] = 1  # C[1,k1] += ...
        col += 1
        
        # M2 = (A[1,0] + A[1,1]) * B[0,k0]
        U[1 * m + 0][col] = 1   # A[1,0]
        U[1 * m + 1][col] = 1   # A[1,1]
        V[0 * p + k0][col] = 1  # B[0,k0]
        W[k0 * n + 1][col] = 1  # C[1,k0] += ...
        W[k1 * n + 1][col] = -1 # C[1,k1] -= ...
        col += 1
        
        # M3 = A[0,0] * (B[0,k1] - B[1,k1])
        U[0 * m + 0][col] = 1   # A[0,0]
        V[0 * p + k1][col] = 1  # B[0,k1]
        V[1 * p + k1][col] = -1 # B[1,k1]
        W[k1 * n + 0][col] = 1  # C[0,k1] += ...
        W[k1 * n + 1][col] = 1  # C[1,k1] += ...
        col += 1
        
        # M4 = A[1,1] * (B[1,k0] - B[0,k0])
        U[1 * m + 1][col] = 1   # A[1,1]
        V[1 * p + k0][col] = 1  # B[1,k0]
        V[0 * p + k0][col] = -1 # B[0,k0]
        W[k0 * n + 0][col] = 1  # C[0,k0] += ...
        W[k0 * n + 1][col] = 1  # C[1,k0] += ...
        col += 1
        
        # M5 = (A[0,0] + A[0,1]) * B[1,k1]
        U[0 * m + 0][col] = 1   # A[0,0]
        U[0 * m + 1][col] = 1   # A[0,1]
        V[1 * p + k1][col] = 1  # B[1,k1]
        W[k0 * n + 0][col] = -1 # C[0,k0] -= ...
        W[k1 * n + 0][col] = 1  # C[0,k1] += ...
        col += 1
        
        # M6 = (A[1,0] - A[0,0]) * (B[0,k0] + B[0,k1])
        U[1 * m + 0][col] = 1   # A[1,0]
        U[0 * m + 0][col] = -1  # A[0,0]
        V[0 * p + k0][col] = 1  # B[0,k0]
        V[0 * p + k1][col] = 1  # B[0,k1]
        W[k1 * n + 1][col] = 1  # C[1,k1] += ...
        col += 1
        
        # M7 = (A[0,1] - A[1,1]) * (B[1,k0] + B[1,k1])
        U[0 * m + 1][col] = 1   # A[0,1]
        U[1 * m + 1][col] = -1  # A[1,1]
        V[1 * p + k0][col] = 1  # B[1,k0]
        V[1 * p + k1][col] = 1  # B[1,k1]
        W[k0 * n + 0][col] = 1  # C[0,k0] += ...
        col += 1
        
        # --- Direct terms for A[0:2,2] * B[2,k0:k1] ---
        # A[0,2]*B[2,k0] -> C[0,k0]
        U[0 * m + 2][col] = 1
        V[2 * p + k0][col] = 1
        W[k0 * n + 0][col] = 1
        col += 1
        
        # A[0,2]*B[2,k1] -> C[0,k1]
        U[0 * m + 2][col] = 1
        V[2 * p + k1][col] = 1
        W[k1 * n + 0][col] = 1
        col += 1
        
        # A[1,2]*B[2,k0] -> C[1,k0]
        U[1 * m + 2][col] = 1
        V[2 * p + k0][col] = 1
        W[k0 * n + 1][col] = 1
        col += 1
        
        # A[1,2]*B[2,k1] -> C[1,k1]
        U[1 * m + 2][col] = 1
        V[2 * p + k1][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1
    
    return U, V, W

def construct():
    """
    Decompose 2x3x8 matrix multiplication tensor.
    Uses rank-44 Strassen-based block decomposition.
    """
    start_time = time.time()
    n, m, p = DIMENSIONS  # 2, 3, 8
    
    # Build rank-44 decomposition
    U, V, W = _build_rank44()
    
    # Verify quickly
    if _verify(U, V, W):
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook rank 48
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
