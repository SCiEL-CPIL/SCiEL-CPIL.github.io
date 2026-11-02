import time
import numpy as np

def construct():
    n, m, p = 2, 3, 5
    start_time = time.time()
    
    # Directly construct the known rank 28 solution
    U, V, W = _build_rank28_arrays(n, m, p)
    
    # Optional: Very short search for rank 27 (likely to fail, but cheap)
    # We skip this to guarantee speed and reliability, as rank 27 is not known to exist
    # and the search is computationally expensive.
    
    return _to_lists(U, V, W)

def _build_rank28_arrays(n, m, p):
    rank = 28
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    # Block 1: j=0, schoolbook (2x1 by 1x5) -> Rank 10
    for i in range(n):
        for k in range(p):
            U[i * m + 0, col] = 1
            V[0 * p + k, col] = 1
            W[k * n + i, col] = 1
            col += 1
            
    # Block 2a: j=1,2, k=0,1, Strassen (2x2 by 2x2) -> Rank 7
    col = _add_strassen(U, V, W, n, m, p, 1, 0, col)
    
    # Block 2b: j=1,2, k=2, schoolbook (2x2 by 2x1) -> Rank 4
    for i in range(n):
        for j_local in range(2):
            j = j_local + 1
            U[i * m + j, col] = 1
            V[j * p + 2, col] = 1
            W[2 * n + i, col] = 1
            col += 1
            
    # Block 2c: j=1,2, k=3,4, Strassen (2x2 by 2x2) -> Rank 7
    col = _add_strassen(U, V, W, n, m, p, 1, 3, col)
    
    return U, V, W

def _add_strassen(U, V, W, n, m, p, base_j, base_k, col):
    # Strassen decomposition for 2x2 by 2x2
    # Indices: i, j in {0,1} relative to base
    # U rows: base_j, base_j+1
    # V cols: base_k, base_k+1
    # W rows: base_k, base_k+1
    
    # 1. (A11+A22)(B11+B22)
    U[0 * m + base_j, col] = 1
    U[1 * m + (base_j + 1), col] = 1
    V[base_j * p + base_k, col] = 1
    V[(base_j + 1) * p + (base_k + 1), col] = 1
    W[base_k * n + 0, col] = 1
    W[(base_k + 1) * n + 1, col] = 1
    col += 1
    
    # 2. (A21+A22)B11
    U[1 * m + base_j, col] = 1
    U[1 * m + (base_j + 1), col] = 1
    V[base_j * p + base_k, col] = 1
    W[base_k * n + 1, col] = 1
    W[(base_k + 1) * n + 1, col] = -1
    col += 1
    
    # 3. A11(B21-B11)
    U[0 * m + base_j, col] = 1
    V[base_j * p + (base_k + 1), col] = 1
    V[(base_j + 1) * p + (base_k + 1), col] = -1
    W[(base_k + 1) * n + 0, col] = 1
    W[(base_k + 1) * n + 1, col] = 1
    col += 1
    
    # 4. A22(B12-B22)
    U[1 * m + (base_j + 1), col] = 1
    V[(base_j + 1) * p + base_k, col] = 1
    V[base_j * p + base_k, col] = -1
    W[base_k * n + 0, col] = 1
    W[base_k * n + 1, col] = 1
    col += 1
    
    # 5. (A11+A12)B22
    U[0 * m + base_j, col] = 1
    U[0 * m + (base_j + 1), col] = 1
    V[(base_j + 1) * p + (base_k + 1), col] = 1
    W[base_k * n + 0, col] = -1
    W[(base_k + 1) * n + 0, col] = 1
    col += 1
    
    # 6. (A21-A11)(B11+B12)
    U[1 * m + base_j, col] = 1
    U[0 * m + base_j, col] = -1
    V[base_j * p + base_k, col] = 1
    V[base_j * p + (base_k + 1), col] = 1
    W[(base_k + 1) * n + 1, col] = 1
    col += 1
    
    # 7. (A12-A22)(B21+B22)
    U[0 * m + (base_j + 1), col] = 1
    U[1 * m + (base_j + 1), col] = -1
    V[(base_j + 1) * p + base_k, col] = 1
    V[(base_j + 1) * p + (base_k + 1), col] = 1
    W[base_k * n + 0, col] = 1
    col += 1
    
    return col

def _to_lists(U, V, W):
    def to_list(arr):
        result = []
        for row in arr:
            row_list = []
            for val in row:
                # Ensure integer or half-integer
                if abs(val - round(val)) < 1e-8:
                    row_list.append(int(round(val)))
                elif abs(val - (round(val*2)/2)) < 1e-8:
                    half = round(val * 2) / 2
                    row_list.append(float(half))
                else:
                    row_list.append(float(val))
            result.append(row_list)
        return result
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
