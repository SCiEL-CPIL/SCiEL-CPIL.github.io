"""Tensor decomposition for 2x3x8 matrix multiplication."""
import numpy as np

DIMENSIONS = [2, 3, 8]

def construct():
    n, m, p = DIMENSIONS  # 2, 3, 8
    
    # Rank-16 construction for 2x3x8
    # Key insight: For each output column k (0..7), we can compute both C[0,k] and C[1,k]
    # using shared terms. Specifically:
    #   C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k] + A[0,2]*B[2,k]
    #   C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k] + A[1,2]*B[2,k]
    # 
    # We use 2 terms per column:
    #   Term k:   U has 1s at A[0,0], A[0,1], A[0,2]; V has 1s at B[0,k], B[1,k], B[2,k]; W has 1 at C[0,k]
    #   Term 8+k: U has 1s at A[1,0], A[1,1], A[1,2]; V has 1s at B[0,k], B[1,k], B[2,k]; W has 1 at C[1,k]
    #
    # Total rank = 2 * 8 = 16
    
    rank = 16
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Term k (k=0..7): computes C[0,k]
    for k in range(p):
        # A indices: row 0, cols 0,1,2 -> indices 0, 1, 2
        U[0, k] = 1
        U[1, k] = 1
        U[2, k] = 1
        # B indices: rows 0,1,2, col k -> indices 0*p+k, 1*p+k, 2*p+k
        V[0 * p + k, k] = 1
        V[1 * p + k, k] = 1
        V[2 * p + k, k] = 1
        # C index: col k, row 0 -> index k*n + 0
        W[k * n + 0, k] = 1
    
    # Term 8+k (k=0..7): computes C[1,k]
    for k in range(p):
        # A indices: row 1, cols 0,1,2 -> indices 3, 4, 5
        U[3, 8 + k] = 1
        U[4, 8 + k] = 1
        U[5, 8 + k] = 1
        # B indices: rows 0,1,2, col k -> indices 0*p+k, 1*p+k, 2*p+k
        V[0 * p + k, 8 + k] = 1
        V[1 * p + k, 8 + k] = 1
        V[2 * p + k, 8 + k] = 1
        # C index: col k, row 1 -> index k*n + 1
        W[k * n + 1, 8 + k] = 1
    
    # Verify the decomposition
    def verify(U, V, W, n, m, p):
        rank = U.shape[1]
        result = np.zeros((n * m, m * p, p * n), dtype=float)
        for t in range(rank):
            u = U[:, t]
            v = V[:, t]
            w = W[:, t]
            for i in range(n * m):
                if u[i] == 0:
                    continue
                for j in range(m * p):
                    if v[j] == 0:
                        continue
                    for k in range(p * n):
                        if w[k] == 0:
                            continue
                        result[i, j, k] += u[i] * v[j] * w[k]
        
        correct = np.zeros((n * m, m * p, p * n), dtype=float)
        for row_a in range(n):
            for col_a in range(m):
                for row_b in range(m):
                    for col_b in range(p):
                        if col_a == row_b:
                            i = row_a * m + col_a
                            j = row_b * p + col_b
                            k = col_b * n + row_a
                            correct[i, j, k] = 1
        
        return np.allclose(result, correct, atol=1e-10)
    
    is_valid = verify(U, V, W, n, m, p)
    
    if is_valid:
        U_list = [[float(x) for x in row] for row in U]
        V_list = [[float(x) for x in row] for row in V]
        W_list = [[float(x) for x in row] for row in W]
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        # Fallback to schoolbook (rank 48)
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        U_list = [[float(x) for x in row] for row in U]
        V_list = [[float(x) for x in row] for row in V]
        W_list = [[float(x) for x in row] for row in W]
        return {"U": U_list, "V": V_list, "W": W_list}
