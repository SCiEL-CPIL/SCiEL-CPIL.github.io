import numpy as np
import time

DIMENSIONS = [3, 5, 5]

def construct():
    n, m, p = DIMENSIONS  # 3, 5, 5
    
    # We use 4 disjoint 2x2x2 Strassen blocks in the 3x5x5 multiplication
    # Blocks: (i in {0,1}, j in {0,1}, k in {0,1}),
    #         (i in {0,1}, j in {2,3}, k in {0,1}),
    #         (i in {0,1}, j in {0,1}, k in {2,3}),
    #         (i in {0,1}, j in {2,3}, k in {2,3})
    # Each block uses 7 rank terms instead of 8, saving 1 per block.
    # Total: 4 * 7 = 28 for the Strassen blocks
    # Remaining terms: 75 - 4*8 = 75 - 32 = 43
    # Total rank: 28 + 43 = 71
    
    total_rank = 71
    
    U = [[0.0] * total_rank for _ in range(n * m)]  # shape (15, 71)
    V = [[0.0] * total_rank for _ in range(m * p)]  # shape (25, 71)
    W = [[0.0] * total_rank for _ in range(p * n)]  # shape (15, 71)
    
    col = 0
    
    blocks = [
        (0, 1, 0, 1, 0, 1),
        (0, 1, 2, 3, 0, 1),
        (0, 1, 0, 1, 2, 3),
        (0, 1, 2, 3, 2, 3),
    ]
    
    for (i0, i1, j0, j1, k0, k1) in blocks:
        # M1 = (a_i0,j0 + a_i1,j1)(b_j0,k0 + b_j1,k1)
        c = col
        U[i0*m + j0][c] += 1
        U[i1*m + j1][c] += 1
        V[j0*p + k0][c] += 1
        V[j1*p + k1][c] += 1
        W[k0*n + i0][c] += 1
        W[k1*n + i1][c] += 1
        col += 1
        
        # M2 = (a_i1,j0 + a_i1,j1)b_j0,k0
        c = col
        U[i1*m + j0][c] += 1
        U[i1*m + j1][c] += 1
        V[j0*p + k0][c] += 1
        W[k0*n + i1][c] += 1
        W[k1*n + i1][c] += -1
        col += 1
        
        # M3 = a_i0,j0(b_j0,k1 - b_j1,k1)
        c = col
        U[i0*m + j0][c] += 1
        V[j0*p + k1][c] += 1
        V[j1*p + k1][c] += -1
        W[k1*n + i0][c] += 1
        W[k1*n + i1][c] += 1
        col += 1
        
        # M4 = a_i1,j1(b_j1,k0 - b_j0,k0)
        c = col
        U[i1*m + j1][c] += 1
        V[j1*p + k0][c] += 1
        V[j0*p + k0][c] += -1
        W[k0*n + i0][c] += 1
        W[k0*n + i1][c] += 1
        col += 1
        
        # M5 = (a_i0,j0 + a_i0,j1)b_j1,k1
        c = col
        U[i0*m + j0][c] += 1
        U[i0*m + j1][c] += 1
        V[j1*p + k1][c] += 1
        W[k0*n + i0][c] += -1
        W[k1*n + i0][c] += 1
        col += 1
        
        # M6 = (a_i1,j0 - a_i0,j0)(b_j0,k0 + b_j0,k1)
        c = col
        U[i1*m + j0][c] += 1
        U[i0*m + j0][c] += -1
        V[j0*p + k0][c] += 1
        V[j0*p + k1][c] += 1
        W[k1*n + i1][c] += 1
        col += 1
        
        # M7 = (a_i0,j1 - a_i1,j1)(b_j1,k0 + b_j1,k1)
        c = col
        U[i0*m + j1][c] += 1
        U[i1*m + j1][c] += -1
        V[j1*p + k0][c] += 1
        V[j1*p + k1][c] += 1
        W[k0*n + i0][c] += 1
        col += 1
    
    # Schoolbook for remaining terms
    for i in range(n):
        for j in range(m):
            for k in range(p):
                in_block = False
                for (i0, i1, j0, j1, k0, k1) in blocks:
                    if i0 <= i <= i1 and j0 <= j <= j1 and k0 <= k <= k1:
                        in_block = True
                        break
                if in_block:
                    continue
                U[i*m + j][col] += 1
                V[j*p + k][col] += 1
                W[k*n + i][col] += 1
                col += 1
    
    assert col == total_rank, f"Expected {total_rank} columns, got {col}"
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    n, m, p = DIMENSIONS
    print(f"Rank: {U.shape[1]}")
    
    T_reconstructed = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_reconstructed[i,j,k] = 0.0
                for c in range(U.shape[1]):
                    T_reconstructed[i,j,k] += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    T_expected = np.ones((n, m, p))
    max_err = np.max(np.abs(T_reconstructed - T_expected))
    print(f"Max error: {max_err}")
    
    if max_err < 1e-10:
        print("Decomposition is correct!")
    else:
        print("Decomposition has errors!")
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if abs(T_reconstructed[i,j,k] - 1.0) > 1e-10:
                        print(f"  T[{i},{j},{k}] = {T_reconstructed[i,j,k]} (expected 1.0)")
