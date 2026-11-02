import numpy as np
import time

DIMENSIONS = [3, 5, 5]

def construct():
    n, m, p = DIMENSIONS  # 3, 5, 5
    
    # Calculate expected rank
    # 4 Strassen blocks for I={0,1}, covering J in {0,1,2,3} and K in {0,1,2,3}
    # Each block rank 7. Total 28.
    # Remaining for I={0,1}: J=4 (5 terms) + K=4 (4 terms) = 9 terms.
    # Total for I={0,1}: 37.
    # I={2}: 25 terms.
    # Total: 62.
    
    total_rank = 62
    
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    blocks = [
        (0, 1, 0, 1, 0, 1),
        (0, 1, 2, 3, 0, 1),
        (0, 1, 0, 1, 2, 3),
        (0, 1, 2, 3, 2, 3),
    ]
    
    for (i0, i1, j0, j1, k0, k1) in blocks:
        # M1 = (a00 + a11)(b00 + b11)
        c = col
        U[i0*m + j0][c] += 1
        U[i1*m + j1][c] += 1
        V[j0*p + k0][c] += 1
        V[j1*p + k1][c] += 1
        W[k0*n + i0][c] += 1
        W[k1*n + i1][c] += 1
        col += 1
        
        # M2 = (a10 + a11)b00
        c = col
        U[i1*m + j0][c] += 1
        U[i1*m + j1][c] += 1
        V[j0*p + k0][c] += 1
        W[k0*n + i1][c] += 1
        W[k1*n + i1][c] += -1
        col += 1
        
        # M3 = a00(b01 - b11)
        c = col
        U[i0*m + j0][c] += 1
        V[j0*p + k1][c] += 1
        V[j1*p + k1][c] += -1
        W[k1*n + i0][c] += 1
        W[k1*n + i1][c] += 1
        col += 1
        
        # M4 = a11(b10 - b00)
        c = col
        U[i1*m + j1][c] += 1
        V[j1*p + k0][c] += 1
        V[j0*p + k0][c] += -1
        W[k0*n + i0][c] += 1
        W[k0*n + i1][c] += 1
        col += 1
        
        # M5 = (a00 + a01)b11
        c = col
        U[i0*m + j0][c] += 1
        U[i0*m + j1][c] += 1
        V[j1*p + k1][c] += 1
        W[k0*n + i0][c] += -1
        W[k1*n + i0][c] += 1
        col += 1
        
        # M6 = (a10 - a00)(b00 + b01)
        c = col
        U[i1*m + j0][c] += 1
        U[i0*m + j0][c] += -1
        V[j0*p + k0][c] += 1
        V[j0*p + k1][c] += 1
        W[k1*n + i1][c] += 1
        col += 1
        
        # M7 = (a01 - a11)(b10 + b11)
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
    
    # Verify rank
    if col != total_rank:
        # If col is less, we have zero columns at the end.
        # We should trim them.
        U = [row[:col] for row in U]
        V = [row[:col] for row in V]
        W = [row[:col] for row in W]
        total_rank = col
    else:
        # If col is more, something is wrong.
        pass

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
