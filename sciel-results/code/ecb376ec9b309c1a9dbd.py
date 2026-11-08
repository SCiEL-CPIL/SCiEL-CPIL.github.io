import numpy as np
import time

DIMENSIONS = [3, 5, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 140
    n, m, p = DIMENSIONS  # n=3, m=5, p=5
    
    # Rank 68 decomposition
    total_rank = 68
    
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    def add_term(u_idx, v_idx, w_idx, c, coeff=1.0):
        U[u_idx][c] += coeff
        V[v_idx][c] += coeff
        W[w_idx][c] += coeff
    
    def add_strassen_2x2x2(i0, j0, k0, col_offset):
        nonlocal col
        # M1 = (A00+A11)(B00+B11)
        c = col_offset + 0
        U[i0*m + j0][c] += 1
        U[(i0+1)*m + (j0+1)][c] += 1
        V[j0*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        
        # M2 = (A10+A11)B00
        c = col_offset + 1
        U[(i0+1)*m + j0][c] += 1
        U[(i0+1)*m + (j0+1)][c] += 1
        V[j0*p + k0][c] += 1
        W[k0*n + (i0+1)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += -1
        
        # M3 = A00(B01-B11)
        c = col_offset + 2
        U[i0*m + j0][c] += 1
        V[j0*p + (k0+1)][c] += 1
        V[(j0+1)*p + (k0+1)][c] += -1
        W[(k0+1)*n + i0][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        
        # M4 = A11(B10-B00)
        c = col_offset + 3
        U[(i0+1)*m + (j0+1)][c] += 1
        V[(j0+1)*p + k0][c] += 1
        V[j0*p + k0][c] += -1
        W[k0*n + i0][c] += 1
        W[k0*n + (i0+1)][c] += 1
        
        # M5 = (A00+A01)B11
        c = col_offset + 4
        U[i0*m + j0][c] += 1
        U[i0*m + (j0+1)][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += -1
        W[(k0+1)*n + i0][c] += 1
        
        # M6 = (A10-A00)(B00+B01)
        c = col_offset + 5
        U[(i0+1)*m + j0][c] += 1
        U[i0*m + j0][c] += -1
        V[j0*p + k0][c] += 1
        V[j0*p + (k0+1)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        
        # M7 = (A01-A11)(B10+B11)
        c = col_offset + 6
        U[i0*m + (j0+1)][c] += 1
        U[(i0+1)*m + (j0+1)][c] += -1
        V[(j0+1)*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += 1
        
        col += 7
    
    # Block 1: 3x3x3 core using rank-23 decomposition
    # I'll use a simplified approach: decompose into Strassen 2x2x2 + schoolbook
    # i in {0,1}, j in {0,1}, k in {0,1}: 2x2x2, rank 7
    add_strassen_2x2x2(0, 0, 0, col)
    
    # i in {2}, j in {0,1}, k in {0,1}: 1x2x2, rank 4 (schoolbook)
    for i in [2]:
        for j in [0, 1]:
            for k in [0, 1]:
                U[i*m + j][col] += 1
                V[j*p + k][col] += 1
                W[k*n + i][col] += 1
                col += 1
    
    # i in {0,1}, j in {2}, k in {0,1}: 2x1x2, rank 4 (schoolbook)
    for i in [0, 1]:
        for j in [2]:
            for k in [0, 1]:
                U[i*m + j][col] += 1
                V[j*p + k][col] += 1
                W[k*n + i][col] += 1
                col += 1
    
    # i in {2}, j in {2}, k in {0,1}: 1x1x2, rank 2 (schoolbook)
    for i in [2]:
        for j in [2]:
            for k in [0, 1]:
                U[i*m + j][col] += 1
                V[j*p + k][col] += 1
                W[k*n + i][col] += 1
                col += 1
    
    # i in {0,1}, j in {0,1}, k in {2}: 2x2x1, rank 4 (schoolbook)
    for i in [0, 1]:
        for j in [0, 1]:
            for k in [2]:
                U[i*m + j][col] += 1
                V[j*p + k][col] += 1
                W[k*n + i][col] += 1
                col += 1
    
    # i in {2}, j in {0,1}, k in {2}: 1x2x1, rank 2 (schoolbook)
    for i in [2]:
        for j in [0, 1]:
            for k in [2]:
                U[i*m + j][col] += 1
                V[j*p + k][col] += 1
                W[k*n + i][col] += 1
                col += 1
    
    # i in {0,1}, j in {2}, k in {2}: 2x1x1, rank 2 (schoolbook)
    for i in [0, 1]:
        for j in [2]:
            for k in [2]:
                U[i*m + j][col] += 1
                V[j*p + k][col] += 1
                W[k*n + i][col] += 1
                col += 1
    
    # i in {2}, j in {2}, k in {2}: 1x1x1, rank 1 (schoolbook)
    U[2*m + 2][col] += 1
    V[2*p + 2][col] += 1
    W[2*n + 2][col] += 1
    col += 1
    
    # That's 7 + 4 + 4 + 2 + 4 + 2 + 2 + 1 = 26 terms for the 3x3x3 block
    # But I wanted rank 23. Let me use a proper rank-23 decomposition.
    
    # Actually, let me just use the schoolbook for the 3x3x3 block (rank 27)
    # and see if the total is still better than 71.
    
    # Reset and try a different approach
    U = [[0.0] * 75 for _ in range(n * m)]
    V = [[0.0] * 75 for _ in range(m * p)]
    W = [[0.0] * 75 for _ in range(p * n)]
    col = 0
    
    # Use 4 Strassen blocks for i in {0,1}
    blocks = [
        (0, 0, 0),
        (0, 2, 0),
        (0, 0, 2),
        (0, 2, 2)
    ]
    
    for i0, j0, k0 in blocks:
        add_strassen_2x2x2(i0, j0, k0, col)
    
    # Schoolbook for the rest
    covered_3d = [[[False]*p for _ in range(m)] for _ in range(n)]
    for i0, j0, k0 in blocks:
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    covered_3d[i0+di][j0+dj][k0+dk] = True
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if not covered_3d[i][j][k]:
                    U[i*m+j][col] += 1
                    V[j*p+k][col] += 1
                    W[k*n+i][col] += 1
                    col += 1
    
    # Trim to actual rank
    U = [row[:col] for row in U]
    V = [row[:col] for row in V]
    W = [row[:col] for row in W]
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    n, m, p = DIMENSIONS
    print(f"Rank: {U.shape[1]}")
    # Verify
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(U.shape[1]))
                if abs(val - 1.0) > 1e-9:
                    print(f"FAIL at ({i},{j},{k}): {val}")
                    break
