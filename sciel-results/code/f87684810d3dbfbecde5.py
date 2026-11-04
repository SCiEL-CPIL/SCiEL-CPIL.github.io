import numpy as np
import time

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140

    def check_decomposition(U, V, W, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(val - 1.0) > 1e-6:
                        return False
        return True

    # Build rank-17 decomposition
    rank = 17
    U = [[0.0]*rank for _ in range(n*m)]
    V = [[0.0]*rank for _ in range(m*p)]
    W = [[0.0]*rank for _ in range(p*n)]

    c = 0

    # --- Part 1: Strassen 2x2x2 for j in {0,1}, k in {0,1} ---
    # Indices:
    # A: i in {0,1}, j in {0,1} -> U[i*3+j]
    # B: j in {0,1}, k in {0,1} -> V[j*3+k]
    # C: k in {0,1}, i in {0,1} -> W[k*2+i]
    
    # M1 = (A[1,1]-A[0,0]) * (B[0,0]+B[1,1])
    # A[1,1]=U[4], A[0,0]=U[0]
    # B[0,0]=V[0], B[1,1]=V[4]
    # C[0,0]=W[0], C[1,1]=W[3]
    U[4][c] += 1; U[0][c] -= 1
    V[0][c] += 1; V[4][c] += 1
    W[0][c] += 1; W[3][c] += 1
    c += 1

    # M2 = (A[1,0]+A[1,1]) * B[0,0]
    # A[1,0]=U[3], A[1,1]=U[4]
    # B[0,0]=V[0]
    # C[1,0]=W[2]
    U[3][c] += 1; U[4][c] += 1
    V[0][c] += 1
    W[2][c] += 1
    c += 1

    # M3 = (A[0,0]+A[0,1]) * B[1,1]
    # A[0,0]=U[0], A[0,1]=U[1]
    # B[1,1]=V[4]
    # C[0,1]=W[1], C[1,1]=W[3]
    U[0][c] += 1; U[1][c] += 1
    V[4][c] += 1
    W[1][c] += 1; W[3][c] += 1
    c += 1

    # M4 = (A[1,1]-A[0,0]) * (B[0,0]+B[0,1])
    # A[1,1]=U[4], A[0,0]=U[0]
    # B[0,0]=V[0], B[0,1]=V[1]
    # C[0,0]=W[0], C[1,0]=W[2]
    U[4][c] += 1; U[0][c] -= 1
    V[0][c] += 1; V[1][c] += 1
    W[0][c] += 1; W[2][c] += 1
    c += 1

    # M5 = (A[0,0]+A[1,0]) * (B[0,0]+B[0,1])
    # A[0,0]=U[0], A[1,0]=U[3]
    # B[0,0]=V[0], B[0,1]=V[1]
    # C[0,1]=W[1], C[1,1]=W[3]
    U[0][c] += 1; U[3][c] += 1
    V[0][c] += 1; V[1][c] += 1
    W[1][c] += 1; W[3][c] += 1
    c += 1

    # M6 = (A[0,1]-A[1,1]) * (B[1,0]+B[1,1])
    # A[0,1]=U[1], A[1,1]=U[4]
    # B[1,0]=V[3], B[1,1]=V[4]
    # C[1,1]=W[3]
    U[1][c] += 1; U[4][c] -= 1
    V[3][c] += 1; V[4][c] += 1
    W[3][c] += 1
    c += 1

    # M7 = (A[1,0]-A[0,0]) * (B[0,0]+B[0,1])
    # A[1,0]=U[3], A[0,0]=U[0]
    # B[0,0]=V[0], B[0,1]=V[1]
    # C[0,0]=W[0]
    U[3][c] += 1; U[0][c] -= 1
    V[0][c] += 1; V[1][c] += 1
    W[0][c] += 1
    c += 1

    # --- Part 2: j in {0,1}, k=2 ---
    # C[0,2] += A[0,0]B[0,2] + A[0,1]B[1,2]
    # C[1,2] += A[1,0]B[0,2] + A[1,1]B[1,2]
    
    # Term for A[0,0]B[0,2] -> C[0,2]
    U[0][c] += 1
    V[2][c] += 1  # j=0, k=2 -> 0*3+2=2
    W[4][c] += 1  # k=2, i=0 -> 2*2+0=4
    c += 1

    # Term for A[0,1]B[1,2] -> C[0,2]
    U[1][c] += 1
    V[5][c] += 1  # j=1, k=2 -> 1*3+2=5
    W[4][c] += 1
    c += 1

    # Term for A[1,0]B[0,2] -> C[1,2]
    U[3][c] += 1
    V[2][c] += 1
    W[5][c] += 1  # k=2, i=1 -> 2*2+1=5
    c += 1

    # Term for A[1,1]B[1,2] -> C[1,2]
    U[4][c] += 1
    V[5][c] += 1
    W[5][c] += 1
    c += 1

    # --- Part 3: j=2, all k ---
    # C[i,k] += A[i,2]B[2,k]
    
    # A[0,2]B[2,0] -> C[0,0]
    U[2][c] += 1
    V[6][c] += 1  # j=2, k=0 -> 2*3+0=6
    W[0][c] += 1
    c += 1

    # A[0,2]B[2,1] -> C[0,1]
    U[2][c] += 1
    V[7][c] += 1  # j=2, k=1 -> 2*3+1=7
    W[1][c] += 1
    c += 1

    # A[0,2]B[2,2] -> C[0,2]
    U[2][c] += 1
    V[8][c] += 1  # j=2, k=2 -> 2*3+2=8
    W[4][c] += 1
    c += 1

    # A[1,2]B[2,0] -> C[1,0]
    U[5][c] += 1
    V[6][c] += 1
    W[2][c] += 1
    c += 1

    # A[1,2]B[2,1] -> C[1,1]
    U[5][c] += 1
    V[7][c] += 1
    W[3][c] += 1
    c += 1

    # A[1,2]B[2,2] -> C[1,2]
    U[5][c] += 1
    V[8][c] += 1
    W[5][c] += 1
    c += 1

    if c == rank and check_decomposition(U, V, W, rank):
        return {"U": U, "V": V, "W": W}

    # Fallback to schoolbook
    rank_schoolbook = n * m * p
    U_s = [[0]*rank_schoolbook for _ in range(n*m)]
    V_s = [[0]*rank_schoolbook for _ in range(m*p)]
    W_s = [[0]*rank_schoolbook for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_s[i*m+j][col] = 1
                V_s[j*p+k][col] = 1
                W_s[k*n+i][col] = 1
                col += 1
    return {"U": U_s, "V": V_s, "W": W_s}
