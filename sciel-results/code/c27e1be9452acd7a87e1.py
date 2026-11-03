import time

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    # Rank 39 decomposition
    rank = 39
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: Strassen for j in {0,1} and k pairs
    # j indices: 0, 1
    # k pairs: (0,1), (2,3), (4,5)
    k_pairs = [(0, 1), (2, 3), (4, 5)]
    
    for k1, k2 in k_pairs:
        # M1
        U[0][col] = 1; U[4][col] = 1
        V[k1][col] = 1; V[7+k2][col] = 1
        W[2*k1][col] = 1; W[2*k1+1][col] = 0; W[2*k2][col] = 1; W[2*k2+1][col] = 1
        col += 1
        # M2
        U[3][col] = 1; U[4][col] = 1
        V[k1][col] = 1
        W[2*k1][col] = 0; W[2*k1+1][col] = 1; W[2*k2][col] = 1; W[2*k2+1][col] = -1
        col += 1
        # M3
        U[0][col] = 1
        V[k2][col] = 1; V[7+k2][col] = -1
        W[2*k1][col] = 0; W[2*k1+1][col] = 0; W[2*k2][col] = 1; W[2*k2+1][col] = 1
        col += 1
        # M4
        U[4][col] = 1
        V[7+k1][col] = 1; V[k1][col] = -1
        W[2*k1][col] = 1; W[2*k1+1][col] = 0; W[2*k2][col] = 0; W[2*k2+1][col] = 0
        col += 1
        # M5
        U[0][col] = 1; U[1][col] = 1
        V[7+k2][col] = 1
        W[2*k1][col] = -1; W[2*k1+1][col] = 0; W[2*k2][col] = -1; W[2*k2+1][col] = 0
        col += 1
        # M6
        U[3][col] = 1; U[0][col] = -1
        V[k1][col] = 1; V[k2][col] = 1
        W[2*k1][col] = 0; W[2*k1+1][col] = 0; W[2*k2][col] = 0; W[2*k2+1][col] = 1
        col += 1
        # M7
        U[1][col] = 1; U[4][col] = -1
        V[7+k1][col] = 1; V[7+k2][col] = 1
        W[2*k1][col] = -1; W[2*k1+1][col] = 1; W[2*k2][col] = 0; W[2*k2+1][col] = 1
        col += 1
        
    # Part 2: Schoolbook for j in {0,1}, k=6
    # i=0, j=0
    U[0][col] = 1; V[6][col] = 1; W[12][col] = 1; col += 1
    # i=0, j=1
    U[1][col] = 1; V[13][col] = 1; W[12][col] = 1; col += 1
    # i=1, j=0
    U[3][col] = 1; V[6][col] = 1; W[13][col] = 1; col += 1
    # i=1, j=1
    U[4][col] = 1; V[13][col] = 1; W[13][col] = 1; col += 1
    
    # Part 3: Schoolbook for j=2
    for i in range(n):
        for k in range(p):
            u_idx = i * m + 2
            v_idx = 2 * p + k
            w_idx = k * n + i
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
