import time

DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 2, 2, 13
    
    # Direct rank-52 decomposition: 4 terms per column k
    # C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]
    # C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k]
    # A index: i*m+j = 2*i+j
    # B index: j*p+k = 13*j+k
    # W index: k*n+i = 2*k+i (official trace convention)
    
    rank = 4 * p  # 52
    U = [[0] * rank for _ in range(n * m)]  # 4 x 52
    V = [[0] * rank for _ in range(m * p)]  # 26 x 52
    W = [[0] * rank for _ in range(p * n)]  # 26 x 52
    
    for k in range(p):
        if time.time() > deadline:
            break
        base = 4 * k
        
        # Term 1: A[0,0] * B[0,k] -> C[0,k]
        U[0][base] = 1
        V[k][base] = 1
        W[2*k][base] = 1
        
        # Term 2: A[0,1] * B[1,k] -> C[0,k]
        U[1][base + 1] = 1
        V[13 + k][base + 1] = 1
        W[2*k][base + 1] = 1
        
        # Term 3: A[1,0] * B[0,k] -> C[1,k]
        U[2][base + 2] = 1
        V[k][base + 2] = 1
        W[2*k + 1][base + 2] = 1
        
        # Term 4: A[1,1] * B[1,k] -> C[1,k]
        U[3][base + 3] = 1
        V[13 + k][base + 3] = 1
        W[2*k + 1][base + 3] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
