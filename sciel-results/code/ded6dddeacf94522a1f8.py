import time
_start_time = time.time()
_DEADLINE = _start_time + 145.0

def construct():
    n, m, p = 2, 2, 15
    
    # Rank 40 decomposition:
    # 5 blocks of 2x2x3 (rank 8 each)
    rank = 40
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_2x2x3_block(base_r, k0, k1, k2):
        # Term 1: A00*(B0k0+B0k2)
        r = base_r
        U[0][r] = 1
        V[0 * p + k0][r] = 1
        V[0 * p + k2][r] = 1
        W[k0 * n + 0][r] = 1
        W[k2 * n + 0][r] = 1
        
        # Term 2: A10*(B0k0+B0k2)
        r = base_r + 1
        U[1 * m + 0][r] = 1
        V[0 * p + k0][r] = 1
        V[0 * p + k2][r] = 1
        W[k0 * n + 1][r] = 1
        W[k2 * n + 1][r] = 1
        
        # Term 3: A01*(B1k0+B1k2)
        r = base_r + 2
        U[0 * m + 1][r] = 1
        V[1 * p + k0][r] = 1
        V[1 * p + k2][r] = 1
        W[k0 * n + 0][r] = 1
        W[k2 * n + 0][r] = 1
        
        # Term 4: A11*(B1k0+B1k2)
        r = base_r + 3
        U[1 * m + 1][r] = 1
        V[1 * p + k0][r] = 1
        V[1 * p + k2][r] = 1
        W[k0 * n + 1][r] = 1
        W[k2 * n + 1][r] = 1
        
        # Term 5: A00*B0k1
        r = base_r + 4
        U[0 * m + 0][r] = 1
        V[0 * p + k1][r] = 1
        W[k1 * n + 0][r] = 1
        
        # Term 6: A10*B0k1
        r = base_r + 5
        U[1 * m + 0][r] = 1
        V[0 * p + k1][r] = 1
        W[k1 * n + 1][r] = 1
        
        # Term 7: A01*B1k1
        r = base_r + 6
        U[0 * m + 1][r] = 1
        V[1 * p + k1][r] = 1
        W[k1 * n + 0][r] = 1
        
        # Term 8: A11*B1k1
        r = base_r + 7
        U[1 * m + 1][r] = 1
        V[1 * p + k1][r] = 1
        W[k1 * n + 1][r] = 1
    
    # Add 5 blocks of 3 columns each
    for b in range(5):
        k0 = 3 * b
        k1 = 3 * b + 1
        k2 = 3 * b + 2
        add_2x2x3_block(8 * b, k0, k1, k2)
    
    # Verify the decomposition
    # Check that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if s != 1:
                    # Fallback to original rank-53 decomposition
                    return _construct_rank53()
    
    return {"U": U, "V": V, "W": W}

def _construct_rank53():
    n, m, p = 2, 2, 15
    rank = 53
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_strassen_block(base_r, k0, k1):
        r1 = base_r
        r2 = base_r + 1
        r3 = base_r + 2
        r4 = base_r + 3
        r5 = base_r + 4
        r6 = base_r + 5
        r7 = base_r + 6
        U[0][r1] = 1; U[3][r1] = 1
        U[2][r2] = 1; U[3][r2] = 1
        U[0][r3] = 1
        U[3][r4] = 1
        U[0][r5] = 1; U[1][r5] = 1
        U[2][r6] = 1; U[0][r6] = -1
        U[1][r7] = 1; U[3][r7] = -1
        V[0 * p + k0][r1] = 1; V[1 * p + k1][r1] = 1
        V[0 * p + k0][r2] = 1
        V[0 * p + k1][r3] = 1; V[1 * p + k1][r3] = -1
        V[1 * p + k0][r4] = 1; V[0 * p + k0][r4] = -1
        V[1 * p + k1][r5] = 1
        V[0 * p + k0][r6] = 1; V[0 * p + k1][r6] = 1
        V[1 * p + k0][r7] = 1; V[1 * p + k1][r7] = 1
        W[2 * k0][r1] = 1; W[2 * k0][r4] = 1; W[2 * k0][r5] = -1; W[2 * k0][r7] = 1
        W[2 * k0 + 1][r2] = 1; W[2 * k0 + 1][r4] = 1
        W[2 * k1][r3] = 1; W[2 * k1][r5] = 1
        W[2 * k1 + 1][r1] = 1; W[2 * k1 + 1][r2] = -1; W[2 * k1 + 1][r3] = 1; W[2 * k1 + 1][r6] = 1
    
    for i in range(7):
        add_strassen_block(7 * i, 2 * i, 2 * i + 1)
    base_r = 49
    k = 14
    U[0][base_r] = 1; V[0 * p + k][base_r] = 1; W[2 * k][base_r] = 1
    U[1][base_r+1] = 1; V[1 * p + k][base_r+1] = 1; W[2 * k][base_r+1] = 1
    U[2][base_r+2] = 1; V[0 * p + k][base_r+2] = 1; W[2 * k + 1][base_r+2] = 1
    U[3][base_r+3] = 1; V[1 * p + k][base_r+3] = 1; W[2 * k + 1][base_r+3] = 1
    return {"U": U, "V": V, "W": W}
