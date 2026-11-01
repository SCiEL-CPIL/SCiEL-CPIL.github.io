import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10 seconds safety margin
    
    n, m, p = 2, 2, 8
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # A indices: 0:00, 1:01, 2:10, 3:11
    # B indices: j*p + k
    # C indices: k*n + i
    
    for b in range(4):
        if time.time() > deadline:
            break
        # B cols: 2b, 2b+1
        # C cols: 2b, 2b+1
        
        # s=1
        r = 7 * b + 0
        U[0][r] += 1; U[3][r] += 1
        V[2*b][r] += 1; V[8+2*b+1][r] += 1
        W[4*b][r] += 1; W[4*b+3][r] += 1
        
        # s=2
        r = 7 * b + 1
        U[2][r] += 1; U[3][r] += 1
        V[2*b][r] += 1
        W[4*b+2][r] += 1; W[4*b+3][r] -= 1
        
        # s=3
        r = 7 * b + 2
        U[0][r] += 1
        V[2*b+1][r] += 1; V[8+2*b+1][r] -= 1
        W[4*b+1][r] += 1; W[4*b+3][r] += 1
        
        # s=4
        r = 7 * b + 3
        U[3][r] += 1
        V[8+2*b][r] += 1; V[2*b][r] -= 1
        W[4*b][r] += 1; W[4*b+2][r] += 1
        
        # s=5
        r = 7 * b + 4
        U[0][r] += 1; U[1][r] += 1
        V[8+2*b+1][r] += 1
        W[4*b][r] -= 1; W[4*b+1][r] += 1
        
        # s=6
        r = 7 * b + 5
        U[2][r] += 1; U[0][r] -= 1
        V[2*b][r] += 1; V[2*b+1][r] += 1
        W[4*b+3][r] += 1
        
        # s=7
        r = 7 * b + 6
        U[1][r] += 1; U[3][r] -= 1
        V[8+2*b][r] += 1; V[8+2*b+1][r] += 1
        W[4*b][r] += 1

    return {"U": U, "V": V, "W": W}
