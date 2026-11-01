import time

DIMENSIONS = [2, 2, 15]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Strassen (2,2,2) decomposition
    sU = [
        [[1,0],[0,1]],
        [[0,0],[1,1]],
        [[1,0],[0,0]],
        [[0,0],[0,1]],
        [[1,1],[0,0]],
        [[-1,0],[1,0]],
        [[0,1],[0,-1]],
    ]
    sV = [
        [[1,0],[0,1]],
        [[1,0],[0,0]],
        [[0,1],[-1,0]],
        [[-1,0],[0,1]],
        [[0,0],[0,1]],
        [[1,0],[1,0]],
        [[0,1],[0,1]],
    ]
    sW = [
        [[1,0],[0,1]],
        [[0,1],[0,0]],
        [[0,0],[1,0]],
        [[-1,1],[0,0]],
        [[0,1],[0,0]],
        [[0,1],[0,0]],
        [[-1,0],[0,1]],
    ]
    
    # Verify Strassen
    for i in range(2):
        for j in range(2):
            for k in range(2):
                total = sum(sU[r][i][j] * sV[r][j][k] * sW[r][k][i] for r in range(7))
                if total != 1:
                    raise ValueError(f"Strassen verify failed")
    
    # Build (2,2,15): 7 pairs + 1 single
    rank = 49 + 4
    U = [[0] * rank for _ in range(4)]
    V = [[0] * rank for _ in range(30)]
    W = [[0] * rank for _ in range(30)]
    
    col = 0
    for pair in range(7):
        k0 = 2 * pair
        k1 = k0 + 1
        for r in range(7):
            for i in range(2):
                for j in range(2):
                    U[i*2+j][col] = sU[r][i][j]
            for j in range(2):
                for kk in range(2):
                    V[j*15 + k0 + kk][col] = sV[r][j][kk]
            for kk in range(2):
                for i in range(2):
                    W[(k0+kk)*2 + i][col] = sW[r][kk][i]
            col += 1
    
    # Single column 14: schoolbook
    for i in range(2):
        for j in range(2):
            U[i*2+j][col] = 1
            V[j*15+14][col] = 1
            W[14*2+i][col] = 1
            col += 1
    
    assert col == rank
    
    # Verify
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = sum(U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] for r in range(rank))
                if total != 1:
                    raise ValueError(f"Verify failed at ({i},{j},{k}): {total}")
    
    return {"U": U, "V": V, "W": W}
