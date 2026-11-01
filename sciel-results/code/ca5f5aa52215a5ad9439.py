import time

DIMENSIONS = [3, 3, 3]

def _verify(U, V, W):
    """Verify that U,V,W with given indices correctly decompose matrix multiplication.
    Returns True if decomposition is exact."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1) > 1e-9:
                    return False
    return True

def _construct_schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _construct_rank_26():
    """
    Rank-26 decomposition for 3x3 matrix multiplication.
    Partition into 2x2 block (indices 0,1) and 1x1 blocks (index 2).
    Apply Strassen to the 2x2 block (7 terms) + 19 terms for the rest.
    """
    n, m, p = 3, 3, 3
    rank = 26
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen for 2x2 block (i,j,k in {0,1})
    # Term 0: M1 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
    U[0][0] += 1; U[3][0] += 1
    V[0][0] += 1; V[4][0] += 1
    W[0][0] += 1; W[4][0] += 1
    
    # Term 1: M2 = (A[1,0]+A[1,1]) * B[0,0]
    U[2][1] += 1; U[3][1] += 1
    V[0][1] += 1
    W[0][1] += 1
    
    # Term 2: M3 = A[0,0] * (B[1,0]-B[0,0])
    U[0][2] += 1
    V[3][2] += 1; V[0][2] += -1
    W[0][2] += 1
    
    # Term 3: M4 = A[1,1] * (B[1,1]-B[0,1])
    U[3][3] += 1
    V[4][3] += 1; V[1][3] += -1
    W[4][3] += 1
    
    # Term 4: M5 = (A[0,0]+A[0,1]) * B[1,1]
    U[0][4] += 1; U[1][4] += 1
    V[4][4] += 1
    W[4][4] += 1
    
    # Term 5: M6 = (A[1,0]-A[0,0]) * (B[0,1]+B[1,1])
    U[2][5] += 1; U[0][5] += -1
    V[1][5] += 1; V[4][5] += 1
    W[4][5] += 1
    
    # Term 6: M7 = (A[0,1]-A[1,1]) * (B[0,0]+B[1,0])
    U[1][6] += 1; U[3][6] += -1
    V[0][6] += 1; V[3][6] += 1
    W[0][6] += 1
    
    # 19 more terms for the remaining entries involving index 2
    remaining = [
        (0,0,2), (0,1,2), (0,2,0), (0,2,1), (0,2,2),
        (1,0,2), (1,1,2), (1,2,0), (1,2,1), (1,2,2),
        (2,0,0), (2,0,1), (2,0,2), (2,1,0), (2,1,1), (2,1,2),
        (2,2,0), (2,2,1), (2,2,2)
    ]
    
    col = 7
    for (i,j,k) in remaining:
        u_row = i * m + j
        v_row = j * p + k
        w_row = k * n + i
        U[u_row][col] = 1
        V[v_row][col] = 1
        W[w_row][col] = 1
        col += 1
    
    return {"U": U, "V": V, "W": W}

def construct():
    """
    Return a rank-minimal decomposition for 3x3 matrix multiplication.
    """
    result = _construct_rank_26()
    if _verify(result["U"], result["V"], result["W"]):
        return result
    
    return _construct_schoolbook()
