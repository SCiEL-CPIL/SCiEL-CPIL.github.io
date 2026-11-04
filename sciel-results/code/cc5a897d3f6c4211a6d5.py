def _try_rank17(n, m, p):
    if n != 2 or m != 3 or p != 3:
        return None
    
    rank = 17
    U = [[0.0]*rank for _ in range(6)]
    V = [[0.0]*rank for _ in range(9)]
    W = [[0.0]*rank for _ in range(6)]
    
    # Strassen subblock: A[0:2, 0:2] x B[0:2, 0:2]
    # P1 = (A00+A11)(B00+B10)
    U[0][0] = 1; U[4][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[2][0] = 1; W[1][0] = 1; W[3][0] = 1
    
    # P2 = (A10+A11)B00
    U[3][1] = 1; U[4][1] = 1
    V[0][1] = 1
    W[1][1] = 2  # contributes to C[1,0] twice? No...
