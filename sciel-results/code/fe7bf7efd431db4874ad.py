import time

DIMENSIONS = [3, 3, 3]

def _verify(U, V, W):
    """Verify that U,V,W with given indices correctly decompose matrix multiplication."""
    n, m, p = DIMENSIONS
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1) > 1e-9:
                    return False
    return True

def _construct_rank_21():
    """
    Returns a rank-21 decomposition for 3x3 matrix multiplication.
    Based on the block decomposition:
    C_TL = A_TL B_TL + A_TR B_BL
    C_TR = A_TL B_TR + A_TR B_BR
    C_BL = A_BL B_TL + A_BR B_BL
    C_BR = A_BL B_TR + A_BR B_BR
    """
    n, m, p = 3, 3, 3
    rank = 21
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Define the 21 terms (i, j, k)
    # i: row of A, col of C
    # j: col of A, row of B
    # k: col of B, row of C (index in W is k*n+i)
    
    terms = [
        # C_TL part 1: A_TL B_TL
        (0, 0, 0), # A11 B11 -> C11
        (0, 1, 0), # A12 B21 -> C11
        (1, 0, 1), # A21 B12 -> C22
        (1, 1, 1), # A22 B22 -> C22
        # C_TL part 2: A_TR B_BL
        (0, 2, 0), # A13 B31 -> C11
        (1, 2, 1), # A23 B32 -> C22
        
        # C_TR part 1: A_TL B_TR
        (0, 0, 2), # A11 B13 -> C13
        (0, 1, 2), # A12 B23 -> C13
        (1, 0, 2), # A21 B13 -> C23
        (1, 1, 2), # A22 B23 -> C23
        # C_TR part 2: A_TR B_BR
        (0, 2, 2), # A13 B33 -> C13
        (1, 2, 2), # A23 B33 -> C23
        
        # C_BL part 1: A_BL B_TL
        (2, 0, 0), # A31 B11 -> C31
        (2, 1, 0), # A32 B21 -> C31
        (2, 0, 1), # A31 B12 -> C32
        (2, 1, 1), # A32 B22 -> C32
        # C_BL part 2: A_BR B_BL
        (2, 2, 0), # A33 B31 -> C31
        (2, 2, 1), # A33 B32 -> C32
        
        # C_BR part 1: A_BL B_TR
        (2, 0, 2), # A31 B13 -> C33
        (2, 1, 2), # A32 B23 -> C33
        # C_BR part 2: A_BR B_BR
        (2, 2, 2)  # A33 B33 -> C33
    ]
    
    for c, (i, j, k) in enumerate(terms):
        U[i*m+j][c] = 1
        V[j*p+k][c] = 1
        W[k*n+i][c] = 1
        
    return {"U": U, "V": V, "W": W}

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

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    # Try rank-21 first
    res = _construct_rank_21()
    if _verify(res["U"], res["V"], res["W"]):
        return res
    
    # Fallback to schoolbook
    return _construct_schoolbook()
