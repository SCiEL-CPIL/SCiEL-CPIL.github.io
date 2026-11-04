import time

DIMENSIONS = [2, 4, 5]

def construct():
    """
    Constructs the matrix multiplication tensor decomposition for 2x4 by 4x5.
    
    Key insight: The tensor T[a,b,c] = 1 iff j_a = j_b AND i_a = i_c, where
    a = i_a*m + j_a, b = j_b*p + k_b, c = k_c*n + i_c.
    
    This structure allows a rank (m*n) decomposition:
    T[a,b,c] = sum_{j,i} f_{i,j}(a) * g_j(b) * h_i(c)
    where f_{i,j}(a) = 1 iff a = i*m+j,
           g_j(b) = 1 iff b//p... wait, b//m? No.
    
    Let me re-derive:
    a = i_a * m + j_a, so i_a = a // m, j_a = a % m
    b = j_b * p + k_b, so j_b = b // p, k_b = b % p
    c = k_c * n + i_c, so i_c = c % n, k_c = c // n
    
    T[a,b,c] = 1 iff j_a = j_b AND i_a = i_c
    = delta(j_a, j_b) * delta(i_a, i_c)
    = (sum_j delta(j_a,j) delta(j_b,j)) * (sum_i delta(i_a,i) delta(i_c,i))
    = sum_{j,i} delta(j_a,j) delta(i_a,i) * delta(j_b,j) * delta(i_c,i)
    = sum_{j,i} f_{i,j}(a) * g_j(b) * h_i(c)
    
    where f_{i,j}(a) = 1 iff a = i*m + j
           g_j(b) = 1 iff b // p = j
           h_i(c) = 1 iff c % n = i
    
    Rank = m * n = 4 * 2 = 8
    """
    start_time = time.time()
    deadline = start_time + 140

    n, m, p = DIMENSIONS  # n=2, m=4, p=5
    rank = m * n  # 8
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    for j in range(m):
        for i in range(n):
            # U: 1 at position i*m + j
            U[i * m + j][r] = 1
            # V: 1 at positions j*p + k for k = 0..p-1
            for k in range(p):
                V[j * p + k][r] = 1
            # W: 1 at positions k*n + i for k = 0..p-1
            for k in range(p):
                W[k * n + i][r] = 1
            r += 1
            if time.time() > deadline:
                return {"U": U, "V": V, "W": W}
        if time.time() > deadline:
            return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
