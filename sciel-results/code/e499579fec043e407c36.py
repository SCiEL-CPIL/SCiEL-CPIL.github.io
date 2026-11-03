import time

def construct():
    start_time = time.time()
    DEADLINE = 120.0
    
    n, m, p = 3, 4, 6
    rank = 24
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # A1 is A[:, 0:2], B1 is B[0:2, :]
    # A2 is A[:, 2:4], B2 is B[2:4, :]
    
    # Process A1 B1
    # j indices in A1: 0, 1
    # j indices in B1: 0, 1
    # k indices: 0..5
    
    for t in range(3):
        k0 = 2 * t
        k1 = 2 * t + 1
        
        # Term 1: A1[:, 0] * B1[0, k0] -> C[:, k0]
        # u = A1[:, 0]
        # v = B1[0, k0] * e_{0, k0}
        # w = e_{:, k0}
        
        # u: A1[:, 0] -> A[:, 0]
        for i in range(n):
            U[i * m + 0][col] = 1
        # v: B1[0, k0] -> B[0, k0]
        V[0 * p + k0][col] = 1
        # w: e_{:, k0} -> C[:, k0] -> W[k0 * n + i]
        for i in range(n):
            W[k0 * n + i][col] = 1
        col += 1
        
        # Term 2: A1[:, 1] * B1[1, k0] -> C[:, k0]
        # u = A1[:, 1]
        # v = B1[1, k0] * e_{1, k0}
        # w = e_{:, k0}
        
        for i in range(n):
            U[i * m + 1][col] = 1
        V[1 * p + k0][col] = 1
        for i in range(n):
            W[k0 * n + i][col] = 1
        col += 1
        
        # Term 3: A1[:, 0] * B1[0, k1] -> C[:, k1]
        # u = A1[:, 0]
        # v = B1[0, k1] * e_{0, k1}
        # w = e_{:, k1}
        
        for i in range(n):
            U[i * m + 0][col] = 1
        V[0 * p + k1][col] = 1
        for i in range(n):
            W[k1 * n + i][col] = 1
        col += 1
        
        # Term 4: A1[:, 1] * B1[1, k1] -> C[:, k1]
        # u = A1[:, 1]
        # v = B1[1, k1] * e_{1, k1}
        # w = e_{:, k1}
        
        for i in range(n):
            U[i * m + 1][col] = 1
        V[1 * p + k1][col] = 1
        for i in range(n):
            W[k1 * n + i][col] = 1
        col += 1
        
    # Process A2 B2
    # j indices in A2: 2, 3
    # j indices in B2: 2, 3
    # k indices: 0..5
    
    for t in range(3):
        k0 = 2 * t
        k1 = 2 * t + 1
        
        # Term 1: A2[:, 0] * B2[0, k0] -> C[:, k0]
        # u = A2[:, 0] -> A[:, 2]
        # v = B2[0, k0] -> B[2, k0]
        # w = e_{:, k0}
        
        for i in range(n):
            U[i * m + 2][col] = 1
        V[2 * p + k0][col] = 1
        for i in range(n):
            W[k0 * n + i][col] = 1
        col += 1
        
        # Term 2: A2[:, 1] * B2[1, k0] -> C[:, k0]
        # u = A2[:, 1] -> A[:, 3]
        # v = B2[1, k0] -> B[3, k0]
        # w = e_{:, k0}
        
        for i in range(n):
            U[i * m + 3][col] = 1
        V[3 * p + k0][col] = 1
        for i in range(n):
            W[k0 * n + i][col] = 1
        col += 1
        
        # Term 3: A2[:, 0] * B2[0, k1] -> C[:, k1]
        # u = A2[:, 0] -> A[:, 2]
        # v = B2[0, k1] -> B[2, k1]
        # w = e_{:, k1}
        
        for i in range(n):
            U[i * m + 2][col] = 1
        V[2 * p + k1][col] = 1
        for i in range(n):
            W[k1 * n + i][col] = 1
        col += 1
        
        # Term 4: A2[:, 1] * B2[1, k1] -> C[:, k1]
        # u = A2[:, 1] -> A[:, 3]
        # v = B2[1, k1] -> B[3, k1]
        # w = e_{:, k1}
        
        for i in range(n):
            U[i * m + 3][col] = 1
        V[3 * p + k1][col] = 1
        for i in range(n):
            W[k1 * n + i][col] = 1
        col += 1
        
    return {"U": U, "V": V, "W": W}
