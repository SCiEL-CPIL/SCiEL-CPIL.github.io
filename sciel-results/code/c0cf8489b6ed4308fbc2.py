import time

def construct():
    n, m, p = 3, 3, 7
    rank = n * m * p  # 63
    
    # Deadline guard: ensure we finish well before 150s
    start_time = time.time()
    deadline = start_time + 120.0
    
    # Initialize U, V, W with zeros
    # Using lists of integers to ensure exact "integer/half-integer" entries
    # U: (n*m, rank) -> (9, 63)
    # V: (m*p, rank) -> (21, 63)
    # W: (p*n, rank) -> (21, 63)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Check deadline just in case, though this loop is trivial
                if time.time() > deadline - 5.0:
                    # If we hit the deadline, return what we have (should be complete by now)
                    return {"U": U, "V": V, "W": W}
                
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
            if time.time() > deadline - 5.0:
                return {"U": U, "V": V, "W": W}
        if time.time() > deadline - 5.0:
            return {"U": U, "V": V, "W": W}
            
    return {"U": U, "V": V, "W": W}
