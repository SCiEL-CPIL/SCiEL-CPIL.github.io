import time

DIMENSIONS = [4, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 145 # 5s margin
    
    n, m, p = DIMENSIONS
    rank = 49 + 32
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # ... (implementation as above) ...
    
    if time.time() > deadline:
         # Return best effort (which is the full construction if it didn't finish, but it should be fast)
         pass
         
    return {"U": U, "V": V, "W": W}
