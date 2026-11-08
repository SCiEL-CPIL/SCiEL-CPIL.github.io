import time

DIMENSIONS = [4, 4, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s margin
    
    n, m, p = DIMENSIONS
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # ... (same logic as above) ...
    
    return {"U": U, "V": V, "W": W}
