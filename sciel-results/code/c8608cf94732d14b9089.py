import time
import random

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 60
    
    def verify(U, V, W, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if s != 1:
                        return False
        return True
    
    # Try ranks from 1 to schoolbook-1
    for r in range(1, schoolbook_rank):
        if time.time() > deadline - 5:
            break
        
        # Limit trials per rank to keep total time bounded
        max_trials = 50000
        for attempt in range(max_trials):
            if time.time() > deadline - 5:
                break
            
            # Use small integer values
            vals = [0, 1, -1]
            U = [[random.choice(vals) for _ in range(r)] for _ in range(n*m)]
            V = [[random.choice(vals) for _ in range(r)] for _ in range(m*p)]
            W = [[random.choice(vals) for _ in range(r)] for _ in range(p*n)]
            
            if verify(U, V, W, r):
                return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook
    r = schoolbook_rank
    U = [[0]*r for _ in range(n*m)]
    V = [[0]*r for _ in range(m*p)]
    W = [[0]*r for _ in range(p*n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][column] = 1
                V[j*p+k][column] = 1
                W[k*n+i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
