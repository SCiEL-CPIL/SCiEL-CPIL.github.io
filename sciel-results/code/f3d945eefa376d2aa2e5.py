"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    def schoolbook():
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
    
    best = schoolbook()
    best_rank = n * m * p
    
    # Verify: check all diagonal and off-diagonal entries
    def verify(U, V, W):
        rank = len(U[0]) if U else 0
        if rank == 0:
            return False
        
        # Check diagonal: T[i*m+j, j*p+k, k*n+i] = 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = sum(U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] for c in range(rank))
                    if abs(s - 1) > 1e-9:
                        return False
        
        # Check off-diagonal: T[a,b,c] = 0 for (a,b,c) not of the form (i*m+j, j*p+k, k*n+i)
        # This is expensive, so do a sampled check
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    # Check if (a,b,c) is a valid diagonal entry
                    valid = False
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                if a == i*m+j and b == j*p+k and c == k*n+i:
                                    valid = True
                                    break
                            if valid:
                                break
                        if valid:
                            break
                    if not valid:
                        s = sum(U[a][d] * V[b][d] * W[c][d] for d in range(rank))
                        if abs(s) > 1e-9:
                            return False
        return True
    
    # Try random search for lower rank
    random.seed(12345)
    
    for target_rank in range(best_rank - 1, 50, -1):
        if time.time() > deadline - 15:
            break
        
        max_attempts = 20
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
            
            vals = [-1, -0.5, 0, 0.5, 1]
            U = [[random.choice(vals) for _ in range(target_rank)] for _ in range(n*m)]
            V = [[random.choice(vals) for _ in range(target_rank)] for _ in range(m*p)]
            W = [[random.choice(vals) for _ in range(target_rank)] for _ in range(p*n)]
            
            if verify(U, V, W):
                best = {"U": U, "V": V, "W": W}
                best_rank = target_rank
                break
    
    return best
# EVOLVE-BLOCK-END
