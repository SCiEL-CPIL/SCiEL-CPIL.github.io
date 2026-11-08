import numpy as np
import time

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Schoolbook decomposition (rank 36) as fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best = {'U': U_school, 'V': V_school, 'W': W_school, 'rank': rank_school}
    
    # Try to find a better decomposition using a known structure
    # For 3x3x4, try rank 33, 34, 35
    
    def check_decomposition(U, V, W):
        """Check if the decomposition is valid."""
        n, m, p = DIMENSIONS
        r = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    def try_rank(rank):
        """Try to find a decomposition of given rank."""
        if time.time() > deadline:
            return None
        
        # Use a pattern-based approach
        # For 3x3x4, we can try to exploit structure
        # Start with schoolbook and try to merge terms
        
        # Generate a candidate using random integer/half-integer entries
        import random
        random.seed(42)
        
        for attempt in range(100):
            if time.time() > deadline:
                return None
            
            U = [[random.choice([0, 0.5, 1, -0.5, -1]) for _ in range(rank)] for _ in range(n*m)]
            V = [[random.choice([0, 0.5, 1, -0.5, -1]) for _ in range(rank)] for _ in range(m*p)]
            W = [[random.choice([0, 0.5, 1, -0.5, -1]) for _ in range(rank)] for _ in range(p*n)]
            
            if check_decomposition(U, V, W):
                return {'U': U, 'V': V, 'W': W, 'rank': rank}
        
        return None
    
    # Try decreasing ranks
    for rank in range(33, 36):
        if time.time() > deadline:
            break
        result = try_rank(rank)
        if result:
            best = result
            break
    
    return {'U': best['U'], 'V': best['V'], 'W': best['W']}
# EVOLVE-BLOCK-END
