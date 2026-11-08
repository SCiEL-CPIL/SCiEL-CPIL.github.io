"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _check_decomposition(U, V, W, n, m, p):
    """Check if decomposition is valid."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s limit
    
    # Trivial decomposition as fallback
    rank_trivial = n * m * p
    U = [[0] * rank_trivial for _ in range(n * m)]
    V = [[0] * rank_trivial for _ in range(m * p)]
    W = [[0] * rank_trivial for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U, "V": V, "W": W}
    best_rank = rank_trivial
    
    # Try to find lower rank decompositions with small integer/half-integer values
    half_ints = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]
    
    for rank in range(rank_trivial - 1, max(15, rank_trivial - 10), -1):
        if time.time() > deadline:
            break
        
        # Try a limited number of random configurations
        np.random.seed(12345)
        max_attempts = 50
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            U_r = np.random.choice(half_ints, size=(n * m, rank)).tolist()
            V_r = np.random.choice(half_ints, size=(m * p, rank)).tolist()
            W_r = np.random.choice(half_ints, size=(p * n, rank)).tolist()
            
            if _check_decomposition(U_r, V_r, W_r, n, m, p):
                best_result = {"U": U_r, "V": V_r, "W": W_r}
                best_rank = rank
                break
        
        if best_rank < rank_trivial:
            break
    
    return best_result
# EVOLVE-BLOCK-END
