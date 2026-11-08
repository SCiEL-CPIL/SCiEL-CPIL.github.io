"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Schoolbook decomposition as fallback
    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank), dtype=int)
        V = np.zeros((mp, rank), dtype=int)
        W = np.zeros((pn, rank), dtype=int)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        return U, V, W, rank
    
    best_U, best_V, best_W, best_rank = schoolbook()
    
    # Precompute indices for faster checking
    indices = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    
    def check_decomposition(U, V, W, rank):
        """Check if U, V, W form a valid decomposition."""
        for i, j, k in indices:
            val = 0
            for c in range(rank):
                val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
            if val != 1:
                return False
        return True
    
    # Try to find a lower rank decomposition using structured search
    # Use a pattern-based initialization
    
    # Try ranks from 32 up to 95
    for rank in range(32, 96):
        if time.time() > deadline:
            break
        
        # Try a few structured initializations
        for attempt in range(20):
            if time.time() > deadline:
                break
            
            # Initialize with a pattern
            if attempt == 0:
                # Try a block structure
                U = np.zeros((nm, rank), dtype=int)
                V = np.zeros((mp, rank), dtype=int)
                W = np.zeros((pn, rank), dtype=int)
                # Fill with a pattern
                for c in range(rank):
                    i = c % n
                    j = (c // n) % m
                    k = (c // (n*m)) % p
                    if i < n and j < m and k < p:
                        U[i*m+j, c] = 1
                        V[j*p+k, c] = 1
                        W[k*n+i, c] = 1
            elif attempt == 1:
                # Try random 0/1
                U = np.random.randint(0, 2, size=(nm, rank))
                V = np.random.randint(0, 2, size=(mp, rank))
                W = np.random.randint(0, 2, size=(pn, rank))
            else:
                # Try random -1/0/1
                U = np.random.randint(-1, 2, size=(nm, rank))
                V = np.random.randint(-1, 2, size=(mp, rank))
                W = np.random.randint(-1, 2, size=(pn, rank))
            
            if check_decomposition(U, V, W, rank):
                best_U, best_V, best_W, best_rank = U, V, W, rank
                break
    
    # Convert to lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
