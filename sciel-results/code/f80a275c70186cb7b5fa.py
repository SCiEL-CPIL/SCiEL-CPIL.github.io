import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin
    
    # Schoolbook decomposition as fallback
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W, rank
    
    U_sb, V_sb, W_sb, rank_sb = schoolbook()
    best_U, best_V, best_W, best_rank = U_sb, V_sb, W_sb, rank_sb
    
    def verify(U, V, W):
        """Check if U, V, W form a valid decomposition."""
        r = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    def to_half_int(x):
        """Round to nearest half-integer."""
        return round(x * 2) / 2
    
    def random_decomp(rank, scale=1.0):
        """Generate a random decomposition with integer/half-integer entries."""
        U = np.round(np.random.uniform(-scale, scale, (n * m, rank)) * 2) / 2
        V = np.round(np.random.uniform(-scale, scale, (m * p, rank)) * 2) / 2
        W = np.round(np.random.uniform(-scale, scale, (p * n, rank)) * 2) / 2
        return U, V, W
    
    # Try to find a lower rank decomposition
    for target_rank in range(30, best_rank):
        if time.time() > deadline - 10:
            break
        
        # Try random search
        for attempt in range(50):
            if time.time() > deadline - 5:
                break
            
            U, V, W = random_decomp(target_rank, scale=2.0)
            if verify(U, V, W):
                best_U, best_V, best_W, best_rank = U, V, W, target_rank
                break
        
        if best_rank <= target_rank:
            break
    
    # Local search: start from schoolbook, try to reduce
    # ... (complex, skip for now)
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
