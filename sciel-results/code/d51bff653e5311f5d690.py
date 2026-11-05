"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    # Schoolbook decomposition as guaranteed fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    # Precompute tensor indices
    tensor_indices = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    
    def verify(U_list, V_list, W_list, rank):
        """Verify exact decomposition with integer/half-integer entries."""
        for (i, j, k) in tensor_indices:
            val = 0.0
            for c in range(rank):
                val += U_list[i * m + j][c] * V_list[j * p + k][c] * W_list[k * n + i][c]
            if abs(val - 1.0) > 1e-9:
                return False
        return True
    
    # Try to find better decompositions using structured search
    # Strategy: Use block decomposition 3x8 = 3x3 + 3x3 + 3x2
    # For each block, try to find efficient decompositions
    
    # Try ranks from 71 down to 50
    for rank in range(best_rank - 1, 49, -1):
        if time.time() > deadline:
            break
        
        # Try multiple random initializations
        max_attempts = 5
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            # Initialize with small random half-integer values
            np.random.seed(attempt * 1000 + rank)
            U = np.random.randint(-4, 5, (n * m, rank)) * 0.5
            V = np.random.randint(-4, 5, (m * p, rank)) * 0.5
            W = np.random.randint(-4, 5, (p * n, rank)) * 0.5
            
            # Check if this decomposes the tensor exactly
            valid = True
            for (i, j, k) in tensor_indices:
                val = 0.0
                for c in range(rank):
                    val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(val - 1.0) > 1e-9:
                    valid = False
                    break
            
            if valid:
                U_list = [[float(x) for x in row] for row in U]
                V_list = [[float(x) for x in row] for row in V]
                W_list = [[float(x) for x in row] for row in W]
                
                # Round to nearest half-integer and re-verify
                U_list = [[round(x * 2) / 2 for x in row] for row in U_list]
                V_list = [[round(x * 2) / 2 for x in row] for row in V_list]
                W_list = [[round(x * 2) / 2 for x in row] for row in W_list]
                
                if verify(U_list, V_list, W_list, rank):
                    best_rank = rank
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                    break  # Found valid decomposition for this rank
        
        if best_rank < rank:
            break  # Found better rank, continue to next lower rank
    
    return best_result
# EVOLVE-BLOCK-END
