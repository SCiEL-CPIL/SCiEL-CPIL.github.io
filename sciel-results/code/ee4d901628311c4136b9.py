"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    
    # Schoolbook as fallback
    rank_schoolbook = n * m * p
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    best = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_schoolbook
    
    # Try to find a better decomposition
    # The condition: for all i,k,j,l: sum_c W[k*n+i,c]*U[i*m+j,c]*V[l*p+k,c] = delta(j,l)
    
    # Try ranks from a lower bound upward
    # Lower bound: for each (i,k), need rank at least m=4, so total rank >= n*p*m / (max sharing) 
    # But at minimum, we need enough terms
    
    # Let's try a structured approach: use the fact that we can share across i or k
    
    # Try rank = 100, 90, 80, etc.
    for target_rank in range(100, 111):
        if time.time() > deadline - 10:
            break
            
        # Random initialization
        U = np.zeros((n * m, target_rank), dtype=float)
        V = np.zeros((m * p, target_rank), dtype=float)
        W = np.zeros((p * n, target_rank), dtype=float)
        
        # Initialize with random values
        for c in range(target_rank):
            for i in range(n):
                for j in range(m):
                    U[i * m + j, c] = random.uniform(-1, 1)
            for j in range(m):
                for k in range(p):
                    V[j * p + k, c] = random.uniform(-1, 1)
            for i in range(n):
                for k in range(p):
                    W[k * n + i, c] = random.uniform(-1, 1)
        
        # Check if this is a valid decomposition
        # For each (i,k,j,l), check if sum_c W[k*n+i,c]*U[i*m+j,c]*V[l*p+k,c] == delta(j,l)
        
        valid = True
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    for l in range(m):
                        s = 0.0
                        for c in range(target_rank):
                            s += W[k * n + i, c] * U[i * m + j, c] * V[l * p + k, c]
                        expected = 1.0 if j == l else 0.0
                        if abs(s - expected) > 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            # Convert to list format with integer/half-integer entries
            # Round to nearest half-integer
            U_list = [[round(val * 2) / 2 for val in U[r, :]] for r in range(n * m)]
            V_list = [[round(val * 2) / 2 for val in V[r, :]] for r in range(m * p)]
            W_list = [[round(val * 2) / 2 for val in W[r, :]] for r in range(p * n)]
            
            # Verify with rounded values
            valid_rounded = True
            for i in range(n):
                for k in range(p):
                    for j in range(m):
                        for l in range(m):
                            s = 0.0
                            for c in range(target_rank):
                                s += W_list[k * n + i][c] * U_list[i * m + j][c] * V_list[l * p + k][c]
                            expected = 1.0 if j == l else 0.0
                            if abs(s - expected) > 1e-9:
                                valid_rounded = False
                                break
                        if not valid_rounded:
                            break
                    if not valid_rounded:
                        break
                if not valid_rounded:
                    break
            
            if valid_rounded:
                best = {"U": U_list, "V": V_list, "W": W_list}
                best_rank = target_rank
                break
    
    return best
# EVOLVE-BLOCK-END
