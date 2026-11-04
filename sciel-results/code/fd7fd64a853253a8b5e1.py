"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    best_result = None
    best_rank = n * m * p  # schoolbook
    
    # Schoolbook as fallback
    U = [[0] * best_rank for _ in range(n * m)]
    V = [[0] * best_rank for _ in range(m * p)]
    W = [[0] * best_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    best_result = {"U": U, "V": V, "W": W}
    
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Try to find better decomposition via random search + verification
    # Lower bound: max(n*m, m*p, p*n)
    lb = max(n * m, m * p, p * n)
    
    for rank in range(lb, best_rank):
        if time.time() > deadline:
            break
        
        # Try multiple random initializations for this rank
        max_trials = 50
        for trial in range(max_trials):
            if time.time() > deadline:
                break
            
            # Random initialization with small rational entries
            U_rand = np.random.randint(0, 3, (n * m, rank)) - 1  # entries in {-1, 0, 1}
            V_rand = np.random.randint(0, 3, (m * p, rank)) - 1
            W_rand = np.random.randint(0, 3, (p * n, rank)) - 1
            
            # Check if this decomposes the tensor exactly
            # Tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
            # For matrix mult: T[i,j,k] = 1 if we're computing (AB)[i,k] = sum_j A[i,j]*B[j,k]
            # So the tensor should have T[i,j,k] = 1 for all i,j,k? No.
            
            # Actually, the matrix multiplication tensor M_{n,m,p} has entries:
            # M[i,j,k] = 1 for all i in [n], j in [m], k in [p]
            # And the decomposition is: M[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
            
            # So we need: for all i,j,k: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1
            
            # This is a system of n*m*p equations
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for c in range(rank):
                            s += U_rand[i*m+j, c] * V_rand[j*p+k, c] * W_rand[k*n+i, c]
                        if abs(s - 1.0) > 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if valid:
                # Found a valid decomposition!
                best_rank = rank
                best_result = {
                    "U": U_rand.tolist(),
                    "V": V_rand.tolist(),
                    "W": W_rand.tolist()
                }
                break  # Move to next rank
        
        if rank < best_rank:
            # If we found something at this rank, continue to lower ranks
            pass
    
    return best_result
# EVOLVE-BLOCK-END
