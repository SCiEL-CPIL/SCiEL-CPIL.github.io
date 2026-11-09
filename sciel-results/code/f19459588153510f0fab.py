import numpy as np
import time

DIMENSIONS = [3, 5, 5]

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Rank for 1x5x5 is 5
    # Total rank is n * 5 = 15
    rank = n * 5
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Fill in the factors
    # For each row i of A (0 to n-1)
    for i in range(n):
        # Local rank indices 0 to 4
        for r_prime in range(5):
            r = i * 5 + r_prime
            
            # U factors: U[i*m + j, r] = 1 for all j
            for j in range(m):
                U[i * m + j][r] = 1
                
            # V factors: V[j*p + k, r] = delta[k, r_prime]
            for j in range(m):
                for k in range(p):
                    if k == r_prime:
                        V[j * p + k][r] = 1
                        
            # W factors: W[k'*n + i, r] = delta[k', r_prime]
            for k_prime in range(p):
                if k_prime == r_prime:
                    W[k_prime * n + i][r] = 1
                    
    return {"U": U, "V": V, "W": W}
