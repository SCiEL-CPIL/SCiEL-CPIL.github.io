import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Schoolbook decomposition as fallback
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
    
    def check_decomposition(U, V, W, R):
        # U: n*m x R, V: m*p x R, W: p*n x R
        # Check: sum_r U[i*m+j, r] * V[l*p+k, r] * W[k*n+i, r] == delta(j,l)
        # Note: V index is j*p+k in schoolbook, but here V row is l*p+k?
        # In schoolbook: V[j*p+k, r] = 1 if j==j_r.
        # My equation: sum_r U[i,j,r] V[l,k,r] W[i,k,r] = delta(j,l)
        # Wait, V in the task is (m*p, R). Row index is j*p+k?
        # In the equation, the second factor is sum_l V[l*p+k, r] B[l,k].
        # So V[row, r] corresponds to B[row//p, row%p].
        # So V[l*p+k, r] is the coefficient for B[l,k].
        # So the equation is correct.
        
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    for l in range(m):
                        s = 0
                        for r in range(R):
                            s += U[i * m + j][r] * V[l * p + k][r] * W[k * n + i][r]
                        target = 1 if j == l else 0
                        if s != target:
                            return False
        return True

    def try_find_decomposition(R, time_limit=5):
        # Random search with local tweaks
        start_time = time.time()
        # Use small integers: -1, 0, 1
        # Maybe allow 0.5?
        # Let's stick to integers first.
        
        # Initialize randomly
        for attempt in range(10000):
            if time.time() - start_time > time_limit:
                return None
            
            U = [[random.choice([-1, 0, 1]) for _ in range(R)] for _ in range(n * m)]
            V = [[random.choice([-1, 0, 1]) for _ in range(R)] for _ in range(m * p)]
            W = [[random.choice([-1, 0, 1]) for _ in range(R)] for _ in range(p * n)]
            
            if check_decomposition(U, V, W, R):
                return {"U": U, "V": V, "W": W}
                
            # Local search: pick a random variable and change it
            # This is very slow for 598 variables.
            # Skip local search for now, rely on more attempts with structured init?
            
        return None

    # Try ranks from 20 to 23
    for R in [20, 21, 22, 23]:
        result = try_find_decomposition(R, time_limit=10)
        if result:
            return result

    # Fallback to schoolbook
    return {"U": U_school, "V": V_school, "W": W_school}
# EVOLVE-BLOCK-END
