"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
import time
import random
from fractions import Fraction
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Schoolbook fallback
    rank_schoolbook = n * m * p
    U_fb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_fb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_fb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][col] = 1
                V_fb[j * p + k][col] = 1
                W_fb[k * n + i][col] = 1
                col += 1
    best_result = {"U": U_fb, "V": V_fb, "W": W_fb}
    best_rank = rank_schoolbook
    
    # Try to find lower rank decompositions
    # Use random search with integer/half-integer entries
    random.seed(42)
    
    # Try various ranks below schoolbook
    for target_rank in range(rank_schoolbook - 1, max(20, rank_schoolbook // 3), -1):
        if time.time() > deadline - 5:
            break
        
        # Try multiple random initializations
        for attempt in range(50):
            if time.time() > deadline - 5:
                break
            
            # Generate random factor matrices with small integer/half-integer entries
            def rand_entry():
                val = random.randint(-3, 3)
                if random.random() < 0.3:
                    val = val + 0.5
                return val
            
            U = [[rand_entry() for _ in range(target_rank)] for _ in range(n * m)]
            V = [[rand_entry() for _ in range(target_rank)] for _ in range(m * p)]
            W = [[rand_entry() for _ in range(target_rank)] for _ in range(p * n)]
            
            # Check if this decomposition is exact
            # The tensor T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
            # should equal 1 for all (i,j,k)
            is_valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0
                        for c in range(target_rank):
                            s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                        if abs(s - 1) > 1e-9:
                            is_valid = False
                            break
                    if not is_valid:
                        break
                if not is_valid:
                    break
            
            if is_valid:
                # Convert to exact fractions if needed
                U_frac = [[Fraction(x).limit_denominator(1000) for x in row] for row in U]
                V_frac = [[Fraction(x).limit_denominator(1000) for x in row] for row in V]
                W_frac = [[Fraction(x).limit_denominator(1000) for x in row] for row in W]
                
                # Verify with fractions
                is_valid_frac = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = Fraction(0)
                            for c in range(target_rank):
                                s += U_frac[i * m + j][c] * V_frac[j * p + k][c] * W_frac[k * n + i][c]
                            if s != 1:
                                is_valid_frac = False
                                break
                        if not is_valid_frac:
                            break
                    if not is_valid_frac:
                        break
                
                if is_valid_frac:
                    best_rank = target_rank
                    best_result = {"U": U_frac, "V": V_frac, "W": W_frac}
                    break  # Found valid decomposition at this rank, try lower
    
    if time.time() > deadline - 2:
        return best_result
    
    # If we found a better rank, try to polish/verify
    if best_rank < rank_schoolbook:
        return best_result
    
    return best_result
# EVOLVE-BLOCK-END
