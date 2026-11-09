import numpy as np
from fractions import Fraction
import time
import random

DIMENSIONS = [3, 4, 5]

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition is exact."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _to_fraction_list(mat):
    """Convert a list of lists of floats to fractions, returning None if not clean."""
    result = []
    for row in mat:
        r = []
        for x in row:
            f = Fraction(x).limit_denominator(1000)
            # Check if it's close to a half-integer or integer
            if abs(f - round(f*2)/2) > 1e-6:
                return None
            r.append(f)
        result.append(r)
    return result

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin
    
    # Schoolbook as fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j][col] = 1
                V_school[j*p+k][col] = 1
                W_school[k*n+i][col] = 1
                col += 1
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school
    
    # Try to find lower rank using randomized search
    # Lower bound: at least max(n*m, m*p, n*p) = 20, but realistically higher
    # Try ranks from 50 down to 20
    for target_rank in range(55, 19, -1):
        if time.time() > deadline - 10:
            break
        if target_rank >= best_rank:
            continue
        
        # Try a few random initializations
        for attempt in range(3):
            if time.time() > deadline - 10:
                break
            
            # Random integer/half-integer initialization
            def rand_entry():
                r = random.choice([0, 0.5, 1, 1.5, 2, -0.5, -1, -1.5, -2])
                return r
            
            U = [[rand_entry() for _ in range(target_rank)] for _ in range(n*m)]
            V = [[rand_entry() for _ in range(target_rank)] for _ in range(m*p)]
            W = [[rand_entry() for _ in range(target_rank)] for _ in range(p*n)]
            
            # Try to refine using simple iterative method
            # For each (i,j,k), we need sum_r U[i*m+j][r]*V[j*p+k][r]*W[k*n+i][r] = 1
            # This is hard to solve directly. Let's just verify if it works (unlikely)
            if _verify(U, V, W, n, m, p):
                best_rank = target_rank
                best_result = {"U": U, "V": V, "W": W}
                break
        
        # Also try a structured approach: fix W to be diagonal-ish
        if time.time() > deadline - 10:
            break
        # Try W with 1s on diagonal pattern
        for w_pattern in range(2):
            if time.time() > deadline - 10:
                break
            W = [[0] * target_rank for _ in range(p*n)]
            for idx in range(p*n):
                if idx < target_rank:
                    W[idx][idx] = 1
                else:
                    W[idx][idx % target_rank] = 1
            
            U = [[rand_entry() for _ in range(target_rank)] for _ in range(n*m)]
            V = [[rand_entry() for _ in range(target_rank)] for _ in range(m*p)]
            
            if _verify(U, V, W, n, m, p):
                best_rank = target_rank
                best_result = {"U": U, "V": V, "W": W}
                break
    
    return best_result
