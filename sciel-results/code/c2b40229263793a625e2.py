import time
import numpy as np
from fractions import Fraction
import random

DIMENSIONS = [2, 5, 5]
_DEADLINE = 145.0

def _remaining():
    return _DEADLINE - time.time()

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}

def _verify(n, m, p, U, V, W):
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if s != 1:
                    return False
    return True

def _try_structured_decomposition(n, m, p):
    """
    Try to find a rank-49 decomposition by exploiting structure.
    Key idea: take 49 schoolbook terms and try to "merge" the 50th term
    by adjusting entries to small rationals.
    """
    if time.time() > _DEADLINE - 20:
        return None
    
    target_rank = 49
    np.random.seed(42)
    random.seed(42)
    
    # Start with schoolbook, drop one term, try to compensate
    # Try dropping each of the 50 terms
    for drop_idx in range(50):
        if time.time() > _DEADLINE - 15:
            break
        
        U = [[0] * target_rank for _ in range(n*m)]
        V = [[0] * target_rank for _ in range(m*p)]
        W = [[0] * target_rank for _ in range(p*n)]
        
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if c == drop_idx:
                        c += 1
                        continue
                    U[i*m+j][c] = 1
                    V[j*p+k][c] = 1
                    W[k*n+i][c] = 1
                    c += 1
        
        # Now we need to find which term was dropped and try to recover it
        # by adjusting existing terms with small rational corrections
        di, dj, dk = drop_idx // (m*p), (drop_idx // p) % m, drop_idx % p
        
        # Try small rational adjustments to the dropped term's row/col
        # The dropped term contributes U[di*m+dj]*V[dj*p+dk]*W[dk*n+di] = 1
        # We need sum over c of adjustments to equal 1
        
        # Strategy: pick a few columns and adjust them
        found = False
        for trial in range(200):
            if time.time() > _DEADLINE - 15:
                break
            
            U_try = [row[:] for row in U]
            V_try = [row[:] for row in V]
            W_try = [row[:] for row in W]
            
            # Pick 1-3 random columns to adjust
            n_adjust = random.randint(1, 3)
            cols = random.sample(range(target_rank), n_adjust)
            
            # For each adjusted column, add small rational perturbations
            valid = True
            for col in cols:
                # Random small rationals from {-1, -1/2, 1/2, 1}
                delta_u = random.choice([0, 0, 0, 1, -1, 0.5, -0.5])
                delta_v = random.choice([0, 0, 0, 1, -1, 0.5, -0.5])
                delta_w = random.choice([0, 0, 0, 1, -1, 0.5, -0.5])
                
                U_try[di*m+dj][col] += delta_u
                V_try[dj*p+dk][col] += delta_v
                W_try[dk*n+di][col] += delta_w
            
            if _verify(n, m, p, U_try, V_try, W_try):
                return {"U": U_try, "V": V_try, "W": W_try}
    
    return None

def _try_greedy_cover(n, m, p):
    """
    Try to find rank-49 by removing one schoolbook term and compensating
    with adjusted entries in other terms.
    """
    if time.time() > _DEADLINE - 10:
        return None
    
    target_rank = 49
    np.random.seed(99)
    
    # For each term to remove, try to find small rational adjustments
    # that make the remaining 49 terms still satisfy all 50 equations
    for drop_idx in range(50):
        if time.time() > _DEADLINE - 8:
            break
        
        # Build base decomposition without term drop_idx
        U = [[0] * target_rank for _ in range(n*m)]
        V = [[0] * target_rank for _ in range(m*p)]
        W = [[0] * target_rank for _ in range(p*n)]
        
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if c == drop_idx:
                        c += 1
                        continue
                    U[i*m+j][c] = 1
                    V[j*p+k][c] = 1
                    W[k*n+i][c] = 1
                    c += 1
        
        # The missing equation is for (di, dj, dk)
        di, dj, dk = drop_idx // (m*p), (drop_idx // p) % m, drop_idx % p
        
        # We need: sum_c U[di*m+dj][c] * V[dj*p+dk][c] * W[dk*n+di][c] = 1
        # Currently this sum is 0 (since we removed the only term that covered it)
        
        # Try to add small rational values to make this work
        # while keeping all other equations satisfied
        
        # Strategy: modify at most 2-3 terms
        for trial in range(500):
            if time.time() > _DEADLINE - 8:
                break
            
            U_try = [row[:] for row in U]
            V_try = [row[:] for row in V]
            W_try = [row[:] for row in W]
            
            # Pick 1-2 columns to modify
            n_mod = random.randint(1, 2)
            cols = random.sample(range(target_rank), n_mod)
            
            for col in cols:
                # Add small values that don't break other equations
                # Only modify entries that don't appear in other equations
                # U[di*m+dj][col] appears in equations (di, dj, k) for all k
                # V[dj*p+dk][col] appears in equations (i, dj, dk) for all i
                # W[dk*n+di][col] appears in equations (i, j, dk) for all i,j
                
                # Safe modifications:
                # - If we modify U[di*m+dj][col], it affects (di,dj,k) for all k
                # - We need those to still sum to 1 for k != dk (they were 1 before)
                # - And sum to 1 for k = dk (they were 0, need to become 1)
                
                # Try: set U[di*m+dj][col] = a, V[dj*p+dk][col] = b, W[dk*n+di][col] = c
                # with a*b*c = 1, and for other k: a * V[dj*p+k][col] * W[k*n+di][col] = 0
                # This requires V[dj*p+k][col] = 0 or W[k*n+di][col] = 0 for k != dk
                
                # In our base, V[dj*p+k][col] = 1 only if (dj, k) was in the kept terms
                # and W[k*n+di][col] = 1 only if (k, di) was in the kept terms
                
                a = random.choice([1, -1, 2, -2, 0.5, -0.5])
                b = random.choice([1, -1, 2, -2, 0.5, -0.5])
                c_val = 1.0 / (a * b) if a * b != 0 else 0
                
                # Check if c_val is a "nice" rational
                if abs(c_val - round(c_val)) > 1e-9 and abs(c_val*2 - round(c_val*2)) > 1e-9:
                    continue
                
                U_try[di*m+dj][col] = a
                V_try[dj*p+dk][col] = b
                W_try[dk*n+di][col] = c_val
            
            if _verify(n, m, p, U_try, V_try, W_try):
                return {"U": U_try, "V": V_try, "W": W_try}
    
    return None

def _construct():
    n, m, p = DIMENSIONS
    best = _schoolbook(n, m, p)
    
    # Try structured decomposition
    result = _try_structured_decomposition(n, m, p)
    if result is not None:
        if _verify(n, m, p, result['U'], result['V'], result['W']):
            return result
    
    # Try greedy cover
    result = _try_greedy_cover(n, m, p)
    if result is not None:
        if _verify(n, m, p, result['U'], result['V'], result['W']):
            return result
    
    return best

def construct():
    try:
        return _construct()
    except Exception:
        n, m, p = DIMENSIONS
        return _schoolbook(n, m, p)
