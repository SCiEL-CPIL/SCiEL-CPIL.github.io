"""Find a rank-minimal exact tensor decomposition for 2x3 * 3x4 matmul.
Output factor follows the official trace convention W[k*n+i, r].
"""
import time
import random
import itertools

DIMENSIONS = [2, 3, 4]
START = time.time()
TIME_LIMIT = 135.0  # safety margin before 150s kill


def _time_left():
    return TIME_LIMIT - (time.time() - START)


def _verify(U, V, W, n, m, p):
    """Verify exact decomposition: sum_r U[i*m+j,r]*V[j'*p+k,r]*W[i'*n+k',r] = delta(i,i')*delta(j,j')*delta(k,k')"""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    for ip in range(n):
                        for kp in range(p):
                            expected = 1 if (i == ip and j == jp and k == kp) else 0
                            s = 0
                            for r in range(rank):
                                s += U[i * m + j][r] * V[jp * p + k][r] * W[kp * n + ip][r]
                            if abs(s - expected) > 1e-9:
                                return False
    return True


def _schoolbook(n, m, p):
    """Standard rank n*m*p decomposition."""
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
    return U, V, W


def _try_structured_search(n, m, p, max_rank, min_rank):
    """Try to find a lower-rank decomposition using structured approaches."""
    best = None
    best_rank = max_rank
    
    # Try various ranks from min_rank up to max_rank-1
    for r in range(min_rank, max_rank):
        if _time_left() < 5:
            break
        
        # Try multiple random structured initializations
        for attempt in range(200):
            if _time_left() < 3:
                break
            
            # Use small integer/half-integer values
            values = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]
            
            U = [[random.choice(values) for _ in range(r)] for _ in range(n * m)]
            V = [[random.choice(values) for _ in range(r)] for _ in range(m * p)]
            W = [[random.choice(values) for _ in range(r)] for _ in range(p * n)]
            
            if _verify(U, V, W, n, m, p):
                best = (U, V, W)
                best_rank = r
                break  # Found valid decomposition for this rank, move to next lower rank
        
        if best is not None and best_rank == r:
            # Found a valid decomposition at rank r
            return best
    
    return best


def construct():
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    best_rank = n * m * p  # 24
    
    # Lower bound: max(n*m, m*p, p*n) = max(6, 12, 8) = 12
    lower_bound = max(n * m, m * p, p * n)
    
    # Try to find lower-rank decompositions
    result = _try_structured_search(n, m, p, best_rank, lower_bound)
    
    if result is not None:
        U, V, W = result
        best_rank = len(U[0])
    
    return {"U": U, "V": V, "W": W}
