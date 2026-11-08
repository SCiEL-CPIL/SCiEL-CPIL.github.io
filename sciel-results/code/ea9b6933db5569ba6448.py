"""
Matrix multiplication tensor decomposition for 3x4 by 4x4.
Strategy: try to find a lower-rank exact decomposition using a
finite-field search over F_2 to identify sparse support structures,
then lift to integers. Fall back to schoolbook (rank 48) if no
improvement is found within the time budget.
"""
import time
import random
import itertools
from collections import defaultdict

DIMENSIONS = [3, 4, 4]
START_TIME = time.time()
TIME_LIMIT = 120.0  # seconds
SAFETY_MARGIN = 10.0  # seconds before hard limit


def _elapsed():
    return time.time() - START_TIME


def _remaining():
    return TIME_LIMIT - SAFETY_MARGIN - _elapsed()


def _check_time():
    if _elapsed() > TIME_LIMIT - SAFETY_MARGIN:
        return False
    return True


def _schoolbook():
    """Return the schoolbook decomposition of rank n*m*p = 48."""
    n, m, p = DIMENSIONS
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


def _verify(U, V, W, n, m, p, rank):
    """
    Verify that the decomposition is exact:
    For all i in [0,n), k in [0,p):
        sum_j A[i,j] * B[j,k] = C[i,k]
    In tensor form: for each (i,j,k), the coefficient of A[i,j]*B[j,k] in C[i,k]
    must equal sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] ... 

    Actually the convention is:
    C[i,k] = sum_j A[i,j] B[j,k]
    And the decomposition says:
    C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j]) * (sum_? ...)
    
    Wait, let me re-read the convention. The output factor follows W[k*n+i, r].
    
    The standard tensor decomposition for matrix mult M_{n,m,p}:
    M_{n,m,p} = sum_r u_r ⊗ v_r ⊗ w_r
    where u_r ∈ F^{n*m}, v_r ∈ F^{m*p}, w_r ∈ F^{p*n}
    
    The multiplication tensor T[i,j,k] (coefficient of A[i,j]*B[j,k] in C[i,k])
    equals 1 if valid, 0 otherwise.
    
    T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    For matrix multiplication, T[i,j,k] = 1 for all valid (i,j,k) in the multiplication,
    specifically T[i,j,k] = 1 when the product A[i,j]*B[j,k] contributes to C[i,k].
    
    Actually for standard matrix multiplication C = AB where A is n×m, B is m×p, C is n×p:
    C[i,k] = sum_{j=0}^{m-1} A[i,j] * B[j,k]
    
    So the tensor T[i,j,k] = 1 for all i ∈ [0,n), j ∈ [0,m), k ∈ [0,p).
    
    And the decomposition requires:
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1  for all i,j,k
    
    Let's verify this.
    """
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if s != 1:
                    return False
    return True


def _random_search_f2(n, m, p, max_rank, trials):
    """
    Try to find a decomposition over F_2 with rank < n*m*p.
    Over F_2, we need:
    sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 (mod 2) for all i,j,k
    
    This is equivalent to finding a rank-r decomposition of the all-ones tensor
    over F_2.
    
    We use a randomized approach: randomly assign U, V and try to solve for W.
    """
    total_entries = n * m * p  # = 48
    
    for attempt in range(trials):
        if not _check_time():
            return None
        
        r = max_rank
        # Randomly initialize U and V over F_2
        U = [[random.randint(0, 1) for _ in range(r)] for _ in range(n * m)]
        V = [[random.randint(0, 1) for _ in range(r)] for _ in range(m * p)]
        
        # For each (i,k), we need:
        # sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        # Let's define for each (i,k): the equation is linear in W[k*n+i, :]
        # sum_r (sum_j U[i*m+j,r] * V[j*p+k,r]) * W[k*n+i,r] = 1
        # Let M_{i,k,r} = sum_j U[i*m+j,r] * V[j*p+k,r] (mod 2)
        # Then we need: for each (i,k): sum_r M_{i,k,r} * W[k*n+i,r] = 1
        
        # Build the system: for each (i,k) pair, we have one equation in W[k*n+i, 0..r-1]
        # But W[k*n+i, :] is shared across different i for the same k!
        # Actually W is indexed by [k*n+i, r], so for fixed k, we have n equations
        # involving W[k*n+0, :], ..., W[k*n+n-1, :]
        
        # For fixed k: we have n equations:
        # For i=0: sum_r M_{0,k,r} * W[k*n+0, r] = 1
        # For i=1: sum_r M_{1,k,r} * W[k*n+1, r] = 1
        # ...
        
        # These are n independent equations, each involving a different row of W.
        # So we can solve them independently!
        
        W = [[0] * r for _ in range(p * n)]
        valid = True
        for k in range(p):
            for i in range(n):
                # Compute M_{i,k,r} for all r
                M_row = [0] * r
                for j in range(m):
                    for rr in range(r):
                        M_row[rr] = (M_row[rr] + U[i * m + j][rr] * V[j * p + k][rr]) % 2
                # Now we need: sum_r M_row[r] * W[k*n+i, r] = 1
                # This is one equation with r unknowns. We need to find any solution.
                # If M_row is all zeros, no solution.
                if all(x == 0 for x in M_row):
                    valid = False
                    break
                # Find a solution: pick a column with M_row[r]=1 and set W[k*n+i,r]=1, others 0
                # But this might not work if we need consistency... actually each (i,k)
                # has its own row of W, so they're independent.
                # Just find one r where M_row[r] = 1 and set that to 1.
                found = False
                for rr in range(r):
                    if M_row[rr] == 1:
                        W[k * n + i][rr] = 1
                        found = True
                        break
                if not found:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            # Verify over F_2
            ok = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0
                        for rr in range(r):
                            s = (s + U[i * m + j][rr] * V[j * p + k][rr] * W[k * n + i][rr]) % 2
                        if s != 1:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            if ok:
                # Lift to integers (0 and 1 are already integers)
                return {"U": [row[:] for row in U], "V": [row[:] for row in V], "W": [row[:] for row in W]}
    
    return None


def _try_rank(r, n, m, p, trials_per_rank):
    """Try to find a decomposition of rank r."""
    if r >= n * m * p:
        return _schoolbook()
    result = _random_search_f2(n, m, p, r, trials_per_rank)
    if result is not None:
        # Verify over integers
        U, V, W = result["U"], result["V"], result["W"]
        if _verify(U, V, W, n, m, p, r):
            return result
    return None


def construct():
    n, m, p = DIMENSIONS
    base_rank = n * m * p  # 48
    
    # Try to find lower rank decompositions
    # Start from rank 1 and go up, but that's too slow.
    # Try a few promising ranks near the base.
    
    # First, try a quick random search for rank 47, 46, 45
    for r in range(base_rank - 1, max(10, base_rank - 15), -1):
        if not _check_time():
            break
        result = _try_rank(r, n, m, p, trials_per_rank=200)
        if result is not None:
            return result
    
    # If we still have time, try some lower ranks with more trials
    if _remaining() > 30:
        for r in range(40, 20, -5):
            if not _check_time():
                break
            result = _try_rank(r, n, m, p, trials_per_rank=500)
            if result is not None:
                return result
    
    # Fall back to schoolbook
    return _schoolbook()
