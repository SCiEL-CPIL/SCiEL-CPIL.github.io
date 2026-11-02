import time
import random
import numpy as np
from fractions import Fraction

# EVOLVE-BLOCK-START
_START_TIME = time.time()
_DEADLINE = 140.0  # seconds, safety margin before 150s kill

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def _to_halfint(x):
    """Convert a Fraction or number to integer or half-integer (multiple of 0.5)."""
    if isinstance(x, Fraction):
        num, den = x.numerator, x.denominator
        if den == 1:
            return num
        if den == 2:
            return num / 2
        return None
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, float):
        if abs(x - round(x)) < 1e-9:
            return int(round(x))
        if abs(2*x - round(2*x)) < 1e-9:
            return round(2*x) / 2
        return None
    return None

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is exact and entries are integer/half-integer."""
    rank = len(U[0]) if U else 0
    # Check shapes
    if len(U) != n*m or len(V) != m*p or len(W) != p*n:
        return False
    # Check all entries are integer or half-integer
    for mat in [U, V, W]:
        for row in mat:
            for val in row:
                h = _to_halfint(val)
                if h is None:
                    return False
    # Verify tensor identity
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if isinstance(s, Fraction):
                    if s != 1:
                        return False
                else:
                    if abs(s - 1) > 1e-9:
                        return False
    return True

def _schoolbook():
    n, m, p = 2, 5, 6
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _try_rank(r, n, m, p, time_budget):
    """
    Try to find a rank-r decomposition over the rationals.
    """
    if time_budget() < 5:
        return None
    
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Number of equations
    num_eqs = n * m * p
    
    # We will try random U, V and solve for W.
    # Since the system is underdetermined for r >= 5, we can pick a particular solution.
    
    attempts = 100
    for _ in range(attempts):
        if time_budget() < 5:
            break
            
        # Random U, V with entries in {-1, 0, 1}
        U = np.random.randint(-1, 2, size=(nm, r))
        V = np.random.randint(-1, 2, size=(mp, r))
        
        # Build A matrix for W
        # A[eq, w_index] where eq is (i,j,k) and w_index is (k*n+i)*r + c
        # A[eq, (k*n+i)*r + c] = U[i*m+j, c] * V[j*p+k, c]
        A = np.zeros((num_eqs, pn * r), dtype=np.int64)
        b = np.ones(num_eqs, dtype=np.int64)
        
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for c in range(r):
                        A[eq_idx, (k*n+i)*r + c] = U[i*m+j, c] * V[j*p+k, c]
                    eq_idx += 1
        
        # Solve A W = b over rationals
        # Use numpy for speed, then check if solution is half-integer
        # Since A is sparse and underdetermined, we use lstsq as a heuristic to get a particular solution
        # But lstsq gives floating point. We need exactness.
        # Given the small size, we can use Fraction-based Gaussian elimination if needed, 
        # but it might be slow. 
        # Let's try to find a solution with small integers.
        
        # If A has full row rank (which is likely since num_eqs=60 and cols=12r>=360),
        # we can pick 60 pivot columns and solve.
        
        # Simplified approach: 
        # If we set most W to 0, can we solve the rest?
        # This is equivalent to finding a subset of columns of A that are linearly independent 
        # and solve for the corresponding W entries.
        
        # Let's just use a randomized search for W directly? No, W is determined by U,V.
        
        # Let's try to solve using numpy linalg.lstsq and check if the result is close to half-integers.
        # If it is, we round and verify exactly.
        
        try:
            # Solve the underdetermined system. 
            # We can use the pseudo-inverse.
            W_flat, residuals, rank, singular_values = np.linalg.lstsq(A, b, rcond=None)
            
            # Check if W_flat entries are close to half-integers
            W_check = W_flat.reshape(pn, r)
            is_halfint = True
            W_rounded = np.zeros_like(W_check)
            for idx in range(W_check.size):
                val = W_check.flat[idx]
                # Check if val is close to k/2
                k = round(2 * val)
                if abs(val - k/2) > 1e-6:
                    is_halfint = False
                    break
                W_rounded.flat[idx] = k/2
            
            if not is_halfint:
                continue
                
            # Convert to list of lists
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W_rounded.tolist()
            
            # Verify exactly
            if _verify(U_list, V_list, W_list, n, m, p):
                return {"U": U_list, "V": V_list, "W": W_list}
                
        except Exception:
            continue
            
    return None

def construct():
    n, m, p = 2, 5, 6
    schoolbook_rank = n * m * p  # 60
    
    # Start with schoolbook as fallback
    best = _schoolbook()
    best_rank = schoolbook_rank
    
    # Try to find lower rank decompositions
    # Lower bound: 30
    for r in range(30, best_rank):
        if _time_left() < 10:
            break
        
        result = _try_rank(r, n, m, p, _time_left)
        if result is not None:
            best = result
            best_rank = r
            # If we find a valid one, we can stop if we want the minimum?
            # The loop goes up, so the first one found is the minimum in this range.
            # But we started from 30, so if we find one at 30, it's the min.
            # If we find one at 59, it's not the min, but better than 60.
            # We should continue to check lower ranks? No, we are iterating 30, 31, ...
            # So if we find one at r, it is the smallest found so far.
            # We can return immediately? 
            # If we find rank 30, that's likely the global min.
            # If we find rank 59, there might be a rank 30 solution we missed.
            # But given the difficulty, finding any r < 60 is a win.
            # Let's return the first one found to save time for verification/other tasks?
            # The task asks for "rank-minimal". 
            # If we find 59, we should try to find 30.
            # But the search is random. 
            # Let's just return the first valid one found, as finding lower is exponentially harder.
            return best
        
    return best
# EVOLVE-BLOCK-END
