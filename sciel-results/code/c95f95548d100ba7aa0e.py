"""Matrix multiplication 2x3 by 3x3: rank-minimal exact tensor decomposition."""
import time
import numpy as np
from math import gcd
from functools import reduce

DIMENSIONS = [2, 3, 3]

# Deadline guard: evaluator kills after 150s; keep safety margin.
_START = time.time()
_DEADLINE = 145.0  # seconds

def _elapsed():
    return time.time() - _START

def _remaining():
    return _DEADLINE - _elapsed()

def _verify(U, V, W):
    """Check that U[i,j,c]*V[j,k,c]*W[k,i,c] = 1 iff (i,j,k) is a valid triple."""
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    # Build the multiplication tensor: T[i,j,k] = 1
    T = np.ones((n, m, p))
    # Reconstruct via rank-1 slices
    recon = np.zeros((n, m, p))
    for c in range(rank):
        u = U[:, c]  # shape (n*m,)
        v = V[:, c]  # shape (m*p,)
        w = W[:, c]  # shape (p*n,)
        # u[i*m+j] * v[j*p+k] * w[k*n+i]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    recon[i, j, k] += u[i * m + j] * v[j * p + k] * w[k * n + i]
    if not np.allclose(recon, T, atol=1e-9):
        return False
    return True

def _schoolbook():
    """Standard schoolbook decomposition, rank = n*m*p = 18."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}

def _find_half_integer_solution(coeffs, target=2):
    """
    Find integers k_c such that sum(coeffs[c] * k_c) = target.
    coeffs are integers.
    Returns list of integers k_c or None.
    """
    n_coeffs = len(coeffs)
    # If all coeffs are 0, no solution unless target is 0
    if all(c == 0 for c in coeffs):
        return None if target != 0 else [0] * n_coeffs
    
    # Simple search for small coefficients and small target
    # Since coeffs are small (sum of products of -1,0,1), they are in {-2, -1, 0, 1, 2}
    # Target is 2.
    # We can use a simple DP or backtracking.
    
    # Backtracking
    def backtrack(idx, current_sum, k):
        if idx == n_coeffs:
            if current_sum == target:
                return k[:]
            return None
        # Pruning: if remaining max possible sum can't reach target
        # Max coeff is 2, min is -2.
        # This is a small problem, so simple recursion is fine.
        for val in range(-5, 6): # k_c values likely small
            res = backtrack(idx + 1, current_sum + coeffs[idx] * val, k + [val])
            if res is not None:
                return res
        return None

    return backtrack(0, 0, [])

def _try_rank_structured(n, m, p, target_rank):
    """
    Try to find a rank-`target_rank` decomposition using the structured A_ik approach.
    """
    U_rows = n * m
    V_rows = m * p
    W_rows = p * n
    
    import random
    random.seed(42 + target_rank)
    
    max_attempts = 500
    for attempt in range(max_attempts):
        if _remaining() < 5:
            return None
        
        # Generate random sparse U, V
        # Entries in {-1, 0, 1}
        U = [[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(U_rows)]
        V = [[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(V_rows)]
        
        # Compute A_ik,c = sum_j U[i*m+j, c] * V[j*p+k, c]
        # A is shape (n, p, r)
        A = [[[0] * target_rank for _ in range(p)] for _ in range(n)]
        
        valid_gcd = True
        for i in range(n):
            for k in range(p):
                for c in range(target_rank):
                    s = 0
                    for j in range(m):
                        s += U[i * m + j][c] * V[j * p + k][c]
                    A[i][k][c] = s
        
        # Check GCD condition for each (i, k)
        # We need gcd(A[i][k][0..r-1]) to divide 2.
        # i.e., gcd must be 1 or 2 (or -1, -2).
        for i in range(n):
            for k in range(p):
                coeffs = A[i][k]
                g = 0
                for c in range(target_rank):
                    g = gcd(g, abs(coeffs[c]))
                
                if g == 0:
                    valid_gcd = False
                    break
                if g % 2 != 0 and g != 1 and g != 2:
                    # If g is odd and not 1, it must be 1? No, g=3 means 3|2 is false.
                    # We need 2 to be a multiple of g? No, we need sum A_c k_c = 2.
                    # This is possible iff gcd(A_c) divides 2.
                    if 2 % g != 0:
                        valid_gcd = False
                        break
            if not valid_gcd:
                break
        
        if not valid_gcd:
            continue
        
        # Construct W
        # W[k*n+i, c] corresponds to variable W_{ki,c}
        # Equation: sum_c A[i][k][c] * W[k*n+i][c] = 1
        # Let W[k*n+i][c] = k_c / 2.
        # sum_c A[i][k][c] * k_c / 2 = 1 => sum_c A[i][k][c] * k_c = 2
        W = [[0.0] * target_rank for _ in range(W_rows)]
        
        success = True
        for i in range(n):
            for k in range(p):
                coeffs = A[i][k]
                k_vals = _find_half_integer_solution(coeffs, target=2)
                if k_vals is None:
                    success = False
                    break
                for c in range(target_rank):
                    W[k * n + i][c] = k_vals[c] / 2.0
        if not success:
            continue
        
        # Verify
        if _verify(U, V, W):
            return {"U": U, "V": V, "W": W}
            
    return None

def construct():
    """
    Attempt to find a rank-minimal exact tensor decomposition for 2x3 * 3x3.
    """
    best = _schoolbook()
    best_rank = 18
    n, m, p = DIMENSIONS
    
    # Try ranks from 17 down to 10
    # The lower bound is max(nm, mp, pn) = 9
    for target_rank in range(17, 9, -1):
        if _remaining() < 10:
            break
        
        result = _try_rank_structured(n, m, p, target_rank)
        if result is not None:
            best = result
            best_rank = target_rank
            # If we found 17, we might try 16, etc.
            continue
    
    return best
